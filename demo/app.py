"""Research website and a public, bounded translation service.

One inference worker per process. Each browser has its own short-lived pose buffer.
There is no upload archive, audio archive, transcript database, or silent sample fallback.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

LOG = logging.getLogger('auslan.service')
STATIC = Path(__file__).parent / 'static'
ID_RE = re.compile(r'^[a-zA-Z0-9_-]{16,80}$')


@dataclass
class Settings:
    mode: str = 'showcase'
    production: bool = False
    public_origin: str = ''
    models: tuple = ('single', 'ensemble')
    coverage: float = 0.5   # target answer rate on validation; thresholds per model live in service.MODELS
    rgb_device: str = 'cpu'
    detector_path: str = ''
    detector_sha256: str = ''
    pose_path: str = ''
    pose_sha256: str = ''
    device: str = 'cpu'
    threads: int = 2
    max_sessions: int = 8
    sessions_per_ip: int = 2
    requests_per_minute: int = 12
    # Raw camera frames: 10 s at up to 30 fps. The 25 fps grid then holds <= 254 frames (model cap 256).
    max_frames: int = 300
    session_timeout: float = 90

    @classmethod
    def from_env(cls):
        env = os.environ
        return cls(mode=env.get('DEMO_MODE', 'showcase'),
                   production=env.get('DEMO_ENV') == 'production',
                   public_origin=env.get('DEMO_PUBLIC_ORIGIN', '').rstrip('/'),
                   models=tuple(m for m in env.get('DEMO_MODELS', 'single,ensemble').split(',') if m),
                   coverage=float(env.get('DEMO_COVERAGE', '0.5')), rgb_device=env.get('DEMO_RGB_DEVICE', 'cpu'),
                   detector_path=env.get('DEMO_DETECTOR', ''), detector_sha256=env.get('DEMO_DETECTOR_SHA256', ''),
                   pose_path=env.get('DEMO_POSE', ''), pose_sha256=env.get('DEMO_POSE_SHA256', ''),
                   device=env.get('DEMO_DEVICE', 'cpu'), threads=int(env.get('DEMO_THREADS', '2')))

    def validate(self):
        if self.mode not in {'showcase', 'model', 'sample'}: raise ValueError('Invalid DEMO_MODE.')
        # Validated answer rates only (SELECTIVE_PREDICTION_2026-09-09: 50-90%, never below 50%).
        if self.coverage not in (0.5, 0.75, 0.9): raise ValueError('Coverage must be 0.5, 0.75 or 0.9.')
        if not self.models or not set(self.models) <= {'single', 'ensemble'} or len(set(self.models)) != len(self.models):
            raise ValueError('Models must be drawn from single, ensemble.')
        if not 1 <= self.threads <= 16 or not 2 <= self.max_frames <= 300:
            raise ValueError('Invalid resource limits.')
        # MPS is slower than CPU for the translator at batch 1 (pose_throughput_m3max_2026-09-15.json).
        if self.device not in {'cpu', 'cuda'}: raise ValueError('Unsupported inference device.')
        if self.rgb_device not in {'cpu', 'cuda', 'mps'}: raise ValueError('Unsupported RGB feature device.')
        if self.production:
            origin = urlsplit(self.public_origin)
            if origin.scheme != 'https' or not origin.netloc or origin.path or origin.query or origin.fragment or origin.username:
                raise ValueError('Production requires DEMO_PUBLIC_ORIGIN=https://your-domain.')
            if self.mode == 'sample': raise ValueError('Sample output is not allowed in production.')
            if self.mode == 'model' and not all([self.detector_path, self.detector_sha256, self.pose_path, self.pose_sha256]):
                raise ValueError('Production inference requires explicit pose assets with SHA-256 checksums.')


@dataclass
class Session:
    ip: str
    touched: float = field(default_factory=time.monotonic)
    created: float = field(default_factory=time.monotonic)
    channels: set = field(default_factory=set)
    recording: bool = False
    request_id: str | None = None
    model: str = ''
    keypoints: list = field(default_factory=list)
    times: list = field(default_factory=list)
    # ponytail: ensemble mode keeps each JPEG (<=1.35 MB) until Finish for the RGB features; spool to disk if memory binds.
    frames: list = field(default_factory=list)
    track: dict = field(default_factory=dict)
    started: float = 0
    last_frame: float = 0
    last_capture_at: float = -100
    translations: deque = field(default_factory=deque)
    messages: deque = field(default_factory=deque)
    task: asyncio.Task | None = None

    def clear(self):
        self.recording = False
        self.request_id = None
        self.keypoints.clear(); self.times.clear(); self.frames.clear(); self.track = {}


class CapacityError(Exception): pass


class Runtime:
    def __init__(self, settings, service=None):
        self.settings = settings
        self.service = service
        self.sessions: dict[str, Session] = {}
        self.attempts: dict[str, deque] = {}
        self.starts: dict[str, deque] = {}
        self.jobs = asyncio.Lock()
        self.tasks = set()
        self.completed = 0
        self.accepted = 0

    def allow_rate(self, counters, ip, limit):
        now = time.monotonic()
        # Expire IP counters instead of growing a permanent IP-address log.
        for key in list(counters):
            if not counters[key] or now - counters[key][-1] >= 60: del counters[key]
        if ip not in counters and len(counters) >= 1024: return False
        q = counters.setdefault(ip, deque())
        while q and now - q[0] >= 60: q.popleft()
        if len(q) >= limit: return False
        q.append(now)
        return True

    def attach(self, session_id, ip, channel):
        if not ID_RE.fullmatch(session_id): raise CapacityError('Invalid session.')
        s = self.sessions.get(session_id)
        if s:
            if s.ip != ip or channel in s.channels: raise CapacityError('Session already connected.')
        else:
            if len(self.sessions) >= self.settings.max_sessions: raise CapacityError('The translator is busy. Please try again shortly.')
            if sum(v.ip == ip for v in self.sessions.values()) >= self.settings.sessions_per_ip:
                raise CapacityError('Close another translator tab and try again.')
            s = self.sessions[session_id] = Session(ip)
        s.channels.add(channel)
        return s

    def detach(self, session_id, channel):
        s = self.sessions.get(session_id)
        if s:
            s.channels.discard(channel)
            s.clear()
            if not s.channels: self.sessions.pop(session_id, None)

    async def work(self, func, *args):
        # No timeout: an ensemble Finish holds the worker for seconds, and a 2 s give-up discarded other visitors'
        # recordings. Waiters are bounded: one call per pose socket at a time, one result task per session.
        await self.jobs.acquire()
        try:
            task = asyncio.create_task(asyncio.to_thread(func, *args))
            try: return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A Python thread cannot be cancelled. Keep the worker reserved until it exits.
                await task
                raise
        finally:
            self.jobs.release()


def create_app(settings=None, service=None):
    config = settings or Settings.from_env()
    config.validate()
    runtime = Runtime(config, service)

    @asynccontextmanager
    async def lifespan(app):
        if runtime.service is None and config.mode != 'showcase':
            from .service import ModelService, PreviewService
            runtime.service = await asyncio.to_thread(PreviewService if config.mode == 'sample' else lambda: ModelService(config))
        yield
        for session in runtime.sessions.values(): session.clear()
        if runtime.tasks: await asyncio.gather(*runtime.tasks, return_exceptions=True)
        runtime.sessions.clear()

    app = FastAPI(title='Auslan Research', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Permissions-Policy'] = 'camera=(self), microphone=(), geolocation=()'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith(('/api/', '/health', '/readyz')) else 'no-cache'
        return response

    @app.get('/')
    async def index(): return FileResponse(STATIC / 'index.html')

    @app.get('/studio')
    async def studio(): return FileResponse(STATIC / 'studio.html')

    @app.get('/privacy')
    async def privacy(): return FileResponse(STATIC / 'privacy.html')

    @app.get('/terms')
    async def terms(): return FileResponse(STATIC / 'terms.html')

    @app.get('/research-notes')
    async def research_notes(): return FileResponse(STATIC / 'research-notes.html')

    @app.get('/credits')
    async def credits(): return FileResponse(STATIC / 'credits.html')

    @app.get('/accessibility')
    async def accessibility(): return FileResponse(STATIC / 'accessibility.html')

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'mode': config.mode, 'translator': runtime.service is not None,
                'models': getattr(runtime.service, 'models', {}), 'coverage_target': config.coverage,
                'max_frames': config.max_frames, 'capture_fps_target': 25,
                'answered': [runtime.accepted, runtime.completed], 'speech': 'device-local'}

    @app.get('/readyz')
    async def readiness():
        ready = config.mode == 'showcase' or runtime.service is not None
        return JSONResponse({'ready': ready, 'mode': config.mode}, status_code=200 if ready else 503)

    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    def origin_allowed(ws):
        origin = ws.headers.get('origin', '')
        if config.public_origin: return origin == config.public_origin
        expected = f"{ws.url.scheme.replace('ws', 'http')}://{ws.headers.get('host', '')}"
        return origin == expected

    async def result(ws, s, request_id, model, points, times, frames):
        try:
            answer = await runtime.work(runtime.service.translate, model, points, times, frames)
            if s.request_id != request_id or not s.channels: return
            info = runtime.service.models[model]
            score = float(answer.confidence)
            empty = not answer.text.strip()
            accepted = not empty and math.isfinite(score) and score <= info['threshold']
            span = max(times[-1] - times[0], 1)
            await ws.send_json({'type': 'result', 'request_id': request_id,
                                'text': answer.text if accepted else '',
                                'confidence': score if math.isfinite(score) else None,
                                'threshold': info['threshold'], 'coverage_target': info['coverage'],
                                'accepted': accepted,
                                'reason': 'ready' if accepted else 'declined: empty' if empty else 'declined: low confidence',
                                'mode': config.mode, 'model': model, 'model_label': info['label'],
                                'capture_fps': round((len(times) - 1) * 1000 / span, 1), 'frames': len(times),
                                'frames_used': answer.frames_used, 'translate_ms': round(answer.translate_ms, 1),
                                'rgb_ms': round(answer.rgb_ms, 1)})
            runtime.completed += 1
            runtime.accepted += accepted
        except (ValueError, CapacityError) as exc:
            if s.request_id == request_id:
                await ws.send_json({'error': str(exc), 'request_id': request_id})
        except Exception:
            LOG.exception('Translation failed; session content omitted')
            if s.request_id == request_id:
                try: await ws.send_json({'error': 'Translation could not be completed. Please try again.', 'request_id': request_id})
                except Exception: pass
        finally:
            points.clear(); times.clear(); frames.clear()

    async def socket(ws, channel):
        if not origin_allowed(ws): await ws.close(code=1008); return
        if runtime.service is None: await ws.close(code=1013); return
        ip = ws.client.host if ws.client else 'unknown'
        if not runtime.allow_rate(runtime.attempts, ip, 30): await ws.close(code=1013); return
        session_id = ws.query_params.get('session_id', '')
        try: s = runtime.attach(session_id, ip, channel)
        except CapacityError: await ws.close(code=1013); return
        await ws.accept()
        try:
            while True:
                try: message = await asyncio.wait_for(ws.receive(), timeout=config.session_timeout)
                except asyncio.TimeoutError: await ws.close(code=1001); break
                if message['type'] == 'websocket.disconnect': raise WebSocketDisconnect(message.get('code', 1000))
                raw = message.get('text')
                if raw is None: await ws.close(code=1003); break
                if time.monotonic() - s.created > 600:
                    await ws.close(code=1001); break
                s.touched = time.monotonic()
                while s.messages and s.touched - s.messages[0] >= 10: s.messages.popleft()
                if len(s.messages) >= 500: await ws.close(code=1008); break
                s.messages.append(s.touched)
                if len(raw) > 1_350_000: await ws.close(code=1009); break
                msg = {}
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict): raise ValueError('Invalid message.')
                    if msg.get('type') == 'ping': await ws.send_json({'type': 'pong'}); continue
                    request_id = msg.get('request_id', '')
                    if not isinstance(request_id, str) or not ID_RE.fullmatch(request_id): raise ValueError('Invalid request.')
                    kind = msg.get('type')
                    if channel == 'pose' and kind == 'frame':
                        if s.request_id != request_id or not s.recording:
                            await ws.send_json({'type': 'frame_ignored', 'request_id': request_id}); continue
                        if len(s.keypoints) >= config.max_frames or time.monotonic() - s.started > 120:
                            raise ValueError('Recording limit reached. Please try a shorter utterance.')
                        now = time.monotonic()
                        captured_at = msg.get('captured_at')
                        if not isinstance(captured_at, (int, float)) or isinstance(captured_at, bool) or not math.isfinite(captured_at) or not 0 <= captured_at <= 10_150 or captured_at - s.last_capture_at < 20:
                            raise ValueError('Invalid camera timing. Please start a new utterance.')
                        s.last_capture_at = captured_at
                        s.last_frame = now
                        from .service import decode_frame
                        data = msg.get('data')
                        frame = decode_frame(data)
                        kp, people = await runtime.work(runtime.service.extract, frame, s.track)
                        del frame
                        if s.request_id != request_id or not s.recording:
                            await ws.send_json({'type': 'frame_ignored', 'request_id': request_id}); continue
                        s.keypoints.append(kp); s.times.append(captured_at)
                        if runtime.service.models[s.model]['needs_frames']: s.frames.append(data)
                        await ws.send_json({'type': 'pose', 'request_id': request_id, 'has_person': people >= 1,
                                            'person_count': people, 'buffer_len': len(s.keypoints),
                                            'pose_ms': round((time.monotonic() - now) * 1000, 1)})
                    elif channel == 'translate' and kind == 'start_recording':
                        now = time.monotonic()
                        if s.task and not s.task.done(): raise CapacityError('Your previous utterance is still finishing. Please wait a moment.')
                        model = msg.get('model', next(iter(runtime.service.models)))
                        if model not in runtime.service.models: raise ValueError('That translation model is not available.')
                        if not runtime.allow_rate(runtime.starts, ip, config.requests_per_minute):
                            raise ValueError('Please wait a minute before starting another utterance.')
                        s.clear(); s.recording = True; s.request_id = request_id; s.model = model
                        s.started = now; s.last_frame = 0; s.last_capture_at = -100
                        await ws.send_json({'type': 'recording_started', 'request_id': request_id})
                    elif channel == 'translate' and kind == 'stop_recording':
                        if s.request_id == request_id: s.clear()
                        await ws.send_json({'type': 'recording_stopped', 'request_id': request_id})
                    elif channel == 'translate' and kind == 'translate':
                        if s.request_id != request_id or not s.recording: raise ValueError('Start a new utterance before translating.')
                        s.recording = False
                        if len(s.keypoints) < 2: s.clear(); raise ValueError('Too few frames were captured. Please try again.')
                        points, times, frames = s.keypoints, s.times, s.frames
                        s.keypoints, s.times, s.frames = [], [], []
                        s.track = {}
                        s.task = asyncio.create_task(result(ws, s, request_id, s.model, points, times, frames))
                        runtime.tasks.add(s.task); s.task.add_done_callback(runtime.tasks.discard)
                    else: raise ValueError('Unsupported message.')
                except (ValueError, CapacityError) as exc:
                    await ws.send_json({'error': str(exc), 'request_id': msg.get('request_id') if isinstance(msg, dict) else None})
                except Exception:
                    LOG.exception('Capture failed; session content omitted')
                    s.clear()
                    await ws.send_json({'error': 'Capture could not be completed. Please try again.'})
        except WebSocketDisconnect: pass
        finally: runtime.detach(session_id, channel)

    @app.websocket('/ws/pose')
    async def pose(ws: WebSocket): await socket(ws, 'pose')

    @app.websocket('/ws/translate')
    async def translate(ws: WebSocket): await socket(ws, 'translate')

    return app


app = create_app()
