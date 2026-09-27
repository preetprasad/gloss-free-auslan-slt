"""Public service boundaries; no research dataset, weights or camera needed."""
import base64
from contextlib import ExitStack
import io
import json
import sys
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from demo.app import Settings, Runtime, CapacityError, create_app
from demo.service import MODELS, ModelService, check_asset, decode_frame, grid_indices

STATIC = Path(__file__).resolve().parent.parent / 'static'

ORIGIN = {'origin': 'http://testserver'}
SID = 'session-00000000001'
RID = 'utterance-000000001'
RID2 = 'utterance-000000002'


class FakeService:
    models = {'single': dict(label='Single', threshold=0.75, coverage=0.5, needs_frames=False),
              'ensemble': dict(label='Ensemble', threshold=0.90, coverage=0.5, needs_frames=True)}

    def __init__(self):
        self.calls, self.confidence, self.text = [], 0.5, 'Test output.'

    def extract(self, frame, track=None):
        if track is not None: track['mid'] = 'seen'
        return np.full((133, 3), float(frame.mean()), np.float32), 1

    def translate(self, model, points, times, frames=None):
        # Keep the list objects too: app.result() must clear them after use.
        self.calls.append(SimpleNamespace(model=model, points=np.stack(points), times=list(times),
                                          frames=list(frames or []), lists=(points, times, frames)))
        return SimpleNamespace(text=self.text, confidence=self.confidence, translate_ms=1.0, rgb_ms=0.0,
                               frames_used=len(points))


def jpeg(value=100, size=(20, 20), color=None):
    out = io.BytesIO()
    Image.new('RGB', size, color=color or (value, value, value)).save(out, 'JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()


def connect(stack, client, channel, sid=SID):
    return stack.enter_context(client.websocket_connect(f'/ws/{channel}?session_id={sid}', headers=ORIGIN))


def start(ws, rid=RID, model=None):
    ws.send_json({'type': 'start_recording', 'request_id': rid, **({'model': model} if model else {})})
    assert ws.receive_json()['type'] == 'recording_started'


def frame(ws, value=100, at=0, rid=RID):
    ws.send_json({'type': 'frame', 'request_id': rid, 'captured_at': at, 'data': jpeg(value)})
    return ws.receive_json()


def finish(ws, rid=RID):
    ws.send_json({'type': 'translate', 'request_id': rid})
    result = ws.receive_json()
    ws.send_json({'type': 'ping'})  # barrier: the result task has finished its cleanup before pong
    assert ws.receive_json()['type'] == 'pong'
    return result


def utterance(client, stack, model=None, times=(0, 40)):
    tr = connect(stack, client, 'translate'); po = connect(stack, client, 'pose')
    start(tr, model=model)
    for i, at in enumerate(times): frame(po, value=10 + 90 * i, at=at)
    return tr, po


def test_pages_security_and_no_model_fallback():
    with TestClient(create_app(Settings())) as client:
        for path in ['/', '/studio', '/privacy', '/terms', '/research-notes', '/credits', '/accessibility',
                     '/static/app.js', '/static/research.css', '/static/documents.css', '/static/research-notes.js']:
            response = client.get(path)
            assert response.status_code == 200
            assert "frame-ancestors 'none'" in response.headers['content-security-policy']
            assert response.headers['x-content-type-options'] == 'nosniff'
        health = client.get('/health').json()
        assert health['translator'] is False and health['models'] == {} and health['capture_fps_target'] == 25
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f'/ws/pose?session_id={SID}', headers=ORIGIN): pass
        assert client.get('/api/translate').status_code == 404
        assert client.get('/static/../app.py').status_code == 404


@pytest.mark.parametrize('values', [
    dict(production=True), dict(production=True, public_origin='https://example.test', mode='sample'),
    dict(mode='model', production=True, public_origin='https://example.test'),
    dict(mode='model', production=True, public_origin='https://example.test', detector_path='d', pose_path='p'),
    dict(threads=0), dict(production=True, public_origin='https://u:p@example.test'),
    dict(coverage=0.4), dict(models=('bad',)), dict(models=()), dict(models=('single', 'single')),
    dict(device='mps'), dict(rgb_device='tpu')])
def test_invalid_configuration_fails_closed(values):
    with pytest.raises(ValueError): create_app(Settings(**values))


@pytest.mark.parametrize('origin', ['', 'https://wrong.example'])
def test_origin_required(origin):
    with TestClient(create_app(Settings(), FakeService())) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f'/ws/pose?session_id={SID}', headers={'origin': origin}): pass


def test_sessions_are_isolated_and_buffers_released():
    service = FakeService(); app = create_app(Settings(), service)
    with TestClient(app) as client:
        with ExitStack() as stack:
            a = connect(stack, client, 'translate'); ap = connect(stack, client, 'pose')
            b = connect(stack, client, 'translate', SID+'2'); bp = connect(stack, client, 'pose', SID+'2')
            start(a); start(b)
            for at in [0, 40]:
                assert frame(ap, 40, at)['buffer_len'] == at//40+1
                assert frame(bp, 200, at)['buffer_len'] == at//40+1
            for ws in [a, b]:
                assert finish(ws)['accepted'] is True
            assert [round(c.points.mean()) for c in service.calls] == [40, 200]
            assert all(not s.keypoints for s in app.state.runtime.sessions.values())
        assert not app.state.runtime.sessions


@pytest.mark.parametrize('model', ['single', 'ensemble'])
def test_model_routing_timing_and_release(model):
    service = FakeService(); app = create_app(Settings(), service)
    with TestClient(app) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack, model, times=(0, 40, 100))
        s = app.state.runtime.sessions[SID]
        assert s.times == [0, 40, 100] and s.track == {'mid': 'seen'}
        assert len(s.frames) == (3 if model == 'ensemble' else 0)
        result = finish(tr)
        call, = service.calls
        assert call.model == model and call.times == [0, 40, 100]
        assert len(call.frames) == (3 if model == 'ensemble' else 0)
        assert all(f.startswith('data:image/jpeg') for f in call.frames)
        means = call.points[:, 0, 0]  # FakeService.extract writes each frame's mean into its keypoints
        assert means[0] < means[1] < means[2]  # distinct frames, kept in capture order
        if model == 'ensemble':  # each retained JPEG stays paired with its own keypoints (RGB crops use both)
            assert np.array_equal(np.float32([decode_frame(f).mean() for f in call.frames]), means)
        assert (result['model'], result['model_label'], result['capture_fps']) == (model, service.models[model]['label'], 20.0)
        assert (result['frames'], result['frames_used'], result['threshold']) == (3, 3, service.models[model]['threshold'])
        # No keypoints, times, frames or tracker state survive the utterance.
        assert not any(call.lists) and not (s.keypoints or s.times or s.frames or s.track)
        assert client.get('/health').json()['answered'] == [1, 1]


def test_stop_releases_buffers_and_tracker():
    app = create_app(Settings(), FakeService())
    with TestClient(app) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack, 'ensemble')
        s = app.state.runtime.sessions[SID]
        assert s.keypoints and s.times and s.frames and s.track
        tr.send_json({'type': 'stop_recording', 'request_id': RID})
        assert tr.receive_json()['type'] == 'recording_stopped'
        assert not (s.keypoints or s.times or s.frames or s.track or s.recording)


def test_unknown_model_is_refused():
    with TestClient(create_app(Settings(), FakeService())) as client, ExitStack() as stack:
        tr = connect(stack, client, 'translate')
        tr.send_json({'type': 'start_recording', 'request_id': RID, 'model': 'vlp80'})
        assert 'not available' in tr.receive_json()['error']


@pytest.mark.parametrize('score', [9.0, float('nan'), float('inf')])
def test_uncertain_outputs_never_leave_server(score):
    service = FakeService(); service.confidence = score
    with TestClient(create_app(Settings(), service)) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack)
        result = finish(tr)
        assert result['accepted'] is False and result['text'] == ''
        assert result['reason'] == 'declined: low confidence'
        assert result['confidence'] == (score if score == 9.0 else None)
        assert client.get('/health').json()['answered'] == [0, 1]  # completed, not answered


@pytest.mark.parametrize('model, accepted', [('single', False), ('ensemble', True)])
def test_each_model_uses_its_own_threshold(model, accepted):
    service = FakeService(); service.confidence = 0.8  # between single 0.75 and ensemble 0.90
    with TestClient(create_app(Settings(), service)) as client, ExitStack() as stack:
        result = finish(utterance(client, stack, model)[0])
        assert (result['accepted'], result['threshold']) == (accepted, service.models[model]['threshold'])
        assert result['text'] == ('Test output.' if accepted else '')


def test_capture_rate_uses_the_first_timestamp():
    with TestClient(create_app(Settings(), FakeService())) as client, ExitStack() as stack:
        assert finish(utterance(client, stack, times=(40, 80, 180))[0])['capture_fps'] == 14.3  # 2 intervals in 140 ms


def test_single_frame_finish_is_refused():
    service = FakeService()
    with TestClient(create_app(Settings(), service)) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack, times=(0,))
        assert 'Too few frames' in finish(tr)['error'] and not service.calls


def test_empty_output_is_declined():
    service = FakeService(); service.text = '  '
    with TestClient(create_app(Settings(), service)) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack)
        result = finish(tr)
        assert (result['accepted'], result['text'], result['reason']) == (False, '', 'declined: empty')
        assert result['model'] == 'single'  # start_recording without a model gets the first published one


def test_translate_error_is_reported_and_buffers_released():
    class Failing(FakeService):
        def translate(self, *args, **kwargs):
            super().translate(*args, **kwargs)
            raise ValueError('Keep one signer in view for the whole utterance, then try again.')
    service = Failing()
    with TestClient(create_app(Settings(), service)) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack, 'ensemble')
        assert finish(tr) == {'error': 'Keep one signer in view for the whole utterance, then try again.', 'request_id': RID}
        assert not any(service.calls[0].lists)
        assert client.get('/health').json()['answered'] == [0, 0]


def test_model_mode_health_publishes_both_operating_points(monkeypatch):
    # The real ModelService.__init__ behind /health; weights, pose estimator and DINOv2 are stubbed out.
    monkeypatch.setattr('demo.service.Translator', lambda spec, device: spec)
    rgb_built = []
    for name, attrs in {'torch': dict(set_num_threads=lambda n: None), 'cv2': dict(setNumThreads=lambda n: None),
                        'rtmlib': dict(Wholebody=lambda **options: options),
                        'demo.rgb_features': dict(RGBFeatures=lambda device: rgb_built.append(device) or device)}.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    with TestClient(create_app(Settings(mode='model', coverage=0.75, rgb_device='mps'))) as client:
        health = client.get('/health').json()
    assert rgb_built == ['mps']  # the ensemble's DINOv2 features are built, on the requested device
    assert health['translator'] is True and health['coverage_target'] == 0.75
    assert list(health['models']) == ['single', 'ensemble']
    keys = ('threshold', 'needs_frames', 'length_penalty', 'validation_bleu4', 'test_bleu4')
    assert [health['models']['single'][k] for k in keys] == [0.859783, False, 0.7, 12.643, 11.652]
    assert [health['models']['ensemble'][k] for k in keys] == [1.088921, True, 0.3, 12.47, 11.802]
    assert all((m['beam'], m['coverage'], m['label']) == (10, 0.75, MODELS[k]['label']) for k, m in health['models'].items())
    assert MODELS['single']['ckpts'] == [('results/ckpts_live/w4A_s202_best.pt',
                                          '0c245b3701d0c80ce0c8ad79a1c81c964afced5f0ac8c3eabe98f87bd35d65b6')]
    assert MODELS['single']['tau'] == {0.5: 0.706661, 0.75: 0.859783, 0.9: 0.974299}
    with TestClient(create_app(Settings(mode='model', models=('single',)))) as client:
        assert list(client.get('/health').json()['models']) == ['single']
    assert rgb_built == ['mps']  # a pose-only deployment never loads DINOv2


def test_published_evidence_matches_the_loaded_models():
    """service.MODELS is the source of truth; the public evidence snapshot must not drift from it."""
    points = json.loads((STATIC / 'evidence.json').read_text())['live_system']['operating_points']
    for name, spec in MODELS.items():
        point = points[name]
        assert {float(q): t for q, t in point['confidence_threshold_nats'].items()} == spec['tau'], name
        assert (point['validation_bleu4'], point['official_test_bleu4']) == (
            spec['evidence']['validation_bleu4'], spec['evidence']['test_bleu4']), name
    (path, sha), = MODELS['single']['ckpts']
    assert (points['single']['checkpoint'], points['single']['checkpoint_sha256']) == (path, sha)
    assert [(m['checkpoint'], m['sha256']) for m in points['ensemble']['members']] == MODELS['ensemble']['ckpts']


@pytest.mark.parametrize('action', ['stop', 'finish'])
def test_late_pose_cannot_seed_the_next_utterance(action):
    # A frame is almost always in pose estimation when the user presses Cancel or Finish. Its tracker write
    # must land in the old utterance's dict, never in the next utterance's (which must seed at its frame 0).
    entered, release = threading.Event(), threading.Event()
    class Slow(FakeService):
        def extract(self, frame, track=None):
            if frame.mean() > 150:  # only the last frame blocks
                entered.set(); assert release.wait(5)
            track['mid'] = 'from-utterance-A'
            return super().extract(frame)
    app = create_app(Settings(), Slow())
    with TestClient(app) as client, ExitStack() as stack:
        tr, po = utterance(client, stack, times=(0, 40))
        s = app.state.runtime.sessions[SID]
        po.send_json({'type': 'frame', 'request_id': RID, 'captured_at': 80, 'data': jpeg(200)})
        assert entered.wait(3)
        if action == 'stop':
            tr.send_json({'type': 'stop_recording', 'request_id': RID})
            assert tr.receive_json()['type'] == 'recording_stopped'
            start(tr, RID2)
        else:
            tr.send_json({'type': 'translate', 'request_id': RID}); tr.send_json({'type': 'ping'})
            assert tr.receive_json()['type'] == 'pong'  # Finish has been handled before the frame completes
        release.set()
        assert po.receive_json()['type'] == 'frame_ignored'
        if action == 'finish': assert tr.receive_json()['accepted'] is True
        assert s.track == {}


def test_cancel_invalidates_late_translation():
    entered = threading.Event(); release = threading.Event()
    class Slow(FakeService):
        def translate(self, *args, **kwargs):
            entered.set(); assert release.wait(5)
            return super().translate(*args, **kwargs)
    app = create_app(Settings(), Slow())
    with TestClient(app) as client, ExitStack() as stack:
        tr, _ = utterance(client, stack)
        tr.send_json({'type': 'translate', 'request_id': RID})
        assert entered.wait(3)
        tr.send_json({'type': 'stop_recording', 'request_id': RID})
        assert tr.receive_json()['type'] == 'recording_stopped'
        release.set()
        tr.send_json({'type': 'ping'})
        assert tr.receive_json()['type'] == 'pong'
        assert app.state.runtime.sessions[SID].request_id is None


def test_recording_bounds_and_bad_messages_recover():
    with TestClient(create_app(Settings(max_frames=2), FakeService())) as client, ExitStack() as stack:
        tr = connect(stack, client, 'translate'); po = connect(stack, client, 'pose')
        for malformed in ['not json', '[]', '{"type":"wrong"}']:
            tr.send_text(malformed); assert 'error' in tr.receive_json()
        start(tr)
        assert frame(po)['buffer_len'] == 1
        for bad in [0, 19, 10_151, True, None]:  # duplicate, faster than 50 fps, past 10 s, not a number
            assert 'timing' in frame(po, at=bad)['error']
        assert frame(po, at=40)['buffer_len'] == 2
        assert 'error' in frame(po, at=80)
        tr.send_json({'type': 'stop_recording', 'request_id': RID})
        assert tr.receive_json()['type'] == 'recording_stopped'
        assert frame(po, at=120)['type'] == 'frame_ignored'


def test_quota_cannot_be_reset_by_reconnecting():
    app = create_app(Settings(requests_per_minute=1), FakeService())
    with TestClient(app) as client:
        with ExitStack() as stack:
            start(connect(stack, client, 'translate'))
        with ExitStack() as stack:
            tr = connect(stack, client, 'translate', SID+'new')
            tr.send_json({'type': 'start_recording', 'request_id': RID})
            assert 'wait a minute' in tr.receive_json()['error']


def test_sample_mode_speaks_the_same_protocol():
    with TestClient(create_app(Settings(mode='sample'))) as client, ExitStack() as stack:
        health = client.get('/health').json()
        assert health['translator'] is True and list(health['models']) == ['single']
        assert health['models']['single']['threshold'] == 3.0
        result = finish(utterance(client, stack, times=(0, 100))[0])
        assert result['accepted'] is True and result['mode'] == 'sample' and result['frames_used'] == 3


def test_capacity_and_cross_client_session_claims():
    rt = Runtime(Settings(max_sessions=1))
    rt.attach(SID, 'visitor-a', 'pose')
    for sid, ip, channel in [(SID, 'visitor-b', 'translate'), (SID, 'visitor-a', 'pose'), (SID+'2', 'visitor-a', 'pose')]:
        with pytest.raises(CapacityError): rt.attach(sid, ip, channel)


def test_image_validation_and_checksum(tmp_path):
    assert decode_frame(jpeg()).shape == (20, 20, 3)
    # rtmlib and the RGB crops (cv2 BGR->RGB) both expect BGR; a grey frame cannot tell the orders apart.
    blue, green, red = decode_frame(jpeg(color=(255, 0, 0))).reshape(-1, 3).mean(0)
    assert red > 240 and blue < 15 and green < 15
    for invalid in [None, 'data:image/png;base64,AAA', 'data:image/jpeg;base64,%%%', jpeg(size=(1300, 1000))]:
        with pytest.raises(ValueError): decode_frame(invalid)
    asset = tmp_path / 'model'; asset.write_bytes(b'abc')
    with pytest.raises(ValueError): check_asset(str(asset), '0'*64)
    assert check_asset(str(asset)).endswith('model')


def test_grid_indices_resample_to_25fps():
    assert grid_indices([0, 100, 200]) == [0, 0, 1, 1, 2, 2]
    assert grid_indices([0, 40, 80]) == [0, 1, 2]


def test_thresholds_cover_every_validated_rate_in_order():
    for spec in MODELS.values():
        assert set(spec['tau']) == {0.5, 0.75, 0.9}
        assert spec['tau'][0.5] < spec['tau'][0.75] < spec['tau'][0.9]


def fake_pose(people):
    """rtmlib Wholebody stand-in: one detection per (left shoulder, right shoulder) pair, confident shoulders only."""
    kp = np.zeros((len(people), 133, 2), np.float32); sc = np.zeros((len(people), 133), np.float32)
    for i, (left, right) in enumerate(people):
        kp[i, 5], kp[i, 6] = left, right
        sc[i, [5, 6]] = 1.0
    return SimpleNamespace(det_model=lambda img: np.zeros((len(people), 4)), pose_model=lambda img, bboxes: (kp, sc))


def test_no_person_does_not_trigger_whole_image_pose_fallback():
    service = ModelService.__new__(ModelService)
    def forbidden(*args, **kwargs): raise AssertionError('Pose must not run without a detection')
    for boxes in [None, np.zeros((0, 4))]:
        service.pose = SimpleNamespace(det_model=lambda img: boxes, pose_model=forbidden)
        points, count = service.extract(np.zeros((20, 20, 3), np.uint8), {})
        assert points.shape == (133, 3) and not points.any() and count == 0


def test_tracker_follows_one_signer_across_frames():
    service, img, track = ModelService.__new__(ModelService), np.zeros((20, 20, 3), np.uint8), {}
    edge, central, big_edge = ((2, 5), (4, 5)), ((8, 10), (12, 10)), ((0, 5), (8, 5))
    shoulders = lambda points: points[5:7, :2].tolist()
    service.pose = fake_pose([edge, central])
    points, count = service.extract(img, track)
    assert count == 2 and points.shape == (133, 3) and shoulders(points) == [[8, 10], [12, 10]]
    assert np.allclose(track['mid'], (10, 10)) and track['wid'] == 4
    # Next frame: a bigger person at index 0, so both kp[0] and a fresh seed would switch; tracking must not.
    service.pose = fake_pose([big_edge, central])
    assert shoulders(service.extract(img)[0]) == [[0, 5], [8, 5]]
    assert shoulders(service.extract(img, track)[0]) == [[8, 10], [12, 10]]


def test_tracker_reseeds_when_the_signer_is_lost_and_centrality_breaks_ties():
    service, img = ModelService.__new__(ModelService), np.zeros((100, 100, 3), np.uint8)
    shoulders = lambda points: points[5:7, :2].tolist()
    # Frame 0, equal widths: the more central person wins even though the other is detection 0.
    service.pose, track = fake_pose([((0, 10), (4, 10)), ((48, 10), (52, 10))]), {}
    assert shoulders(service.extract(img, track)[0]) == [[48, 10], [52, 10]]
    # The nearest candidate is 2.25 shoulder widths away (> GATE 1.6): re-seed to the largest, not the nearest.
    service.pose = fake_pose([((40, 10), (42, 10)), ((70, 10), (78, 10))])
    assert shoulders(service.extract(img, track)[0]) == [[70, 10], [78, 10]]
    assert np.allclose(track['mid'], (74, 10)) and track['wid'] == 8


class Recorder:
    def __call__(self, kps, rgb_feats=None):
        self.kps, self.rgb = kps, rgb_feats
        return SimpleNamespace(text='x', confidence=0.1)


class FakeRGB:
    def sample_indices(self, n): return np.array([0, n - 1])
    def __call__(self, frames, kps):
        self.frames, self.kps = frames, kps
        return np.zeros((len(frames), 1152), np.float32)


def test_translate_resamples_and_computes_rgb_only_for_the_ensemble():
    service = ModelService.__new__(ModelService)
    service.translators, service.rgb = {name: Recorder() for name in MODELS}, FakeRGB()
    kps = [np.full((133, 3), i + 1, np.float32) for i in range(3)]
    times, frames = [0, 100, 200], [jpeg(v) for v in (10, 100, 200)]
    out = service.translate('single', kps, times, frames)
    single = service.translators['single']
    assert single.kps[:, 0, 0].tolist() == [1, 1, 2, 2, 3, 3] and single.rgb is None and out.frames_used == 6
    assert not hasattr(service.rgb, 'frames')
    service.translate('ensemble', kps, times, frames)
    assert [round(f.mean()) for f in service.rgb.frames] == [10, 200] and service.rgb.kps[:, 0, 0].tolist() == [1, 3]
    assert service.translators['ensemble'].rgb.shape == (2, 1152)
    with pytest.raises(ValueError, match='one signer'):
        service.translate('single', [kps[0], np.zeros((133, 3)), np.zeros((133, 3))], times)
    # At least 80% of the 25 fps grid must show the signer: 80 of 100 passes, 79 of 100 does not.
    for shown, ok in [(80, True), (79, False)]:
        pts = [np.ones((133, 3), np.float32)] * shown + [np.zeros((133, 3), np.float32)] * (100 - shown)
        if ok: assert service.translate('single', pts, np.arange(100) * 40).frames_used == 100
        else:
            with pytest.raises(ValueError, match='one signer'): service.translate('single', pts, np.arange(100) * 40)
