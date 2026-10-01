/* One utterance at a time. A request ID prevents a cleared or superseded result returning. */
'use strict';
const $ = (id) => document.getElementById(id);
const video = $('videoElement');
let localVoice = null;
let speechGeneration = 0;
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const MAX_SECONDS = 10;
const TARGET_FPS = 25;
// Capture encoding, tunable in one place (results/live_demo_2026-09-24/capture_fidelity_output.txt).
const CAPTURE_MAX_WIDTH = 960;
const CAPTURE_MAX_HEIGHT = 720;
const JPEG_QUALITY = 0.75;
const MIN_FRAME_GAP_MS = 30; // a smaller camera media-time advance means no new camera frame
const FALLBACK_POLL_MS = 10; // without requestVideoFrameCallback, poll faster than any camera period; the gap check drops repeats
const MODEL_NAMES = { single: 'Single best model', ensemble: 'Six-model ensemble' };
// The basis for "best" (job 21736, results/seedens_21736/analysis.txt): highest validation of 21 same-recipe singles.
const MODEL_NOTES = { single: 'Validation-best of 21 adopted-recipe runs (chosen on validation; its test score is not the highest of the 21)' };
const state = {
  phase: 'off', stream: null, sockets: [], connected: false, sample: false, model: 'single', models: {},
  id: null, pendingFrame: false, finishPending: false, started: 0, finishTime: 0, t0: null, lastCaptureAt: -100,
  frames: 0, skipped: 0, captured: 0, lastPresented: null, progressAt: 0, queue: [], queuedBytes: 0, result: null, history: [], cameraEpoch: 0,
  poseMs: [], queueMs: [], finishMs: null, speechMs: null,
  frameTimer: null, frameCallback: null, clockTimer: null, responseTimer: null, reconnectTimer: null,
  toastTimer: null, disposed: false, maxFrames: 300, heartbeat: null, reconnectAttempts: 0, translatorReady: false, speaking: false,
  // engine 'server': WebSockets to demo/app.py. 'browser': the Web Worker demo/static/engine/engine.js, same messages.
  // Decided once (updateHealth) and never changed afterwards.
  engine: null, worker: null, capabilities: null, consent: false, loaded: {}, download: null, unavailable: '', loadError: {},
  providers: {}, poseProbeMs: null, forgetting: false, inflight: null, retranslating: false, peopleCleared: false,
};
const isBusy = () => ['starting', 'recording', 'processing'].includes(state.phase);
// Stored models can be removed when nothing uses them: no utterance, no download, no init in flight.
// With a worker: any time except while init runs (files from an earlier visit may exist even before consent is asked
// again). Without one (the unavailable state after consent), a bare worker is started just to delete the stored files.
const canForget = () => !state.forgetting && !isBusy() && !state.download
  && (state.worker ? !(state.consent && !state.translatorReady) : !!state.unavailable && state.consent);
const mb = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`;
// Consent to the one-time model download. Storage may be blocked or throw; consent then lasts for this page only.
const CONSENT_KEY = 'auslan-live-consent-v1';
function consentStored() { try { return localStorage.getItem(CONSENT_KEY) === '1'; } catch { return false; } }
function storeConsent(on) { try { if (on) localStorage.setItem(CONSENT_KEY, '1'); else localStorage.removeItem(CONSENT_KEY); } catch { /* this page only */ } }
// Bytes the selected model still needs; the pose models and runtimes are shared and fetched with the first model.
const downloadBytes = (key) => (state.models[key]?.download_bytes || 0) + (Object.values(state.loaded).some(Boolean) ? 0 : state.models[key]?.shared_download_bytes || 0);

const modelEl = (key, suffix = '') => $(`model${key[0].toUpperCase()}${key.slice(1)}${suffix}`);
const modelName = (key) => (!state.sample && MODEL_NAMES[key]) || state.models[key]?.label || key || '—';
// Linear-interpolated percentile, as np.percentile in pose_throughput_bench.summarise.
function pct(values, q) {
  const s = [...values].sort((a, b) => a - b), i = (s.length - 1) * q, lo = Math.floor(i);
  return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo);
}
const ms = (v) => Number.isFinite(v) ? `${Math.round(v)} ms` : '—';
const spread = (v) => v.length ? `${Math.round(pct(v, .5))} / ${Math.round(pct(v, .95))} ms` : '—';
const captureRate = () => state.captured > 1 && state.lastCaptureAt > 0 ? (state.captured - 1) * 1000 / state.lastCaptureAt : NaN;
// Move focus only when the focused control was just disabled or hidden, so keyboard focus never falls to <body>.
function refocus(el) {
  const a = document.activeElement;
  if (!a || a === document.body || a.disabled || a.closest?.('[hidden]')) el.focus?.();
}

function render() {
  const cameraOn = !!state.stream;
  const busy = ['starting', 'recording', 'processing'].includes(state.phase);
  const browser = state.engine === 'browser';
  if (browser) {
    state.connected = !!state.loaded[state.model] && !state.unavailable;
    renderEngine(busy);
  }
  const keys = Object.keys(state.models);
  $('modelChoice').hidden = !keys.length;
  $('modelFixed').hidden = keys.length !== 1;
  $('modelFixed').textContent = keys.length === 1 ? modelName(keys[0]) : '';
  for (const key of Object.keys(MODEL_NAMES)) {
    modelEl(key, 'Option').hidden = keys.length < 2 || !keys.includes(key);
    Object.assign(modelEl(key), { checked: key === state.model, disabled: busy });
  }
  const info = state.models[state.model];
  $('modelDetail').textContent = info ? [info.label, !state.sample && MODEL_NOTES[state.model],
    Number.isFinite(info.validation_bleu4) && `Stored benchmark clips: validation ${info.validation_bleu4.toFixed(3)} · test ${info.test_bleu4.toFixed(3)} BLEU-4 (not measured on live camera input)`,
    info.needs_frames && (browser ? 'Adds image features after Finish; the time depends on this device' : 'Adds about a second after Finish for image features')].filter(Boolean).join('. ') + '.' : '';
  const recording = state.phase === 'recording';
  const accepted = state.result && state.result.accepted !== false && !state.result.reason?.startsWith('declined');
  $('cameraStage').classList.toggle('has-stream', cameraOn);
  $('cameraStage').classList.toggle('is-recording', recording);
  $('cameraPlaceholder').hidden = cameraOn;
  $('framingGuide').hidden = !cameraOn || !$('guideToggle').checked;
  $('framingHint').hidden = !cameraOn;
  $('cameraOffButton').disabled = !cameraOn && state.phase !== 'requesting';
  $('enableCameraButton').disabled = state.phase === 'requesting' || !state.translatorReady;
  $('cameraBadgeText').textContent = recording ? 'Recording' : cameraOn ? 'Preview only' : 'Camera off';
  // aria-disabled, not disabled, while starting or processing: the focused button keeps keyboard focus (the handlers check the phase).
  $('recordButton').disabled = !cameraOn || !state.connected;
  $('recordButton').setAttribute('aria-disabled', String(['starting', 'processing'].includes(state.phase)));
  $('recordLabel').textContent = recording ? 'Finish & translate' : state.phase === 'starting' ? 'Getting ready…' : state.phase === 'processing' ? 'Translating…' : state.result ? 'Sign again' : 'Start signing';
  $('recordIcon').classList.toggle('finish', recording);
  $('clearButton').disabled = !busy && !state.result && !state.frames;
  $('clearLabel').textContent = busy ? 'Cancel' : 'Clear';
  $('copyButton').disabled = !accepted;
  $('speakButton').disabled = !accepted || !localVoice;
  $('captureIndicator').className = 'indicator ' + (recording ? 'recording' : busy ? 'processing' : '');
  const r = state.result || {};
  const details = {
    modelUsedDetail: r.model_label || '—',
    thresholdDetail: Number.isFinite(r.threshold) ? `${r.threshold.toFixed(3)} nats${Number.isFinite(r.coverage_target) ? ` · ${Math.round(r.coverage_target * 100)}%` : ''}` : '—',
    capturedDetail: String(state.captured), framesDetail: String(state.frames), backlogDetail: String(state.queue.length),
    skippedDetail: String(state.skipped),
    captureRateDetail: Number.isFinite(r.capture_fps) ? `${r.capture_fps.toFixed(1)} fps · ${r.frames_used} frames on the 25 fps grid`
      : Number.isFinite(captureRate()) ? `${captureRate().toFixed(1)} fps so far` : '—',
    poseDetail: spread(state.poseMs), queueDetail: spread(state.queueMs),
    latencyDetail: Number.isFinite(state.finishMs) ? `${(state.finishMs / 1000).toFixed(1)} s` : '—',
    speechDetail: ms(state.speechMs), translateDetail: ms(r.translate_ms),
    rgbDetail: r.rgb_ms ? ms(r.rgb_ms) : state.result ? 'Not used' : '—',
    scoreDetail: Number.isFinite(r.confidence) ? `${r.confidence.toFixed(3)} nats` : state.result ? 'Unavailable' : '—',
    personsDetail: r.persons ? peopleSummary(r.persons) : state.result ? 'Not reported' : '—',
  };
  for (const [id, text] of Object.entries(details)) $(id).textContent = text;
  ['stepCamera', 'stepSign', 'stepReview'].forEach((id) => $(id).removeAttribute('aria-current'));
  $(state.result ? 'stepReview' : cameraOn ? 'stepSign' : 'stepCamera').setAttribute('aria-current', 'step');
}
// The in-browser engine's consent, download, capability and provider state.
function renderEngine(busy) {
  const info = state.models[state.model], dl = state.download, off = state.unavailable;
  const needs = !!info && !state.loaded[state.model] && !off;
  const name = modelName(state.model), bytes = downloadBytes(state.model), error = state.loadError[state.model];
  const asking = !!state.capabilities && !state.consent && !off; // the consent card, before any download
  const local = !!dl && !dl.total;                                 // every file is already in this browser's storage
  const settingUp = !!dl && !local && dl.loaded >= dl.total;       // downloaded; sessions and the pose probe still to run
  $('consentCard').hidden = !asking;
  $('consentWarning').hidden = state.capabilities?.webgpu !== false;
  $('engineStatus').hidden = !needs && !dl;
  $('engineProgress').hidden = !dl || local;
  if (dl) Object.assign($('engineProgress'), { max: dl.total || 1, value: dl.loaded });
  $('downloadButton').hidden = !needs || !!dl || !state.worker;
  $('downloadButton').disabled = busy || state.forgetting;
  $('downloadLabel').textContent = bytes ? `Download ${name} · ${mb(bytes)}` : `Load ${name} from this browser`;
  $('engineStatusText').textContent = local ? `Loading ${modelName(dl.model)} from this browser. Files are checked before use.`
    : settingUp ? `Setting up ${modelName(dl.model)} in this browser and checking this device's speed.`
    : dl ? `Downloading ${modelName(dl.model)} to this browser · ${mb(dl.loaded)} of ${mb(dl.total)}. Files are checked and kept for your next visit.`
    : error ? `${error} You can try the download again.`
    : !needs ? '' : bytes ? `${name} runs in this browser after a one-time download of ${mb(bytes)}, kept for your next visit. Nothing is uploaded.`
    : `${name} is already stored in this browser.`;
  $('retryButton').hidden = !off;
  $('forgetButton').hidden = off ? !state.consent : !state.worker;
  // aria-disabled while removing, so the focused button keeps focus (the handler checks canForget).
  $('forgetButton').disabled = !canForget() && !state.forgetting;
  $('forgetButton').setAttribute('aria-disabled', String(state.forgetting));
  $('modeNotice').hidden = !off;
  if (off) {
    $('modeNotice').querySelector('strong').textContent = 'Live translation is unavailable in this browser';
    $('modeNotice').querySelector('p span').textContent = `${off} The research overview is still available.`;
  }
  const p = state.providers;
  $('runtimeDetail').textContent = off ? 'Unavailable in this browser'
    : p.pose ? `In this browser · pose ${p.pose}${Number.isFinite(state.poseProbeMs) ? ` (pose probe ~${Math.round(state.poseProbeMs)} ms/frame)` : ''} · translator ${p[`translator_${state.model}`] || p.translator_single || '—'}${p.threads ? ` · ${p.threads} WASM thread${p.threads > 1 ? 's' : ''}` : ''}`
    : asking ? 'In this browser · not started' : 'In this browser · loading';
  $('engineDetail').textContent = Object.keys(state.loaded).filter((k) => state.loaded[k]).map(modelName).join(' · ') || (off || asking ? 'None' : 'Loading');
  $('connectionBadge').className = 'connection ' + (off ? 'disconnected' : state.connected ? 'connected' : '');
  $('connectionText').textContent = off ? 'Unavailable' : state.connected ? 'Runs in this browser'
    : dl ? (local || settingUp ? 'Loading' : 'Downloading') : asking ? 'Not started' : 'Preparing';
}

function message(text) { $('captureMessage').textContent = text; }
function toast(text) {
  clearTimeout(state.toastTimer);
  $('toast').textContent = text;
  $('toast').classList.add('show');
  state.toastTimer = setTimeout(() => $('toast').classList.remove('show'), 3500);
}
function warning(text = '') {
  const el = $('captureWarning'); // a live region: rewrite only on change, or screen readers repeat it per frame
  if (el.textContent === text && el.hidden === !text) return;
  el.textContent = text; el.hidden = !text;
}
function idleResult(heading = 'A place for your words.', hint = 'Your translation will appear here after you finish signing.', badge = 'Waiting') {
  $('resultEmpty').hidden = false;
  $('resultContent').hidden = true;
  $('resultHeading').textContent = heading;
  $('resultHint').textContent = hint;
  $('resultBadge').textContent = badge;
  $('resultBadge').className = 'result-badge';
  $('resultArea').classList.remove('processing');
  $('personBox').hidden = true;
}
function stopAudio() {
  speechGeneration++;
  window.speechSynthesis?.cancel();
  state.speaking = false;
  $('speakLabel').textContent = 'Read aloud';
  $('audioMessage').textContent = '';
}
function updateVoices() {
  const voices = window.speechSynthesis?.getVoices().filter(v => v.localService && /^en([-_]|$)/i.test(v.lang)) || [];
  // The research measured macOS "Karen" (en-AU); prefer it, then any local en-AU voice, then any local English voice.
  localVoice = voices.find(v => /^Karen\b/.test(v.name) && /^en[-_]AU$/i.test(v.lang))
    || voices.find(v => /^en[-_]AU$/i.test(v.lang)) || voices[0] || null;
  $('autoSpeak').disabled = !localVoice;
  if (state.result?.accepted && !localVoice) $('audioMessage').textContent = 'No local English voice is available in this browser. You can still copy the text.';
  render();
}
function stopTimers() {
  clearInterval(state.frameTimer); clearInterval(state.clockTimer); clearTimeout(state.responseTimer);
  if (state.frameCallback !== null) video.cancelVideoFrameCallback?.(state.frameCallback);
  state.frameTimer = state.frameCallback = state.clockTimer = state.responseTimer = null;
}
function send(channel, value) {
  if (state.engine === 'browser') {
    if (!state.worker) return false;
    state.worker.postMessage(value); // one worker serves both channels; replies say which
    return true;
  }
  const ws = state.sockets[channel];
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(value));
  return true;
}
function resetCurrent({ notify = true } = {}) {
  const id = state.id;
  state.id = null; // Invalidate before any queued response is handled.
  stopTimers(); stopAudio();
  if (notify && id) send(1, { type: 'stop_recording', request_id: id });
  state.phase = state.stream ? 'ready' : 'off';
  state.pendingFrame = state.finishPending = false;
  state.frames = state.skipped = state.captured = 0; state.lastPresented = null;
  state.queue = []; state.queuedBytes = 0; state.t0 = null; state.lastCaptureAt = -100;
  state.poseMs = []; state.queueMs = []; state.finishMs = state.speechMs = null;
  state.result = null; state.retranslating = false;
  $('captureProgress').style.width = '0%';
  $('timer').firstChild.textContent = '00:00 ';
  warning(); idleResult();
  message(state.stream ? 'Ready. Sign one short utterance, then select Finish.' : 'Enable your camera to get started.');
  render();
}

async function enableCamera() {
  const epoch = ++state.cameraEpoch;
  state.phase = 'requesting';
  message('Allow camera access in your browser to continue.');
  render();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw { name: 'UnsupportedError' };
    const stream = await navigator.mediaDevices.getUserMedia({ video: {
      facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 }, frameRate: { ideal: TARGET_FPS },
    }, audio: false });
    if (epoch !== state.cameraEpoch || state.disposed) { stream.getTracks().forEach(t => t.stop()); return; }
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    if (epoch !== state.cameraEpoch) return;
    const scale = Math.min(1, CAPTURE_MAX_WIDTH / video.videoWidth, CAPTURE_MAX_HEIGHT / video.videoHeight);
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    $('cameraDetail').textContent = `${settings.width || video.videoWidth} × ${settings.height || video.videoHeight} · ${settings.frameRate ? settings.frameRate.toFixed(1) + ' fps' : 'rate unavailable'}`;
    track.onended = () => { disableCamera(); message('Your camera was disconnected. Enable it to try again.'); };
    state.phase = 'ready';
    message(state.connected ? 'Ready. Sign one short utterance, then select Finish.'
      : state.engine === 'browser' ? 'Camera ready. Waiting for the translator to finish loading.' : 'Camera ready. Waiting for the translation service.');
    if (settings.frameRate && settings.frameRate < 24) warning('This camera is below the 25 fps capture target. Translation may be affected.');
    render();
    refocus(state.connected ? $('recordButton') : $('cameraOffButton')); // Enable camera is now hidden
  } catch (err) {
    if (epoch !== state.cameraEpoch) return;
    state.stream?.getTracks().forEach(t => t.stop()); state.stream = null;
    state.phase = 'off'; video.srcObject = null;
    const messages = {
      NotAllowedError: 'Camera access is blocked. Allow access in your browser settings, then try again.',
      NotFoundError: 'No camera was found. Connect a camera, then try again.',
      NotReadableError: 'Your camera is busy. Close other camera apps, then try again.',
      UnsupportedError: 'Camera access needs localhost or a secure HTTPS connection in a supported browser.',
    };
    message(messages[err.name] || 'The camera could not start. Check the connection and try again.');
    $('cameraPlaceholderTitle').textContent = 'Let’s get your camera ready.';
    render();
    refocus($('enableCameraButton'));
  }
}
function disableCamera() {
  ++state.cameraEpoch;
  state.stream?.getTracks().forEach(t => { t.onended = null; t.stop(); });
  state.stream = null; video.srcObject = null;
  $('cameraDetail').textContent = 'Off';
  resetCurrent();
  forgetPeople();
  refocus($('enableCameraButton'));
}

// The transport is decided once, by the first definite answer: HTTP 200 -> the Python service (it translates when it
// reports a translator, otherwise the showcase / offline notice); 404 -> a static host (Vercel, serve_static.py) -> this
// browser. A failed check (network, timeout, 5xx) decides nothing and never selects the browser engine, and a service
// that has answered stays the transport, so a restart or a slow check cannot switch a session to the browser.
async function updateHealth() {
  let status = 0, data = null;
  try {
    const res = await fetch('/health', { cache: 'no-store', signal: AbortSignal.timeout(8000) });
    status = res.status;
    if (res.ok) data = await res.json();
  } catch { data = null; }
  if (state.engine === null && status === 404 && typeof Worker === 'function') { state.engine = 'browser'; render(); return; }
  try {
    if (!data) throw Error('Health check failed');
    state.engine = 'server';
    state.sample = data.mode === 'sample';
    state.translatorReady = !!data.translator;
    state.maxFrames = data.max_frames || 300; // the server refuses frame max_frames + 1
    state.models = data.models || {};
    if (!(state.model in state.models)) state.model = Object.keys(state.models)[0] || 'single';
    $('modeNotice').hidden = !state.sample && state.translatorReady;
    $('modeNotice').querySelector('strong').textContent = state.sample ? 'Interface preview' : 'Translation service offline';
    $('modeNotice').querySelector('p span').textContent = state.sample ? 'Sample phrases will be shown. A trained translation model is not loaded.' : 'The research showcase is available. Live translation will be enabled when the model service is ready.';
    $('engineDetail').textContent = Object.keys(state.models).map(modelName).join(' · ') || 'Unavailable';
    $('runtimeDetail').textContent = 'Python service';
  } catch {
    state.translatorReady = false; state.models = {};
    $('engineDetail').textContent = 'Live service not connected';
    $('modeNotice').hidden = false;
    $('modeNotice').querySelector('strong').textContent = 'Research showcase';
    $('modeNotice').querySelector('p span').textContent = 'Live translation is not available on this site. Camera translation needs the separate Python service.';
  }
  render();
}
function connect() {
  if (state.disposed) return;
  clearTimeout(state.reconnectTimer);
  state.sockets.forEach(ws => { ws.onclose = ws.onmessage = ws.onopen = null; ws.close(); });
  const base = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`;
  const sessionId = crypto.randomUUID();
  state.sockets = [new WebSocket(`${base}/ws/pose?session_id=${sessionId}`), new WebSocket(`${base}/ws/translate?session_id=${sessionId}`)];
  for (const [index, ws] of state.sockets.entries()) {
    ws.onopen = () => {
      if (!state.sockets.every(socket => socket.readyState === WebSocket.OPEN)) return;
      state.connected = true; state.reconnectAttempts = 0;
      clearInterval(state.heartbeat);
      state.heartbeat = setInterval(() => { send(0, {type: 'ping'}); send(1, {type: 'ping'}); }, 25000);
      $('connectionBadge').className = 'connection connected';
      $('connectionText').textContent = 'Server connected';
      if (state.stream && state.phase === 'ready') message('Ready. Sign one short utterance, then select Finish.');
      updateHealth(); render();
    };
    ws.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      onServiceMessage(index, data);
    };
    ws.onerror = () => { /* Close handles recovery once for the connection pair. */ };
    ws.onclose = () => {
      if (state.disposed) return;
      state.connected = false; state.inflight = null; clearInterval(state.heartbeat);
      const interrupted = ['starting', 'recording', 'processing'].includes(state.phase);
      if (interrupted) resetCurrent({ notify: false });
      $('connectionBadge').className = 'connection disconnected';
      $('connectionText').textContent = 'Reconnecting';
      message(interrupted ? 'The connection was lost. This recording was discarded; please try again after reconnecting.' : 'The translation service is busy or unavailable. Reconnecting…');
      render(); refocus($('captureMessage'));
      state.sockets.forEach(other => { other.onclose = null; other.close(); });
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = setTimeout(boot, Math.min(30000, 2500 * 2 ** state.reconnectAttempts++));
    };
  }
}
// A reply from the service: index 0 is the pose channel, 1 the translate channel (a socket, or the worker's `channel`).
function onServiceMessage(index, data) {
  if (index === 1 && data.request_id && data.request_id === state.inflight && (data.type === 'result' || data.type === 'cancelled' || data.error)) state.inflight = null;
  if (data.request_id && data.request_id !== state.id) return;
  if (data.error) { fail(data.error); return; }
  if (index === 0 && (data.type === 'pose' || data.type === 'frame_ignored')) {
    state.pendingFrame = false;
    if (data.type === 'pose') {
      state.frames = data.buffer_len;
      if (Number.isFinite(data.pose_ms)) state.poseMs.push(data.pose_ms);
      render();
      if (data.person_count > 1) warning(state.engine === 'browser'
        ? 'More than one person is visible. The translator picks the person moving most like a signer and says who; you can switch afterwards.'
        : 'More than one person is visible. The translator follows one signer; keep only the signer in frame if you can.');
      else if (!data.has_person) warning('Move into the frame so your face, hands and upper body are visible.');
      else if (state.captured > TARGET_FPS && captureRate() < 24) warning('Capture is running below the 25 fps the model was trained on (rate in Session details). Translation may be affected.');
      else warning();
    }
    if (state.phase === 'processing') armFinishTimer();
    pumpFrames();
    if (state.finishPending && !state.pendingFrame && !state.queue.length) submitTranslation();
  } else if (data.type === 'recording_started' && state.phase === 'starting') beginCapture();
  else if (data.type === 'result' && state.phase === 'processing' && state.id) showResult(data);
}
// The in-browser engine: model files are fetched once, verified and cached; frames never leave the device.
// Page -> worker: check (no network) -> init (only after consent) -> load; forget removes every stored file.
function startEngine() {
  if (state.worker || state.unavailable) return;
  $('privacyNote').textContent = 'In this browser: camera frames are processed on this device and are not uploaded or saved. Model files are downloaded once, checked and kept in this browser’s storage (Session details can remove them). Capture buffers are released after completion, cancellation or when you leave the page, except that when two or more people were in view their movement coordinates (and, for the six-model ensemble, the frames) are kept until your next utterance, Clear or turning the camera off so you can switch person. People are named Person A, B… by where their shoulders are, not by face; the names and each person’s last position (kept even with one person in view) exist only in this tab and reset with Clear conversation or turning the camera off. Turning the camera off stops its live feed. Use Clear to discard the current utterance.';
  $('peopleNote').hidden = false;
  $('latencyHelp').textContent = 'Pose runs on every frame as you sign, so the wait after Finish depends on this device (Session details shows the time per frame). The six-model ensemble adds image features after Finish.';
  if (!spawnWorker(() => unavailable('The in-browser translator could not start in this browser.'))) {
    unavailable('This browser cannot start the in-browser translator.'); return;
  }
  state.worker.postMessage({ type: 'check' });
  render();
}
function spawnWorker(onerror) {
  try { state.worker = new Worker('/static/engine/engine.js', { type: 'module' }); } catch { return false; }
  state.worker.onmessage = (event) => onEngineMessage(event.data);
  state.worker.onerror = onerror;
  return true;
}
function stopWorker() {
  if (state.worker) { state.worker.onmessage = state.worker.onerror = null; state.worker.terminate(); }
  // models too: a page restored from the back-forward cache must not offer a download before init has run again
  Object.assign(state, { worker: null, capabilities: null, models: {}, loaded: {}, download: null, translatorReady: false,
    forgetting: false, inflight: null });
}
function loadModel(key) {
  if (!state.worker || !state.translatorReady || state.forgetting || state.download || state.loaded[key]) return;
  delete state.loadError[key];
  // A cached model downloads nothing: its total is 0 and it is shown as loading from this browser.
  state.download = { model: key, loaded: 0, total: state.models[key]?.cached ? 0 : downloadBytes(key) };
  state.worker.postMessage({ type: 'load', model: key });
  if (!isBusy()) message(`Preparing ${modelName(key)} in this browser.`);
  render();
}
// A final state until Retry: nothing is offered here, the research pages stay reachable. Never "Preparing" for ever.
function unavailable(reason) {
  if (isBusy()) resetCurrent({ notify: false });
  stopWorker();
  state.unavailable = reason; state.models = {};
  message(reason);
  render(); refocus($('captureMessage'));
}
function onEngineMessage(data) {
  if ('channel' in data) { const { channel, ...reply } = data; onServiceMessage(channel, reply); return; }
  if (data.type === 'capabilities') {
    state.capabilities = data;
    if (state.consent || consentStored()) { state.consent = true; state.worker.postMessage({ type: 'init' }); }
    else if (!isBusy()) message('Live translation needs a one-time download first. Start it below when you are ready.');
  } else if (data.type === 'engine') {
    state.models = data.models; state.maxFrames = data.max_frames; state.translatorReady = true;
    if (!(state.model in state.models)) state.model = Object.keys(state.models)[0] || 'single';
    loadModel('single' in state.models ? 'single' : state.model); // the single model loads by default; the ensemble only on request
  } else if (data.type === 'progress') {
    if (state.download) Object.assign(state.download, { loaded: data.loaded, total: data.total });
  } else if (data.type === 'loaded') {
    state.loaded[data.model] = true; state.download = null;
    state.providers = data.providers || {};
    if (Number.isFinite(data.pose_probe_ms)) state.poseProbeMs = data.pose_probe_ms;
    toast(`${modelName(data.model)} is ready in this browser.`);
    if (data.model === state.model && state.phase === 'ready') message('Ready. Sign one short utterance, then select Finish.');
    else if (data.model === state.model && state.phase === 'off') message('Enable your camera to get started.');
  } else if (data.type === 'forgotten') {
    storeConsent(false); stopWorker();
    Object.assign(state, { consent: false, unavailable: '', models: {}, providers: {}, poseProbeMs: null, loadError: {} });
    toast('The downloaded models were removed from this browser.');
    startEngine(); // a fresh worker: check, then the consent card again
    return;
  } else if (data.type === 'unsupported') {
    if (data.model) { storeConsent(false); state.consent = false; } // refused after a download: ask again, never re-download
    unavailable(data.reason); return;
  }
  else if (data.type === 'engine_error') {
    const text = data.error || 'The translator could not be loaded.';
    if (state.forgetting) { // nothing else is running while models are removed
      state.forgetting = false;
      if (!state.translatorReady) stopWorker(); // the bare worker started from the unavailable state: allow a retry
      toast(`The downloaded models could not be removed. ${text}`); render(); return;
    }
    // Before the engine message nothing but init can have failed.
    if (data.stage === 'init' || !state.translatorReady) { unavailable(text); return; }
    toast(text);
    if (data.stage === 'runtime') { if (!isBusy()) message(text); } // the utterance's own error reply resets a recording
    else {
      const key = data.model ?? state.download?.model ?? state.model;
      state.download = null; state.loadError[key] = text;
      if (key === state.model && !isBusy()) message(text); // never over the recording instruction or another model
    }
  }
  render();
}
function fail(detail) {
  resetCurrent();
  message(detail || 'Something went wrong. Please try signing again.');
  idleResult('Let’s try that again.', 'The utterance could not be translated. Clear your space and try again.', 'Try again');
  $('resultBadge').classList.add('warning');
  toast(detail || 'Translation could not be completed.');
}
function startSigning() {
  if (!state.stream || !state.connected || ['starting', 'recording', 'processing'].includes(state.phase)) return;
  // app.py refuses this too; the in-browser worker cannot answer until the running translation returns.
  if (state.inflight) { message('Your previous utterance is still finishing. Please wait a moment.'); return; }
  resetCurrent();
  state.peopleCleared = false;
  state.id = crypto.randomUUID(); state.phase = 'starting';
  message('Getting ready to record…');
  if (!send(1, { type: 'start_recording', request_id: state.id, model: state.model })) { fail('The connection is not ready. Please try again.'); return; }
  state.responseTimer = setTimeout(() => fail('The server did not start recording. Please try again.'), 10000);
  render();
}
function beginCapture() {
  clearTimeout(state.responseTimer);
  state.phase = 'recording'; state.started = performance.now();
  message('Recording. Select Finish & translate when your utterance is complete.');
  render();
  // Capture on each camera-frame callback. Polling currentTime at the camera period aliases and loses ~20-30% of frames.
  // ponytail: a 30 fps camera sends every frame (the server resamples to 25 fps); max_frames 300 covers 10 s at 30 fps, a faster camera stops early.
  if (video.requestVideoFrameCallback) state.frameCallback = video.requestVideoFrameCallback(onVideoFrame);
  else { state.frameTimer = setInterval(() => sendFrame(), FALLBACK_POLL_MS); sendFrame(); }
  state.clockTimer = setInterval(() => {
    const seconds = Math.min(MAX_SECONDS, (performance.now() - state.started) / 1000);
    $('timer').firstChild.textContent = `00:${String(Math.floor(seconds)).padStart(2, '0')} `;
    $('captureProgress').style.width = `${seconds / MAX_SECONDS * 100}%`;
    if (seconds >= MAX_SECONDS || state.captured >= state.maxFrames) finishSigning(true);
  }, 100);
}
function onVideoFrame(_now, meta) {
  if (state.phase !== 'recording') return;
  // rVFC does not promise one callback per frame; presentedFrames exposes the ones it never delivered.
  if (state.lastPresented !== null) state.skipped += Math.max(0, meta.presentedFrames - state.lastPresented - 1) || 0;
  state.lastPresented = meta.presentedFrames;
  // Not mediaTime: WebKit leaves it at 0 for camera streams. Both fields share the performance.now clock.
  sendFrame((meta.captureTime ?? meta.presentationTime) / 1000);
  if (state.phase === 'recording') state.frameCallback = video.requestVideoFrameCallback(onVideoFrame);
}
function sendFrame(mediaTime = video.currentTime) {
  if (state.phase !== 'recording' || video.readyState < 2) return;
  // Stamp with camera media time, not the timer tick, so the server can place frames on the 25 fps grid.
  const t = state.t0 === null ? 0 : Math.round((mediaTime - state.t0) * 1000);
  if (t === state.lastCaptureAt) return; // the same camera frame read again by the fallback poll
  if (t - state.lastCaptureAt < MIN_FRAME_GAP_MS) { state.skipped++; return; }
  if (state.captured >= state.maxFrames || t > MAX_SECONDS * 1000) { finishSigning(true); return; }
  if (state.t0 === null) state.t0 = mediaTime;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const data = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
  if (state.queuedBytes + data.length > 32 * 1024 * 1024) {
    fail('This recording exceeded the capture limit. Try a shorter utterance.'); return;
  }
  state.queue.push({ type: 'frame', request_id: state.id, captured_at: t, data, queuedAt: performance.now() });
  state.queuedBytes += data.length; state.captured++; state.lastCaptureAt = t;
  pumpFrames();
}
function pumpFrames() {
  if (state.pendingFrame || !state.queue.length || !['recording', 'processing'].includes(state.phase)) return;
  const { queuedAt, ...frame } = state.queue.shift(); state.queuedBytes -= frame.data.length;
  state.queueMs.push(performance.now() - queuedAt);
  state.pendingFrame = send(0, frame);
  if (!state.pendingFrame) { fail('The connection was lost. Please try again.'); return; }
  // A live region: at most one progress update a second.
  if (state.finishPending && performance.now() - state.progressAt >= 1000) {
    state.progressAt = performance.now();
    message(`Preparing your translation · ${state.frames} of ${state.captured} frames processed.`);
  }
}
function finishSigning(atLimit = false) {
  if (state.phase !== 'recording') return;
  stopTimers(); state.phase = 'processing'; state.finishPending = true;
  state.finishTime = state.progressAt = performance.now();
  message(atLimit ? 'Recording limit reached. Translating the captured utterance…' : 'Signing finished. Preparing your translation…');
  idleResult('Finding the words…', 'Your complete utterance is being translated.', 'Translating');
  $('resultArea').classList.add('processing'); render();
  armFinishTimer();
  // Finish is sent only after the final in-flight pose frame has been acknowledged.
  if (!state.pendingFrame && !state.queue.length) submitTranslation();
}
// 120 s without progress after Finish. Each pose reply re-arms it: a device at the in-browser pose limit (800 ms/frame)
// needs minutes for a 10 s backlog, but never 120 s for one frame.
function armFinishTimer() {
  clearTimeout(state.responseTimer);
  state.responseTimer = setTimeout(() => fail('Translation is taking too long. Please try a shorter utterance.'), 120000);
}
function submitTranslation() {
  if (!state.finishPending) return;
  state.finishPending = false;
  if (state.frames < 2) { fail('Too few frames were captured. Please sign for a little longer and try again.'); return; }
  if (!send(1, { type: 'translate', request_id: state.id })) { fail('The connection was lost. Please try again.'); return; }
  state.inflight = state.id;
}
function showResult(data) {
  clearTimeout(state.responseTimer); stopAudio();
  state.result = data; state.phase = 'result'; state.finishMs = performance.now() - state.finishTime;
  $('resultArea').classList.remove('processing');
  const declined = data.accepted === false || data.reason?.startsWith('declined');
  const sample = state.sample || data.mode === 'sample';
  if (declined) {
    const c = data.coverage_target;
    const rate = c === 0.5 ? 'about half' : Number.isFinite(c) ? `about ${Math.round(c * 100)}%` : '';
    idleResult('Please sign that again.', data.reason === 'declined: empty'
      ? 'The model did not produce a translation. Try one short, clear utterance.'
      : `The model was not confident enough to offer a translation.${rate ? ` Its threshold is set to answer ${rate} of the stored validation clips; the rate on live signing may differ.` : ''} Try one short, clear utterance.`, 'Try again');
    $('resultBadge').classList.add('warning');
    message(data.reason === 'declined: empty' ? 'Try signing again. No translation was produced.' : 'Try signing again. This uncertain output will not be read aloud.');
  } else {
    $('resultEmpty').hidden = true; $('resultContent').hidden = false;
    $('resultText').textContent = data.text;
    $('resultNote').textContent = sample ? 'Sample phrase for testing the interface. This is not a translation of your signing.'
      : data.capture_fps < 24 ? `Captured at ${data.capture_fps.toFixed(1)} fps, below the 25 fps the model was trained on. Treat this result with extra caution.`
      : 'Review the wording before sharing. A translation appearing does not confirm that signing was recognised.';
    $('resultBadge').textContent = sample ? 'Sample output' : 'Ready to review';
    $('resultBadge').className = 'result-badge ' + (sample ? 'warning' : 'ready');
    message('Finished. Review the result, or sign another utterance.');
    $('audioMessage').textContent = localVoice ? `Voice: ${localVoice.name} · on this device${/^Karen\b/.test(localVoice.name) ? '' : ' · not the voice evaluated in the research'}` : 'No local English voice is available in this browser. You can still copy the text.';
    state.history.unshift({ id: state.id, text: data.text, sample, model: modelName(data.model), label: personLabel(data.persons),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) });
    state.history = state.history.slice(0, 10); renderHistory();
    if (localVoice && $('autoSpeak').checked) playSpeech();
  }
  state.retranslating = false;
  renderPersons(data);
  render();
  refocus(declined ? $('recordButton') : $('resultText'));
}
// People, when the in-browser engine reports who was in view (the Python service does not). The preview is mirrored, so a
// person at the right of the raw camera frame is at the left of what the visitor sees.
const seenAt = (x) => !Number.isFinite(x) ? '' : x > 0.6 ? ' on the left of your camera view' : x < 0.4 ? ' on the right of your camera view' : ' in the middle of your camera view';
const showsPeople = (p) => !!p && p.count >= 1 && (p.count >= 2 || p.session_people >= 2);
const personLabel = (p) => (showsPeople(p) && p.chosen) || null;
function peopleSummary(p) {
  const chosen = p.chosen ? `translated ${p.chosen}` : 'no one picked';
  return p.count ? `${p.count} in view · ${chosen}${p.mode === 'manual' ? ' (your choice)' : ''}` : 'Nobody counted';
}
function personSentence(p, declined) {
  const at = seenAt(p.people.find((q) => q.label === p.chosen)?.x);
  const inView = `${p.count} ${p.count === 1 ? 'person was' : 'people were'} in view.`;
  if (p.mode === 'single') return `${p.chosen}${at}. ${inView}`;
  if (p.mode === 'default') return `${inView} Nobody stayed in view long enough to compare their movement, so the translator followed the person it was already following.`;
  const how = p.mode === 'manual' ? 'as you chose' : p.mode === 'online' ? 'the person the translator was already following'
    : 'the person moving most like a signer. That is a guess, not a recognition';
  return declined ? `${p.chosen}${at} was picked (${how}), but the model was not confident enough to translate it. ${inView}`
    : `Translated ${p.chosen}${at}, ${how}. ${inView}`;
}
function renderPersons(data) {
  const p = data.persons, box = $('personBox'), switcher = $('personSwitch');
  switcher.replaceChildren();
  box.hidden = state.sample || !showsPeople(p);
  if (box.hidden) return;
  $('personText').textContent = personSentence(p, data.accepted === false || !!data.reason?.startsWith('declined'));
  if (p.mode === 'single' || state.peopleCleared) return;   // after Clear conversation the worker no longer holds anyone to switch to
  for (const q of p.people) {
    if (q.label === p.chosen || !q.eligible) continue;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'button button-outline person-button';
    b.textContent = `Translate ${q.label} instead`;
    b.addEventListener('click', () => retranslate(q.label));
    switcher.append(b);
  }
}
// Same utterance, another person: the worker still holds the people's keypoints (persons.js) until the next utterance.
function retranslate(label) {
  if (state.phase !== 'result' || !state.id || state.inflight) return;
  stopAudio();
  state.phase = 'processing'; state.finishTime = performance.now(); state.retranslating = true;
  state.result = null;
  state.history = state.history.filter((e) => e.id !== state.id); renderHistory();   // a switch replaces that utterance's entry
  idleResult('Finding the words…', `Translating ${label} from the same utterance.`, 'Translating');
  $('resultArea').classList.add('processing'); render();
  armFinishTimer();
  if (!send(1, { type: 'retranslate', request_id: state.id, person: label })) { fail('The connection was lost. Please try again.'); return; }
  state.inflight = state.id;
}
// Clear conversation and camera off: the worker forgets who was in view (labels, positions, a retained utterance).
function forgetPeople() { state.peopleCleared = true; if (state.engine === 'browser') send(1, { type: 'clear_session' }); }
function playSpeech() {
  if (!state.result?.text || state.result.accepted === false || !localVoice) return;
  if (state.speaking) { stopAudio(); return; }
  const generation = ++speechGeneration, requested = performance.now();
  const utterance = new SpeechSynthesisUtterance(state.result.text);
  utterance.voice = localVoice;
  utterance.lang = localVoice.lang;
  utterance.rate = 1;
  utterance.onstart = () => {
    if (generation !== speechGeneration) return;
    state.speaking = true; state.speechMs = performance.now() - requested; render();
    $('speakLabel').textContent = 'Stop speech';
    $('audioMessage').textContent = 'Playing English audio on this device.';
  };
  utterance.onend = () => {
    if (generation !== speechGeneration) return;
    state.speaking = false; $('speakLabel').textContent = 'Read aloud';
    $('audioMessage').textContent = 'Playback finished. You can listen again.';
  };
  utterance.onerror = () => {
    if (generation !== speechGeneration) return;
    state.speaking = false; $('speakLabel').textContent = 'Read aloud';
    $('audioMessage').textContent = 'Speech could not play. Try again or copy the text.';
  };
  window.speechSynthesis.speak(utterance);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied to clipboard.'); }
  catch { toast('Copy is unavailable. Select the translation text and copy it manually.'); }
}
function renderHistory() {
  const list = $('historyList'); list.replaceChildren();
  $('historyCount').textContent = state.history.length;
  $('historyEmpty').hidden = !!state.history.length;
  $('copySessionButton').disabled = $('clearSessionButton').disabled = !state.history.length;
  for (const entry of state.history) {
    const item = document.createElement('li'); const body = document.createElement('div');
    const stamp = document.createElement('span'); stamp.textContent = `${entry.time} · ${entry.label ? `${entry.label} · ` : ''}${entry.sample ? 'Sample phrase' : entry.model}`;
    const text = document.createElement('p'); text.textContent = entry.text;
    const copy = document.createElement('button'); copy.type = 'button'; copy.textContent = 'Copy';
    copy.setAttribute('aria-label', `Copy: ${entry.text}`); copy.addEventListener('click', () => copyText(entry.text));
    body.append(stamp, text); item.append(body, copy); list.append(item);
  }
}
$('enableCameraButton').addEventListener('click', enableCamera);
$('cameraOffButton').addEventListener('click', disableCamera);
$('recordButton').addEventListener('click', () => state.phase === 'recording' ? finishSigning() : startSigning());
$('clearButton').addEventListener('click', () => { resetCurrent(); refocus($('recordButton')); });
$('guideToggle').addEventListener('change', render);
for (const key of Object.keys(MODEL_NAMES)) modelEl(key).addEventListener('change', () => { if (modelEl(key).checked) { state.model = key; render(); } });
$('copyButton').addEventListener('click', () => { if (state.result?.text) copyText(state.result.text); });
$('copySessionButton').addEventListener('click', () => copyText([...state.history].reverse().map(e => `${e.sample ? '[Sample] ' : ''}${e.label ? `${e.label}: ` : ''}${e.text}`).join('\n')));
$('clearSessionButton').addEventListener('click', () => { state.history = []; forgetPeople(); $('personBox').hidden = true; $('personSwitch').replaceChildren(); renderHistory(); toast('Conversation cleared.'); refocus($('sessionTitle')); });
$('speakButton').addEventListener('click', playSpeech);
$('autoSpeak').addEventListener('change', () => { if (!$('autoSpeak').checked) stopAudio(); });
document.querySelector('.help-link').addEventListener('click', () => { $('guide').open = true; });
// The clicked button is hidden next, so focus moves to the status text.
$('downloadButton').addEventListener('click', () => { loadModel(state.model); refocus($('captureMessage')); });
$('consentButton').addEventListener('click', () => {
  storeConsent(true); state.consent = true;
  state.worker?.postMessage({ type: 'init' });
  message('Checking the model files…');
  render(); refocus($('captureMessage'));
});
$('retryButton').addEventListener('click', () => {
  state.unavailable = ''; state.loadError = {};
  message('Checking this browser…');
  startEngine(); refocus($('captureMessage'));
});
$('forgetButton').addEventListener('click', () => {
  if (!canForget()) return;
  // Unavailable (offline, manifest error): a bare worker deletes the stored files; forget needs no manifest or network.
  if (!state.worker && !spawnWorker(() => { stopWorker(); toast('The downloaded models could not be removed.'); render(); })) {
    toast('The downloaded models could not be removed.'); return;
  }
  state.forgetting = true; state.worker.postMessage({ type: 'forget' }); render();
});
window.addEventListener('pagehide', () => { state.disposed = true; disableCamera(); clearInterval(state.heartbeat); clearTimeout(state.reconnectTimer); state.sockets.forEach(ws => { ws.onclose = ws.onmessage = ws.onopen = null; ws.close(); });
  if (state.engine === 'browser') stopWorker(); }); // models reload from Cache Storage
window.addEventListener('pageshow', (event) => { if (event.persisted) { state.disposed = false; boot(); } });
document.addEventListener('visibilitychange', () => { if (document.hidden && ['recording', 'starting'].includes(state.phase)) { resetCurrent(); message('Recording stopped because this tab was hidden. Start again when you’re ready.'); } });
async function boot() {
  if (state.disposed) return;
  if (state.engine !== 'browser') await updateHealth();
  if (state.engine === 'browser') startEngine();
  else if (state.translatorReady) connect();
  else {
    $('connectionBadge').className = 'connection disconnected';
    $('connectionText').textContent = 'Translator offline';
    state.reconnectTimer = setTimeout(boot, 15000);
  }
}
window.speechSynthesis?.addEventListener('voiceschanged', updateVoices);
render(); updateVoices(); boot();
