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
};

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
    info.needs_frames && 'Adds about a second after Finish for image features'].filter(Boolean).join('. ') + '.' : '';
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
  };
  for (const [id, text] of Object.entries(details)) $(id).textContent = text;
  ['stepCamera', 'stepSign', 'stepReview'].forEach((id) => $(id).removeAttribute('aria-current'));
  $(state.result ? 'stepReview' : cameraOn ? 'stepSign' : 'stepCamera').setAttribute('aria-current', 'step');
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
  state.result = null;
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
    message(state.connected ? 'Ready. Sign one short utterance, then select Finish.' : 'Camera ready. Waiting for the translation service.');
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
  refocus($('enableCameraButton'));
}

async function updateHealth() {
  try {
    const res = await fetch('/health', { cache: 'no-store', signal: AbortSignal.timeout(8000) });
    if (!res.ok) throw Error('Health check failed');
    const data = await res.json();
    state.sample = data.mode === 'sample';
    state.translatorReady = !!data.translator;
    state.maxFrames = data.max_frames || 300; // the server refuses frame max_frames + 1
    state.models = data.models || {};
    if (!(state.model in state.models)) state.model = Object.keys(state.models)[0] || 'single';
    $('modeNotice').hidden = !state.sample && state.translatorReady;
    $('modeNotice').querySelector('strong').textContent = state.sample ? 'Interface preview' : 'Translation service offline';
    $('modeNotice').querySelector('p span').textContent = state.sample ? 'Sample phrases will be shown. A trained translation model is not loaded.' : 'The research showcase is available. Live translation will be enabled when the model service is ready.';
    $('engineDetail').textContent = Object.keys(state.models).map(modelName).join(' · ') || 'Unavailable';
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
      if (data.request_id && data.request_id !== state.id) return;
      if (data.error) { fail(data.error); return; }
      if (index === 0 && (data.type === 'pose' || data.type === 'frame_ignored')) {
        state.pendingFrame = false;
        if (data.type === 'pose') {
          state.frames = data.buffer_len;
          if (Number.isFinite(data.pose_ms)) state.poseMs.push(data.pose_ms);
          render();
          if (data.person_count > 1) warning('More than one person is visible. The translator follows one signer; keep only the signer in frame if you can.');
          else if (!data.has_person) warning('Move into the frame so your face, hands and upper body are visible.');
          else if (state.captured > TARGET_FPS && captureRate() < 24) warning('Capture is running below the 25 fps the model was trained on (rate in Session details). Translation may be affected.');
          else warning();
        }
        pumpFrames();
        if (state.finishPending && !state.pendingFrame && !state.queue.length) submitTranslation();
      } else if (data.type === 'recording_started' && state.phase === 'starting') beginCapture();
      else if (data.type === 'result' && state.phase === 'processing' && state.id) showResult(data);
    };
    ws.onerror = () => { /* Close handles recovery once for the connection pair. */ };
    ws.onclose = () => {
      if (state.disposed) return;
      state.connected = false; clearInterval(state.heartbeat);
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
function fail(detail) {
  resetCurrent();
  message(detail || 'Something went wrong. Please try signing again.');
  idleResult('Let’s try that again.', 'The utterance could not be translated. Clear your space and try again.', 'Try again');
  $('resultBadge').classList.add('warning');
  toast(detail || 'Translation could not be completed.');
}
function startSigning() {
  if (!state.stream || !state.connected || ['starting', 'recording', 'processing'].includes(state.phase)) return;
  resetCurrent();
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
  state.responseTimer = setTimeout(() => fail('Translation is taking too long. Please try a shorter utterance.'), 120000);
  // Finish is sent only after the final in-flight pose frame has been acknowledged.
  if (!state.pendingFrame && !state.queue.length) submitTranslation();
}
function submitTranslation() {
  if (!state.finishPending) return;
  state.finishPending = false;
  if (state.frames < 2) { fail('Too few frames were captured. Please sign for a little longer and try again.'); return; }
  if (!send(1, { type: 'translate', request_id: state.id })) fail('The connection was lost. Please try again.');
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
    state.history.unshift({ text: data.text, sample, model: modelName(data.model), time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) });
    state.history = state.history.slice(0, 10); renderHistory();
    if (localVoice && $('autoSpeak').checked) playSpeech();
  }
  render();
  refocus(declined ? $('recordButton') : $('resultText'));
}
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
    const stamp = document.createElement('span'); stamp.textContent = `${entry.time} · ${entry.sample ? 'Sample phrase' : entry.model}`;
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
$('copySessionButton').addEventListener('click', () => copyText([...state.history].reverse().map(e => `${e.sample ? '[Sample] ' : ''}${e.text}`).join('\n')));
$('clearSessionButton').addEventListener('click', () => { state.history = []; renderHistory(); toast('Conversation cleared.'); refocus($('sessionTitle')); });
$('speakButton').addEventListener('click', playSpeech);
$('autoSpeak').addEventListener('change', () => { if (!$('autoSpeak').checked) stopAudio(); });
document.querySelector('.help-link').addEventListener('click', () => { $('guide').open = true; });
window.addEventListener('pagehide', () => { state.disposed = true; disableCamera(); clearInterval(state.heartbeat); clearTimeout(state.reconnectTimer); state.sockets.forEach(ws => { ws.onclose = ws.onmessage = ws.onopen = null; ws.close(); }); });
window.addEventListener('pageshow', (event) => { if (event.persisted) { state.disposed = false; boot(); } });
document.addEventListener('visibilitychange', () => { if (document.hidden && ['recording', 'starting'].includes(state.phase)) { resetCurrent(); message('Recording stopped because this tab was hidden. Start again when you’re ready.'); } });
async function boot() {
  if (state.disposed) return;
  await updateHealth();
  if (state.translatorReady) connect();
  else {
    $('connectionBadge').className = 'connection disconnected';
    $('connectionText').textContent = 'Translator offline';
    state.reconnectTimer = setTimeout(boot, 15000);
  }
}
window.speechSynthesis?.addEventListener('voiceschanged', updateVoices);
render(); updateVoices(); boot();
