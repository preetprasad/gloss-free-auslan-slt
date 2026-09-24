/* Behavioural tests for the browser controller using fake camera/transport adapters.
 * No physical camera, microphone, clipboard or browser permissions are used. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../static/studio.html'), 'utf8');
const script = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');

let focused = null; // document.activeElement; focus() refuses disabled or hidden controls, as a browser does
class Element {
  constructor() {
    this.style = {}; this.listeners = {}; this.children = []; this.checked = false;
    this.firstChild = {textContent: ''}; this.textContent = ''; this.hidden = false;
    this.classList = {toggle() {}, add() {}, remove() {}};
  }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  removeAttribute(name) { delete this[name]; }
  setAttribute(name, value) { this[name] = value; }
  querySelector(name) { return this.children[name] ||= new Element(); }
  append(...values) { this.children.push(...values); }
  replaceChildren() { this.children = []; }
  focus() { if (!this.disabled && !this.hidden) focused = this; }
}
const HEALTH = {mode: 'model', translator: true, coverage_target: 0.5, max_frames: 300, models: {
  single: {label: 'Single best model · adopted recipe (pose augmentation + 1.1:1 duration stitch), wave-4 seed 202 · pose only', threshold: 0.706661,
    coverage: 0.5, beam: 10, length_penalty: 0.7, needs_frames: false, validation_bleu4: 12.643, test_bleu4: 11.652},
  ensemble: {label: 'Six-model ensemble (S6) · 3 pose-only + 3 pose+RGB members', threshold: 0.904779,
    coverage: 0.5, beam: 10, length_penalty: 0.3, needs_frames: true, validation_bleu4: 12.47, test_bleu4: 11.802}}};
const LOCAL_VOICE = {name: 'Local', localService: true, lang: 'en-AU'};
async function setup({voice = false, voices = voice ? [LOCAL_VOICE] : [], health = HEALTH, rvfc = false} = {}) {
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const video = elements.videoElement;
  Object.assign(video, {videoWidth: 640, videoHeight: 480, readyState: 4, currentTime: 0, play: async () => {}});
  const frameCallbacks = new Map(); let nextCallback = 1;
  if (rvfc) Object.assign(video, {requestVideoFrameCallback: cb => { frameCallbacks.set(nextCallback, cb); return nextCallback++; },
    cancelVideoFrameCallback: id => frameCallbacks.delete(id)});
  let stopped = false, now = 0, spoken = 0, presented = 0;
  const track = {stop() {stopped = true;}, getSettings: () => ({width: 640, height: 480, frameRate: 25})};
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  class Socket {
    static OPEN = 1;
    constructor(url) {this.url = url; this.readyState = 0; this.bufferedAmount = 0; this.sent = [];}
    send(value) {this.sent.push(JSON.parse(value));}
    close() {this.readyState = 3;}
    receive(data) {this.onmessage?.({data: JSON.stringify(data)});}
  }
  const window = new Element();
  window.speechSynthesis = {getVoices: () => voices, cancel() {}, addEventListener() {}, speak(u) {spoken++; u.onstart();}};
  const document = new Element();
  document.body = new Element(); focused = document.body;
  Object.defineProperty(document, 'activeElement', {get: () => focused});
  Object.assign(document, {getElementById: id => elements[id], querySelector: () => new Element(), createElement: tag => {
    const e = new Element(); if (tag === 'canvas') Object.assign(e, {getContext: () => ({drawImage() {}}), toDataURL: () => 'data:image/jpeg;base64,test'}); return e;
  }});
  const context = vm.createContext({document, window, navigator: {mediaDevices: {getUserMedia: async () => stream}, clipboard: {writeText: async () => {}}},
    crypto, WebSocket: Socket, location: {protocol: 'http:', host: 'localhost'}, performance: {now: () => now},
    fetch: async () => ({ok: true, json: async () => health}), AbortSignal, SpeechSynthesisUtterance: class {constructor(text) {this.text = text;}},
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {}, console});
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  const run = code => vm.runInContext(code, context);
  const sockets = run('state.sockets'); sockets.forEach(s => s.readyState = 1); sockets.forEach(s => s.onopen());
  return {elements, run, sockets, video, window, frameCallbacks, stopped: () => stopped, spoken: () => spoken,
    // One rVFC callback; Chrome-like metadata by default (mediaTime and presentationTime advance together).
    present(seconds, extra = {}) {const meta = {mediaTime: seconds, presentationTime: seconds * 1000, presentedFrames: ++presented, ...extra};
      const cbs = [...frameCallbacks.values()]; frameCallbacks.clear(); cbs.forEach(cb => cb(now, meta));},
    advance(ms=40, media=ms/1000) {now += ms; video.currentTime += media;},
    async start() {await run('enableCamera()'); run('startSigning()'); sockets[1].receive({type: 'recording_started', request_id: run('state.id')});},
    ack(n, extra = {}) {sockets[0].receive({type: 'pose', request_id: run('state.id'), buffer_len: n, person_count: 1, has_person: true, ...extra});},
    async finish(result) {await this.start(); this.ack(1); this.advance(); run('sendFrame()'); this.ack(2); run('finishSigning()');
      sockets[1].receive({type: 'result', request_id: run('state.id'), ...result});}};
}

test('camera stays off until requested; remote voices cannot trigger speech', async () => {
  const app = await setup();
  assert.equal(app.run('state.stream'), null);
  assert.equal(app.elements.autoSpeak.disabled, true);
  assert.equal(app.elements.recordButton.disabled, true);
});

test('slow transport preserves captured frames and Finish waits for the last acknowledgement', async () => {
  const app = await setup(); await app.start();
  app.advance(); app.run('sendFrame()'); app.advance(); app.run('sendFrame()');
  assert.equal(app.run('state.queue.length'), 2);
  app.run('finishSigning()');
  assert.equal(app.sockets[1].sent.some(m => m.type === 'translate'), false);
  app.ack(1);
  assert.equal(app.elements.captureMessage.textContent, 'Signing finished. Preparing your translation…'); // live region: at most 1 update/s
  app.advance(1000); app.ack(2);
  assert.equal(app.elements.captureMessage.textContent, 'Preparing your translation · 2 of 3 frames processed.');
  assert.equal(app.sockets[1].sent.some(m => m.type === 'translate'), false);
  app.ack(3);
  assert.equal(app.sockets[1].sent.at(-1).type, 'translate');
  assert.deepEqual(app.sockets[0].sent.map(m => m.captured_at), [0, 40, 80]);
  assert.equal(app.run('state.queue.length'), 0);
});

test('cancel discards frames and ignores a stale result', async () => {
  const app = await setup(); await app.start(); const oldId = app.run('state.id');
  app.advance(); app.run('sendFrame(); resetCurrent()');
  app.sockets[1].receive({type: 'result', request_id: oldId, text: 'Old words', accepted: true});
  assert.equal(app.run('state.result'), null);
  assert.equal(app.run('state.queue.length'), 0);
  assert.equal(app.elements.copyButton.disabled, true);
});

test('declined output never enters history or speech even with autoplay selected', async () => {
  const app = await setup({voice: true}); await app.start(); app.ack(1); app.advance(); app.run('sendFrame()'); app.ack(2); app.run('finishSigning()');
  app.elements.autoSpeak.checked = true;
  app.sockets[1].receive({type: 'result', request_id: app.run('state.id'), text: '', accepted: false, confidence: 5});
  assert.equal(app.run('state.history.length'), 0);
  assert.equal(app.spoken(), 0);
  assert.equal(app.elements.speakButton.disabled, true);
});

test('accepted output is reviewed first, safely rendered as text, and camera-off cancels capture', async () => {
  const app = await setup({voice: true}); await app.start(); app.ack(1); app.advance(); app.run('sendFrame()'); app.ack(2); app.run('finishSigning()');
  app.sockets[1].receive({type: 'result', request_id: app.run('state.id'), text: '<script>inert text</script>', accepted: true, confidence: 1});
  assert.equal(app.elements.resultText.textContent, '<script>inert text</script>');
  assert.equal(app.spoken(), 0);
  assert.equal(app.run('state.history.length'), 1);
  app.run('playSpeech()'); assert.equal(app.spoken(), 1);
  app.run('disableCamera()'); assert.equal(app.stopped(), true);
  assert.equal(app.run('state.phase'), 'off');
});

test('a tab interruption cancels recording and clears local frames', async () => {
  const app = await setup(); await app.start();
  app.window.listeners.pagehide();
  assert.equal(app.stopped(), true);
  assert.equal(app.run('state.disposed'), true);
  assert.equal(app.run('state.queue.length'), 0);
});

test('start_recording carries the selected model; single is the default', async () => {
  const app = await setup(); await app.start();
  assert.equal(app.sockets[1].sent.find(m => m.type === 'start_recording').model, 'single');
  assert.match(app.elements.modelDetail.textContent, /Validation-best of 21 adopted-recipe runs \(chosen on validation; its test score is not the highest of the 21\)\. Stored benchmark clips: validation 12\.643 · test 11\.652 BLEU-4 \(not measured on live camera input\)/);
  assert.equal(app.run('state.maxFrames'), 300);
  assert.doesNotMatch(app.elements.modelDetail.textContent, /image features/);
});

test('switching to the ensemble is reflected in the detail line and the request', async () => {
  const app = await setup();
  app.elements.modelEnsemble.checked = true; app.elements.modelEnsemble.listeners.change();
  assert.equal(app.run('state.model'), 'ensemble');
  assert.match(app.elements.modelDetail.textContent, /Stored benchmark clips: validation 12\.470 · test 11\.802 BLEU-4 \(not measured on live camera input\)/);
  assert.match(app.elements.modelDetail.textContent, /image features/);
  assert.doesNotMatch(app.elements.modelDetail.textContent, /Validation-best/);
  await app.start();
  assert.equal(app.sockets[1].sent.find(m => m.type === 'start_recording').model, 'ensemble');
});

test('model control is disabled while recording', async () => {
  const app = await setup();
  assert.equal(app.elements.modelSingle.disabled, false);
  await app.start();
  assert.equal(app.run('state.phase'), 'recording');
  assert.equal(app.elements.modelSingle.disabled, true);
  assert.equal(app.elements.modelEnsemble.disabled, true);
});

test('an unavailable model is hidden and a lone model shows as fixed text', async () => {
  const app = await setup({health: {...HEALTH, models: {single: HEALTH.models.single}}});
  assert.equal(app.elements.modelEnsembleOption.hidden, true);
  assert.equal(app.elements.modelSingleOption.hidden, true);
  assert.equal(app.elements.modelFixed.hidden, false);
  assert.equal(app.elements.modelFixed.textContent, 'Single best model');
});

test('offline service keeps the camera button disabled', async () => {
  const app = await setup({health: {mode: 'showcase', translator: false, models: {}}});
  assert.equal(app.elements.enableCameraButton.disabled, true);
  assert.equal(app.elements.modelChoice.hidden, true);
});

test('captured_at follows camera media time, not the timer, and repeats are skipped', async () => {
  const app = await setup(); await app.start();
  app.advance(40, 0.1); app.run('sendFrame()');
  app.advance(40, 0.01); app.run('sendFrame()');
  assert.equal(app.run('state.skipped'), 1);
  app.ack(1);
  assert.deepEqual(app.sockets[0].sent.map(m => m.captured_at), [0, 100]);
  assert.equal('queuedAt' in app.sockets[0].sent[1], false);
});

test('a new utterance resets the media-time origin', async () => {
  const app = await setup(); await app.start();
  app.advance(400); app.run('sendFrame()');
  app.run('resetCurrent()'); app.advance(40); app.run('startSigning()');
  app.sockets[1].receive({type: 'recording_started', request_id: app.run('state.id')});
  assert.equal(app.sockets[0].sent.at(-1).captured_at, 0);
});

test('the research voice (Karen, en-AU) is preferred over another local en-AU voice', async () => {
  const app = await setup({voices: [LOCAL_VOICE, {name: 'Karen', localService: true, lang: 'en-AU'}]});
  assert.equal(app.run('localVoice.name'), 'Karen');
  await app.finish({text: 'hello', accepted: true, reason: 'ready', confidence: 0.5});
  assert.doesNotMatch(app.elements.audioMessage.textContent, /not the voice evaluated/);
});

test('a different local voice is labelled as not the evaluated one', async () => {
  const app = await setup({voice: true});
  await app.finish({text: 'hello', accepted: true, reason: 'ready', confidence: 0.5});
  assert.match(app.elements.audioMessage.textContent, /Local · on this device · not the voice evaluated in the research/);
});

test('pose_ms and result telemetry fill the session details', async () => {
  const app = await setup(); await app.start();
  app.ack(1, {pose_ms: 190}); app.advance(); app.run('sendFrame()'); app.ack(2, {pose_ms: 210}); app.run('finishSigning()');
  assert.equal(app.elements.poseDetail.textContent, '200 / 209 ms');
  app.sockets[1].receive({type: 'result', request_id: app.run('state.id'), text: 'hello', accepted: true, reason: 'ready',
    confidence: 0.6, threshold: 0.706661, coverage_target: 0.5, model: 'single', model_label: HEALTH.models.single.label,
    capture_fps: 25, frames: 2, frames_used: 2, translate_ms: 41.2, rgb_ms: 0});
  assert.equal(app.elements.thresholdDetail.textContent, '0.707 nats · 50%');
  assert.equal(app.elements.captureRateDetail.textContent, '25.0 fps · 2 frames on the 25 fps grid');
  assert.equal(app.elements.translateDetail.textContent, '41 ms');
  assert.equal(app.elements.rgbDetail.textContent, 'Not used');
  assert.equal(app.elements.capturedDetail.textContent, '2');
  assert.equal(app.run('state.history[0].model'), 'Single best model');
});

test('declined outputs explain empty output and the answer rate', async () => {
  const empty = await setup();
  await empty.finish({text: '', accepted: false, reason: 'declined: empty', confidence: null, coverage_target: 0.5});
  assert.equal(empty.elements.resultHint.textContent, 'The model did not produce a translation. Try one short, clear utterance.');
  assert.equal(empty.elements.captureMessage.textContent, 'Try signing again. No translation was produced.');
  const unsure = await setup();
  await unsure.finish({text: '', accepted: false, reason: 'declined: low confidence', confidence: 1.4, coverage_target: 0.5});
  assert.match(unsure.elements.resultHint.textContent, /threshold is set to answer about half of the stored validation clips; the rate on live signing may differ\./);
  assert.equal(unsure.elements.captureMessage.textContent, 'Try signing again. This uncertain output will not be read aloud.');
});

test('low capture rate replaces the review note with a caution', async () => {
  const app = await setup();
  await app.finish({text: 'hello', accepted: true, reason: 'ready', confidence: 0.5, capture_fps: 14.6});
  assert.equal(app.elements.resultNote.textContent, 'Captured at 14.6 fps, below the 25 fps the model was trained on. Treat this result with extra caution.');
  const ok = await setup();
  await ok.finish({text: 'hello', accepted: true, reason: 'ready', confidence: 0.5, capture_fps: 25});
  assert.match(ok.elements.resultNote.textContent, /does not confirm that signing was recognised/);
});

test('with requestVideoFrameCallback each delivered 25 fps camera frame is captured once', async () => {
  const app = await setup({rvfc: true}); await app.start();
  assert.equal(app.run('state.captured'), 0); // no timer capture; the first presented frame starts the clock
  for (let k = 0; k < 100; k++) { app.present(7 + k * 0.04); app.ack(k + 1); }
  assert.equal(app.run('state.captured'), 100);
  assert.equal(app.run('state.skipped'), 0);
  assert.deepEqual(app.sockets[0].sent.map(m => m.captured_at), Array.from({length: 100}, (_, k) => k * 40));
  assert.equal(app.elements.captureWarning.hidden, true);
  app.run('finishSigning()');
  assert.equal(app.frameCallbacks.size, 0);
});

test('a capture rate below 24 fps is reported as a rate, not blamed on missing camera frames', async () => {
  const app = await setup({rvfc: true}); await app.start();
  let text = app.elements.captureWarning.textContent, writes = 0; // captureWarning is a live region
  Object.defineProperty(app.elements.captureWarning, 'textContent', {get: () => text, set: v => { writes++; text = v; }});
  for (let k = 0; k < 30; k++) { app.present(k * 0.06); app.ack(k + 1); }
  assert.equal(text, 'Capture is running below the 25 fps the model was trained on (rate in Session details). Translation may be affected.');
  assert.equal(writes, 1); // written once, not re-announced on every frame
  assert.equal(app.elements.captureRateDetail.textContent, '16.7 fps so far');
});

test('WebKit camera frames (mediaTime always 0) are timed by presentationTime; captureTime wins when present', async () => {
  const app = await setup({rvfc: true}); await app.start();
  for (let k = 0; k < 30; k++) { app.present(1000 + k * 0.04, {mediaTime: 0}); app.ack(k + 1); }
  assert.equal(app.run('state.captured'), 30);
  assert.equal(app.run('state.skipped'), 0);
  assert.deepEqual(app.sockets[0].sent.map(m => m.captured_at), Array.from({length: 30}, (_, k) => k * 40));
  const chrome = await setup({rvfc: true}); await chrome.start();
  for (let k = 0; k < 5; k++) { chrome.present(k * 0.04, {presentationTime: 5000 + k * 40 + (k % 2) * 15, captureTime: 2000 + k * 40}); chrome.ack(k + 1); }
  assert.deepEqual(chrome.sockets[0].sent.map(m => m.captured_at), [0, 40, 80, 120, 160]);
});

test('camera frames requestVideoFrameCallback never delivered are counted as skipped', async () => {
  const app = await setup({rvfc: true}); await app.start();
  [[0, 10], [0.04, 11], [0.16, 14], [0.2, 15]].forEach(([t, n], k) => { app.present(t, {presentedFrames: n}); app.ack(k + 1); });
  assert.equal(app.run('state.captured'), 4);
  assert.equal(app.run('state.skipped'), 2);
});

test('the fallback poll ignores repeated reads of one camera frame', async () => {
  const app = await setup(); await app.start(); // frame at t = 0; the poll then reads the same media time again
  app.run('sendFrame(); sendFrame()');
  assert.equal(app.run('state.captured'), 1);
  assert.equal(app.run('state.skipped'), 0);
});

test('keyboard focus stays on a usable control through the whole utterance', async () => {
  const app = await setup({voice: true}); const e = app.elements;
  const active = () => app.run('document.activeElement');
  e.enableCameraButton.closest = () => e.cameraPlaceholder.hidden ? e.cameraPlaceholder : null;
  e.enableCameraButton.focus(); await e.enableCameraButton.listeners.click();
  assert.equal(active(), e.recordButton);
  e.recordButton.listeners.click(); // starting: aria-disabled, never disabled, so focus is kept
  assert.equal(e.recordButton.disabled, false); assert.equal(e.recordButton['aria-disabled'], 'true');
  app.sockets[1].receive({type: 'recording_started', request_id: app.run('state.id')});
  app.ack(1); app.advance(); app.run('sendFrame()'); app.ack(2);
  e.recordButton.listeners.click(); // Finish
  assert.equal(app.run('state.phase'), 'processing');
  assert.equal(active(), e.recordButton); assert.equal(e.recordButton['aria-disabled'], 'true');
  assert.equal(e.recordLabel.textContent, 'Translating…');
  e.recordButton.listeners.click(); // ignored while processing
  assert.equal(app.run('state.phase'), 'processing');
  app.sockets[1].receive({type: 'result', request_id: app.run('state.id'), text: 'hello', accepted: true, reason: 'ready', confidence: 0.5});
  assert.equal(active(), e.recordButton); assert.equal(e.recordButton['aria-disabled'], 'false');
  e.clearButton.focus(); e.clearButton.listeners.click();
  assert.equal(active(), e.recordButton);
  e.clearSessionButton.focus(); e.clearSessionButton.listeners.click();
  assert.equal(active(), e.sessionTitle);
  e.cameraOffButton.focus(); e.cameraOffButton.listeners.click();
  assert.equal(active(), e.enableCameraButton);
});

test('with only the ensemble published, the model falls back to it', async () => {
  const app = await setup({health: {...HEALTH, models: {ensemble: HEALTH.models.ensemble}}});
  assert.equal(app.run('state.model'), 'ensemble');
  assert.equal(app.elements.modelFixed.textContent, 'Six-model ensemble');
  await app.start();
  assert.equal(app.sockets[1].sent.find(m => m.type === 'start_recording').model, 'ensemble');
});

test('interface preview never claims the research selection note', async () => {
  const app = await setup({health: {mode: 'sample', translator: true, max_frames: 300,
    models: {single: {label: 'Interface preview · sample phrases', threshold: 3, coverage: null, needs_frames: false}}}});
  assert.equal(app.elements.modelDetail.textContent, 'Interface preview · sample phrases.');
});

test('the server max_frames ends capture; no frame beyond it is sent', async () => {
  const app = await setup({health: {...HEALTH, max_frames: 3}});
  await app.start(); app.ack(1);
  for (let k = 2; k <= 3; k++) { app.advance(); app.run('sendFrame()'); app.ack(k); }
  app.advance(); app.run('sendFrame()');
  assert.equal(app.sockets[0].sent.length, 3);
  assert.equal(app.run('state.phase'), 'processing');
  assert.equal(app.sockets[1].sent.at(-1).type, 'translate');
  assert.equal(app.elements.captureMessage.textContent, 'Recording limit reached. Translating the captured utterance…');
});

test('capture ends after 10 s of camera time', async () => {
  const app = await setup(); await app.start();
  app.ack(1); app.advance(); app.run('sendFrame()'); app.ack(2);
  app.advance(40, 10); app.run('sendFrame()'); // camera time 10.08 s
  assert.equal(app.sockets[0].sent.length, 2);
  assert.equal(app.run('state.phase'), 'processing');
  assert.equal(app.sockets[1].sent.at(-1).type, 'translate');
});

test('a stale pose acknowledgement from a cancelled utterance does not release the next one', async () => {
  const app = await setup(); await app.start(); const oldId = app.run('state.id');
  app.run('resetCurrent()'); app.advance(); app.run('startSigning()');
  app.sockets[1].receive({type: 'recording_started', request_id: app.run('state.id')});
  const newId = app.run('state.id');
  app.advance(); app.run('sendFrame()'); // queued behind the in-flight first frame
  for (const type of ['frame_ignored', 'pose']) app.sockets[0].receive({type, request_id: oldId, buffer_len: 9, has_person: true});
  assert.equal(app.run('state.pendingFrame'), true);
  assert.equal(app.sockets[0].sent.filter(m => m.request_id === newId).length, 1);
  assert.equal(app.run('state.frames'), 0);
});
