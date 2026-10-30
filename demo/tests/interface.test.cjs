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
// The in-browser engine as the page sees it: a module Worker that answers with the service's messages.
class FakeWorker {
  static made = [];
  constructor(url, options) { Object.assign(this, {url, options, sent: [], terminated: false}); FakeWorker.made.push(this); }
  postMessage(value) { this.sent.push(JSON.parse(JSON.stringify(value))); }
  terminate() { this.terminated = true; }
  emit(data) { this.onmessage?.({data}); }
}
const BROWSER_MODELS = Object.fromEntries(Object.entries(HEALTH.models).map(([k, v]) => [k, {...v, cached: false,
  download_bytes: k === 'single' ? 44703528 : 391900416, shared_download_bytes: 367881347}]));
const CONSENT_KEY = 'auslan-live-consent-v1';
// status: the /health HTTP status, or 'error' for a network failure; storage: initial localStorage entries, or 'throw'.
async function setup({voice = false, voices = voice ? [LOCAL_VOICE] : [], health = HEALTH, rvfc = false, worker = false, status = 200, storage = {}} = {}) {
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const timers = []; // recorded, never fired
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
  const store = new Map(storage === 'throw' ? [] : Object.entries(storage));
  const globals = {document, window, navigator: {mediaDevices: {getUserMedia: async () => stream}, clipboard: {writeText: async () => {}}},
    crypto, WebSocket: Socket, location: {protocol: 'http:', host: 'localhost'}, performance: {now: () => now},
    fetch: async () => { if (status === 'error') throw new TypeError('Failed to fetch'); return {ok: status === 200, status, json: async () => health}; },
    AbortSignal, SpeechSynthesisUtterance: class {constructor(text) {this.text = text;}},
    setTimeout: (fn, ms) => timers.push({fn, ms}), clearTimeout() {}, setInterval: () => 1, clearInterval() {}, console, Worker: worker ? FakeWorker : undefined};
  if (storage === 'throw') Object.defineProperty(globals, 'localStorage', {get() { throw new Error('SecurityError: storage is blocked'); }});
  else globals.localStorage = {getItem: k => store.has(k) ? store.get(k) : null, setItem: (k, v) => store.set(k, String(v)), removeItem: k => store.delete(k)};
  const context = vm.createContext(globals);
  FakeWorker.made = [];
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  const run = code => vm.runInContext(code, context);
  const sockets = run('state.sockets'); sockets.forEach(s => s.readyState = 1); sockets.forEach(s => s.onopen());
  return {elements, run, sockets, video, window, frameCallbacks, timers, store, stopped: () => stopped, spoken: () => spoken,
    setHealth(value) {status = value;},
    // One rVFC callback; Chrome-like metadata by default (mediaTime and presentationTime advance together).
    present(seconds, extra = {}) {const meta = {mediaTime: seconds, presentationTime: seconds * 1000, presentedFrames: ++presented, ...extra};
      const cbs = [...frameCallbacks.values()]; frameCallbacks.clear(); cbs.forEach(cb => cb(now, meta));},
    advance(ms=40, media=ms/1000) {now += ms; video.currentTime += media;},
    async start() {await run('enableCamera()'); run('startSigning()'); sockets[1].receive({type: 'recording_started', request_id: run('state.id')});},
    ack(n, extra = {}) {sockets[0].receive({type: 'pose', request_id: run('state.id'), buffer_len: n, person_count: 1, has_person: true, ...extra});},
    async finish(result) {await this.start(); this.ack(1); this.advance(); run('sendFrame()'); this.ack(2); run('finishSigning()');
      sockets[1].receive({type: 'result', request_id: run('state.id'), ...result});},
    // Browser engine: the worker, its ready sequence, and pose / translate replies on its two channels.
    get worker() { return FakeWorker.made.at(-1); },
    caps(webgpu = true) {this.worker.emit({type: 'capabilities', webgpu, isolated: true});},
    agree() {this.caps(); this.elements.consentButton.listeners.click();}, // capabilities, then the visitor consents
    engine(models = BROWSER_MODELS) {this.worker.emit({type: 'engine', models, max_frames: 300, runtime: {webgpu: true, isolated: true}});},
    ready(model = 'single') {if (!this.worker.sent.some(m => m.type === 'init')) this.agree(); this.engine();
      this.worker.emit({type: 'loaded', model, providers: {pose: 'webgpu', [`translator_${model}`]: 'wasm', threads: 4}, pose_probe_ms: 131.6, pose_probe: 'detector + 2 pose crops'});},
    reply(channel, data) {this.worker.emit({channel, request_id: run('state.id'), ...data});},
    frames() {return this.worker.sent.filter(m => m.type === 'frame');},
    async startBrowser() {await run('enableCamera()'); run('startSigning()'); this.reply(1, {type: 'recording_started'});},
    poseReply(n) {this.reply(0, {type: 'pose', buffer_len: n, person_count: 1, has_person: true, pose_ms: 140});}};
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

// ---- Browser transport: demo/static/engine/engine.js in a module Worker, used on a static host (/health 404).
const STATIC = {worker: true, status: 404};
const OFFLINE = {mode: 'showcase', translator: false, models: {}};
const tick = () => new Promise(resolve => setImmediate(resolve));
// refocus() looks for a hidden ancestor, as a browser does; the fake elements need to be told which one.
const inside = (el, parent) => { el.closest = () => parent.hidden ? parent : el.hidden ? el : null; };

test('a static host (/health 404) starts the in-browser engine with a capability check only', async () => {
  const app = await setup(STATIC);
  assert.equal(app.run('state.engine'), 'browser');
  assert.equal(app.worker.url, '/static/engine/engine.js');
  assert.equal(app.worker.options.type, 'module');
  assert.deepEqual(app.worker.sent, [{type: 'check'}]); // no init, so no network, before consent
  assert.equal(app.run('state.sockets.length'), 0);
});

test('a Python service without a translator (showcase, sample off, offline) shows its notice and never starts a worker', async () => {
  const app = await setup({worker: true, health: OFFLINE}); const e = app.elements;
  assert.equal(app.run('state.engine'), 'server');
  assert.equal(FakeWorker.made.length, 0);
  assert.equal(e.modeNotice.hidden, false);
  assert.equal(e.modeNotice.querySelector('strong').textContent, 'Translation service offline');
  assert.equal(e.enableCameraButton.disabled, true);
  assert.equal(e.connectionText.textContent, 'Translator offline');
  assert.equal(app.timers.at(-1).ms, 15000); // the health check is retried
  await app.timers.at(-1).fn();
  assert.equal(FakeWorker.made.length, 0);
});

test('a Python translator keeps the server transport even when workers exist', async () => {
  const app = await setup({worker: true});
  assert.equal(app.run('state.engine'), 'server');
  assert.equal(FakeWorker.made.length, 0);
  assert.equal(app.elements.runtimeDetail.textContent, 'Python service');
});

test('a failed health check after the service has answered keeps the server transport', async () => {
  const app = await setup({worker: true}); const e = app.elements;
  app.setHealth('error');
  app.sockets[0].onopen(); await tick(); // every socket open re-checks health
  assert.equal(app.run('state.engine'), 'server');
  app.sockets[0].onclose();              // the service restarts: the reconnect boot() re-checks health, which fails
  await app.timers.at(-1).fn();
  assert.equal(app.run('state.engine'), 'server');
  assert.equal(e.engineDetail.textContent, 'Live service not connected');
  app.setHealth(404);                    // not even a 404 switches a decided transport
  await app.timers.at(-1).fn();
  app.setHealth(503);
  await app.timers.at(-1).fn();
  assert.equal(app.run('state.engine'), 'server');
  assert.equal(FakeWorker.made.length, 0);
  assert.equal(app.run('state.worker'), null);
  app.setHealth(200);                    // the service is back: the server transport reconnects
  await app.timers.at(-1).fn();
  assert.equal(app.run('state.sockets.length'), 2);
});

test('a failed first health check (network, timeout, 5xx) never selects the browser engine', async () => {
  for (const status of ['error', 503]) {
    const app = await setup({worker: true, status}); const e = app.elements;
    assert.equal(app.run('state.engine'), null);
    assert.equal(FakeWorker.made.length, 0);
    assert.equal(e.engineDetail.textContent, 'Live service not connected');
    assert.equal(e.modeNotice.querySelector('strong').textContent, 'Research showcase');
    assert.equal(e.connectionText.textContent, 'Translator offline');
    assert.equal(app.timers.at(-1).ms, 15000);
    app.setHealth(404);                  // the first definite answer decides: a static host
    await app.timers.at(-1).fn();
    assert.equal(app.run('state.engine'), 'browser');
    assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  }
});

test('nothing is downloaded before consent; the consent click stores consent, sends init and moves focus', async () => {
  const app = await setup(STATIC); const e = app.elements;
  assert.match(html, /Live translation runs in this browser after a one-time download of about 400 MB from Hugging Face \(huggingface\.co\) and jsDelivr \(cdn\.jsdelivr\.net\), kept in this browser for later visits\. Camera frames stay on this device\./);
  assert.equal(e.consentCard.hidden, true); // no card before the capability check answers
  app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  assert.equal(e.consentCard.hidden, false);
  assert.equal(e.consentWarning.hidden, true); // WebGPU present
  assert.equal(e.enableCameraButton.disabled, true);
  assert.equal(e.connectionText.textContent, 'Not started');
  assert.equal(e.captureMessage.textContent, 'Live translation needs a one-time download first. Start it below when you are ready.');
  assert.equal(e.forgetButton.hidden, false); // files from an earlier visit can be removed without consenting again
  inside(e.consentButton, e.consentCard); e.consentButton.focus();
  e.consentButton.listeners.click();
  assert.deepEqual(app.worker.sent, [{type: 'check'}, {type: 'init'}]);
  assert.equal(app.store.get(CONSENT_KEY), '1');
  assert.equal(e.consentCard.hidden, true);
  assert.equal(app.run('document.activeElement'), e.captureMessage);
  app.engine();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'single'}); // the single model, never the ensemble
  assert.equal(app.worker.sent.some(m => m.model === 'ensemble'), false);
  assert.equal(e.enableCameraButton.disabled, false);
  assert.equal(e.engineStatusText.textContent, 'Downloading Single best model to this browser · 0 MB of 413 MB. Files are checked and kept for your next visit.');
  assert.equal(e.connectionText.textContent, 'Downloading');
  app.worker.emit({type: 'progress', model: 'single', loaded: 206e6, total: 412584875});
  assert.equal(e.engineProgress.value, 206e6); assert.equal(e.engineProgress.max, 412584875);
  await app.run('enableCamera()');
  assert.equal(e.recordButton.disabled, true); // camera on, model still downloading
  assert.equal(e.captureMessage.textContent, 'Camera ready. Waiting for the translator to finish loading.');
  app.worker.emit({type: 'loaded', model: 'single', providers: {pose: 'webgpu', translator_single: 'wasm', threads: 4}, pose_probe_ms: 131.6, pose_probe: 'detector + 2 pose crops'});
  assert.equal(e.recordButton.disabled, false);
  assert.equal(e.engineStatus.hidden, true);
  assert.equal(e.connectionText.textContent, 'Runs in this browser');
  assert.equal(e.runtimeDetail.textContent, 'In this browser · pose webgpu (pose probe ~132 ms/frame) · translator wasm · 4 WASM threads');
  assert.match(e.privacyNote.textContent, /not uploaded or saved/);
});

test('without WebGPU the consent card warns that pose may be too slow and that the files are then removed', async () => {
  const app = await setup(STATIC);
  app.caps(false);
  assert.equal(app.elements.consentCard.hidden, false);
  assert.equal(app.elements.consentWarning.hidden, false);
  assert.match(html, /id="consentWarning"[^>]*>This browser does not offer WebGPU, so pose estimation may be too slow here\. That is checked after the download, and if it is too slow the downloaded files are removed again\./);
});

test('a stored consent skips the card and starts init after the capability check', async () => {
  const app = await setup({...STATIC, storage: {[CONSENT_KEY]: '1'}});
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}, {type: 'init'}]);
  assert.equal(app.elements.consentCard.hidden, true);
});

test('blocked storage (every access throws) still asks, and the consent click still works for this page', async () => {
  const app = await setup({...STATIC, storage: 'throw'}); const e = app.elements;
  app.caps();
  assert.equal(e.consentCard.hidden, false);
  e.consentButton.listeners.click();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'init'});
  app.engine();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'single'});
});

test('a model already in this browser is loaded from it, never shown as downloading', async () => {
  const app = await setup({...STATIC, storage: {[CONSENT_KEY]: '1'}}); const e = app.elements;
  const CACHED = {...BROWSER_MODELS, single: {...BROWSER_MODELS.single, cached: true, download_bytes: 0, shared_download_bytes: 0}};
  app.caps(); app.engine(CACHED);
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'single'});
  assert.equal(e.engineStatusText.textContent, 'Loading Single best model from this browser. Files are checked before use.');
  assert.equal(e.connectionText.textContent, 'Loading');
  assert.equal(e.engineProgress.hidden, true);
  app.worker.emit({type: 'progress', model: 'single', loaded: 0, total: 0});
  assert.doesNotMatch(e.engineStatusText.textContent + e.connectionText.textContent, /Download/);
});

test('an engine failure at init is a final unavailable state with Retry, never "Preparing"', async () => {
  const app = await setup(STATIC); const e = app.elements;
  app.agree(); const first = app.worker;
  const reason = 'The model manifest could not be loaded (HTTP 404).';
  first.emit({type: 'engine_error', stage: 'init', error: reason});
  assert.equal(first.terminated, true);
  assert.equal(e.modeNotice.hidden, false);
  assert.equal(e.modeNotice.querySelector('strong').textContent, 'Live translation is unavailable in this browser');
  assert.equal(e.modeNotice.querySelector('p span').textContent, `${reason} The research overview is still available.`);
  assert.equal(e.retryButton.hidden, false);
  assert.equal(e.connectionBadge.className, 'connection disconnected');
  assert.equal(e.connectionText.textContent, 'Unavailable');
  assert.equal(e.enableCameraButton.disabled, true);
  assert.equal(e.recordButton.disabled, true);
  assert.equal(e.captureMessage.textContent, reason);
  assert.equal(e.consentCard.hidden, true);
  assert.equal(e.forgetButton.hidden, false); // consent was given, so stored files may exist: they can still be removed
  inside(e.retryButton, e.retryButton); e.retryButton.focus();
  e.retryButton.listeners.click();         // a new worker: check, then init again (consent was given)
  assert.notEqual(app.worker, first);
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  assert.equal(e.retryButton.hidden, true);
  assert.equal(e.modeNotice.hidden, true);
  assert.equal(app.run('document.activeElement'), e.captureMessage);
  app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}, {type: 'init'}]);
  app.engine();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'single'});
});

test('an engine error before the engine message is an init failure even without a stage', async () => {
  const app = await setup(STATIC);
  app.agree(); app.worker.emit({type: 'engine_error', error: 'Failed to fetch'});
  assert.equal(app.elements.connectionText.textContent, 'Unavailable');
  assert.equal(app.elements.retryButton.hidden, false);
});

test('an unsupported device is told so, is never offered a translation, and can retry', async () => {
  const app = await setup(STATIC); const e = app.elements;
  const reason = 'This browser offers neither WebGPU nor multi-threaded WebAssembly, so pose estimation would take over 2 s per camera frame. Translation is not offered on this device.';
  app.worker.emit({type: 'unsupported', reason});
  assert.equal(app.worker.terminated, true);
  assert.equal(e.enableCameraButton.disabled, true);
  assert.equal(e.recordButton.disabled, true);
  assert.equal(e.modeNotice.hidden, false);
  assert.equal(e.consentCard.hidden, true);
  assert.equal(e.connectionText.textContent, 'Unavailable');
  assert.equal(e.runtimeDetail.textContent, 'Unavailable in this browser');
  assert.equal(e.captureMessage.textContent, reason);
  assert.equal(e.retryButton.hidden, false);
  e.retryButton.listeners.click();
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
});

test('a slow device found after the download (pose probe) stops an open camera session from recording', async () => {
  const app = await setup(STATIC); const e = app.elements;
  app.agree(); app.engine();
  await app.run('enableCamera()');
  app.worker.emit({type: 'unsupported', model: 'single', reason: 'Pose estimation takes about 1900 ms per camera frame on this device (limit 800 ms), so translation is not offered here.'});
  assert.equal(e.recordButton.disabled, true);
  assert.equal(e.retryButton.hidden, false);
  app.run('startSigning()');
  assert.equal(app.run('state.phase'), 'ready');
  // the engine already deleted the files; consent is withdrawn so nothing downloads again without asking
  assert.equal(app.store.has(CONSENT_KEY), false);
  assert.equal(e.forgetButton.hidden, true);
  e.retryButton.listeners.click(); app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  assert.equal(e.consentCard.hidden, false);
});

test('after an init failure the stored files can still be removed, by a worker started only to delete them', async () => {
  const app = await setup(STATIC); const e = app.elements;
  app.agree(); const first = app.worker;
  first.emit({type: 'engine_error', stage: 'init', error: 'Failed to fetch'});
  assert.equal(first.terminated, true);
  assert.equal(e.forgetButton.hidden, false);
  e.forgetButton.listeners.click();
  const bare = app.worker;
  assert.notEqual(bare, first);
  assert.deepEqual(bare.sent, [{type: 'forget'}]); // no check, no init: nothing is fetched
  bare.emit({type: 'forgotten'});
  assert.equal(bare.terminated, true);
  assert.equal(app.store.has(CONSENT_KEY), false);
  assert.equal(e.modeNotice.hidden, true);          // out of the unavailable state: check, then the consent card
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
});

test('a new utterance waits until a cancelled translation replies (the worker cannot answer while it translates)', async () => {
  const app = await setup(STATIC); app.ready();
  await app.startBrowser();
  app.advance(); app.run('sendFrame()'); app.poseReply(1);
  app.advance(); app.run('sendFrame()'); app.poseReply(2);
  app.run('finishSigning()'); app.poseReply(3); // Finish captures one last frame, then translates
  const old = app.run('state.id');
  assert.deepEqual(app.worker.sent.at(-1), {type: 'translate', request_id: old});
  app.run('resetCurrent()'); // Cancel while the worker is translating
  const starts = () => app.worker.sent.filter(m => m.type === 'start_recording').length, n = starts();
  app.run('startSigning()');
  assert.equal(starts(), n);
  assert.equal(app.elements.captureMessage.textContent, 'Your previous utterance is still finishing. Please wait a moment.');
  app.worker.emit({channel: 1, type: 'result', request_id: old, text: '', accepted: false, reason: 'declined: low confidence'});
  assert.equal(app.run('state.result'), null); // the stale result is still ignored
  app.run('startSigning()');
  assert.equal(starts(), n + 1);
});

test('"Remove downloaded models" forgets every stored file, clears consent and returns to the consent card', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  assert.equal(app.store.get(CONSENT_KEY), '1');
  assert.equal(e.forgetButton.hidden, false);
  assert.equal(e.forgetButton.disabled, false);
  const first = app.worker;
  e.forgetButton.focus();
  e.forgetButton.listeners.click(); e.forgetButton.listeners.click(); // one request, however often it is clicked
  assert.deepEqual(first.sent.filter(m => m.type === 'forget'), [{type: 'forget'}]);
  assert.equal(first.terminated, false); // the worker removes the files first
  assert.equal(e.forgetButton['aria-disabled'], 'true'); assert.equal(e.forgetButton.disabled, false); // focus is kept
  assert.equal(app.run('document.activeElement'), e.forgetButton);
  first.emit({type: 'forgotten'});
  assert.equal(e.toast.textContent, 'The downloaded models were removed from this browser.');
  assert.equal(first.terminated, true);
  assert.equal(app.store.has(CONSENT_KEY), false);
  assert.notEqual(app.worker, first);
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}]); // consent is asked again
  assert.equal(e.consentCard.hidden, false);
  assert.equal(e.recordButton.disabled, true);
  assert.equal(e.enableCameraButton.disabled, true);
  assert.equal(e.engineDetail.textContent, 'None');
});

test('a failed removal is reported and can be tried again; the page keeps working', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  e.forgetButton.listeners.click();
  app.worker.emit({type: 'engine_error', stage: 'runtime', error: 'Cache Storage is unavailable.'});
  assert.equal(e.toast.textContent, 'The downloaded models could not be removed. Cache Storage is unavailable.');
  assert.equal(app.worker.terminated, false);
  assert.equal(e.connectionText.textContent, 'Runs in this browser');
  assert.equal(e.forgetButton['aria-disabled'], 'false');
  e.forgetButton.listeners.click();
  assert.equal(app.worker.sent.filter(m => m.type === 'forget').length, 2);
});

test('models cannot be removed while init is running (a later error then stays an init failure)', async () => {
  const app = await setup(STATIC); const e = app.elements;
  app.agree();
  assert.equal(e.forgetButton.disabled, true);
  e.forgetButton.listeners.click();
  assert.equal(app.worker.sent.some(m => m.type === 'forget'), false);
  app.worker.emit({type: 'engine_error', stage: 'init', error: 'Failed to fetch'});
  assert.equal(e.connectionText.textContent, 'Unavailable');
});

test('models cannot be removed while an utterance is in progress or a download is running', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.startBrowser();
  assert.equal(e.forgetButton.disabled, true);
  e.forgetButton.listeners.click();
  assert.equal(app.worker.sent.some(m => m.type === 'forget'), false);
  app.run('resetCurrent()');
  e.modelEnsemble.checked = true; e.modelEnsemble.listeners.change(); e.downloadButton.listeners.click();
  assert.equal(e.forgetButton.disabled, true);
});

test('browser transport: one frame in flight, never dropped; Finish waits for the last pose reply', async () => {
  const app = await setup(STATIC); app.ready();
  await app.startBrowser();
  const start = app.worker.sent.find(m => m.type === 'start_recording');
  assert.equal(start.model, 'single'); assert.equal(start.request_id, app.run('state.id'));
  app.advance(); app.run('sendFrame()'); app.advance(); app.run('sendFrame()');
  assert.equal(app.frames().length, 1); // the first frame is in the worker; two wait on the page
  assert.equal(app.run('state.queue.length'), 2);
  const f = app.frames()[0];
  assert.deepEqual(Object.keys(f).sort(), ['captured_at', 'data', 'request_id', 'type']);
  assert.equal(f.data, 'data:image/jpeg;base64,test'); // the same JPEG data URL the server receives
  app.run('finishSigning()');
  app.poseReply(1); assert.equal(app.frames().length, 2);
  app.poseReply(2); assert.equal(app.frames().length, 3);
  assert.equal(app.worker.sent.some(m => m.type === 'translate'), false);
  app.poseReply(3);
  assert.deepEqual(app.worker.sent.at(-1), {type: 'translate', request_id: app.run('state.id')});
  assert.deepEqual(app.frames().map(m => m.captured_at), [0, 40, 80]);
  assert.equal(app.elements.poseDetail.textContent, '140 / 140 ms');
  app.reply(1, {type: 'result', text: 'hello', accepted: true, reason: 'ready', confidence: 0.5, threshold: 0.706661, coverage_target: 0.5,
    model: 'single', model_label: HEALTH.models.single.label, capture_fps: 25, frames: 3, frames_used: 3, translate_ms: 88.2, rgb_ms: 0});
  assert.equal(app.elements.resultText.textContent, 'hello');
  assert.equal(app.run('state.history.length'), 1);
  assert.equal(app.elements.translateDetail.textContent, '88 ms');
});

test('after Finish, each pose reply re-arms the 120 s timeout, so a long in-browser backlog is not cut off', async () => {
  const app = await setup(STATIC); app.ready();
  await app.startBrowser();
  app.advance(); app.run('sendFrame()'); app.advance(); app.run('sendFrame()');
  app.run('finishSigning()');
  const armed = () => app.timers.filter(t => t.ms === 120000).length;
  assert.equal(armed(), 1);
  app.poseReply(1); assert.equal(armed(), 2);
  app.poseReply(2); assert.equal(armed(), 3);
  app.timers.filter(t => t.ms === 120000).at(-1).fn();
  assert.equal(app.elements.captureMessage.textContent, 'Translation is taking too long. Please try a shorter utterance.');
});

test('browser transport: worker errors fail the utterance and replies for a superseded request are ignored', async () => {
  const app = await setup(STATIC); app.ready();
  await app.startBrowser(); const oldId = app.run('state.id');
  app.run('resetCurrent()');
  assert.deepEqual(app.worker.sent.at(-1), {type: 'stop_recording', request_id: oldId});
  app.advance(); app.run('startSigning()'); app.reply(1, {type: 'recording_started'});
  const newId = app.run('state.id');
  app.advance(); app.run('sendFrame()');                     // queued behind the in-flight first frame
  for (const type of ['frame_ignored', 'pose']) app.worker.emit({channel: 0, type, request_id: oldId, buffer_len: 9, has_person: true});
  app.worker.emit({channel: 1, type: 'result', request_id: oldId, text: 'Old words', accepted: true});
  assert.equal(app.run('state.pendingFrame'), true);
  assert.equal(app.frames().filter(m => m.request_id === newId).length, 1);
  assert.equal(app.run('state.frames'), 0);
  assert.equal(app.run('state.result'), null);
  app.reply(0, {error: 'Keep one signer in view for the whole utterance, then try again.'});
  assert.equal(app.run('state.phase'), 'ready');
  assert.equal(app.elements.captureMessage.textContent, 'Keep one signer in view for the whole utterance, then try again.');
  assert.equal(app.run('state.queue.length'), 0);
});

test('a runtime pose failure ends the utterance with the engine message and keeps the page usable', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.startBrowser();
  const error = 'Pose estimation stopped working in this browser. Reload the page to try again.';
  app.worker.emit({type: 'engine_error', stage: 'runtime', error});
  app.reply(0, {error});
  assert.equal(app.run('state.phase'), 'ready');
  assert.equal(e.captureMessage.textContent, error);
  assert.equal(e.retryButton.hidden, true);
  assert.equal(e.engineStatus.hidden, true);
});

test('the ensemble downloads only after the visitor asks, shows its size first, and focus follows the status', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.run('enableCamera()');
  e.modelEnsemble.checked = true; e.modelEnsemble.listeners.change();
  assert.equal(e.recordButton.disabled, true);
  assert.equal(e.downloadButton.hidden, false);
  assert.equal(e.downloadLabel.textContent, 'Download Six-model ensemble · 392 MB'); // pose and runtimes already loaded with the single model
  assert.match(e.engineStatusText.textContent, /one-time download of 392 MB/);
  assert.equal(app.worker.sent.some(m => m.type === 'load' && m.model === 'ensemble'), false);
  inside(e.downloadButton, e.downloadButton); e.downloadButton.focus();
  e.downloadButton.listeners.click();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'ensemble'});
  assert.equal(e.downloadButton.hidden, true);
  assert.equal(app.run('document.activeElement'), e.captureMessage);
  app.worker.emit({type: 'loaded', model: 'ensemble', providers: {pose: 'webgpu', translator_single: 'wasm', translator_ensemble: 'wasm', dinov2: 'wasm', threads: 4}});
  assert.equal(e.recordButton.disabled, false);
  app.run('startSigning()');
  assert.equal(app.worker.sent.find(m => m.type === 'start_recording').model, 'ensemble');
  e.modelSingle.checked = true; e.modelSingle.listeners.change(); // both stay loaded: switching back needs no download
  assert.equal(e.downloadButton.hidden, true);
});

test('a failed download is reported and can be retried', async () => {
  const app = await setup(STATIC); const e = app.elements;
  app.agree(); app.engine();
  app.worker.emit({type: 'engine_error', stage: 'load', model: 'single', error: 'Checksum mismatch for /models/pose/yolox_m.onnx; the file was not used.'});
  assert.equal(e.downloadButton.hidden, false);
  assert.equal(e.downloadLabel.textContent, 'Download Single best model · 413 MB');
  assert.match(e.engineStatusText.textContent, /Checksum mismatch .* You can try the download again\./);
  assert.match(e.captureMessage.textContent, /Checksum mismatch/);
  assert.equal(e.retryButton.hidden, true); // a download failure is not the final state
  e.downloadButton.listeners.click();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'load', model: 'single'});
});

test('a failed download of another model, or during recording, never overwrites the capture message', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.run('enableCamera()');
  e.modelEnsemble.checked = true; e.modelEnsemble.listeners.change(); e.downloadButton.listeners.click();
  e.modelSingle.checked = true; e.modelSingle.listeners.change();
  app.run('resetCurrent()');
  const ready = 'Ready. Sign one short utterance, then select Finish.';
  assert.equal(e.captureMessage.textContent, ready);
  const error = 'Checksum mismatch for /models/dinov2/model.onnx; the file was not used.';
  app.worker.emit({type: 'engine_error', stage: 'load', model: 'ensemble', error});
  assert.equal(e.captureMessage.textContent, ready); // another model's failure: a toast only
  assert.equal(e.toast.textContent, error);
  e.modelEnsemble.checked = true; e.modelEnsemble.listeners.change();
  assert.match(e.engineStatusText.textContent, /dinov2.* You can try the download again\./); // shown with that model
  e.downloadButton.listeners.click();
  e.modelSingle.checked = true; e.modelSingle.listeners.change();
  app.run('startSigning()'); app.reply(1, {type: 'recording_started'});
  const recording = 'Recording. Select Finish & translate when your utterance is complete.';
  assert.equal(e.captureMessage.textContent, recording);
  app.run('state.model = "ensemble"'); // even the selected model's failure waits while recording
  app.worker.emit({type: 'engine_error', stage: 'load', model: 'ensemble', error});
  assert.equal(e.captureMessage.textContent, recording);
  assert.equal(app.run('state.phase'), 'recording');
});

test('leaving the page terminates the worker; returning from the back-forward cache starts a new one', async () => {
  const app = await setup(STATIC); app.ready();
  const first = app.worker;
  app.window.listeners.pagehide();
  assert.equal(first.terminated, true);
  app.window.listeners.pageshow({persisted: true});
  await tick();
  assert.notEqual(app.worker, first);
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  assert.equal(app.elements.downloadButton.hidden, true); // no stale offer: models are known again only after init
  app.run('loadModel("single")');
  assert.deepEqual(app.worker.sent, [{type: 'check'}]);
  app.caps();
  assert.deepEqual(app.worker.sent, [{type: 'check'}, {type: 'init'}]); // consent was given on this page
});

// ---- people: who was translated, the switch, session labels (the in-browser engine only)
const RESULT = {type: 'result', text: 'hello', accepted: true, reason: 'ready', confidence: 0.5, threshold: 0.706661, coverage_target: 0.5,
  model: 'single', model_label: HEALTH.models.single.label, capture_fps: 25, frames: 3, frames_used: 3, translate_ms: 88.2, rgb_ms: 0};
const TWO = {mode: 'activity', chosen: 'Person A', count: 2, session_people: 2, people: [
  {label: 'Person A', frames: 50, share: 1, eligible: true, score: 1.2, x: 0.72}, {label: 'Person B', frames: 50, share: 1, eligible: true, score: -0.4, x: 0.28}]};
async function finishBrowser(app, persons, extra = {}) {
  await app.startBrowser(); app.poseReply(1); app.advance(); app.run('sendFrame()'); app.poseReply(2); app.run('finishSigning()');
  app.reply(1, {...RESULT, ...extra, ...(persons && {persons})});
}

test('two people in view: the result says who was translated (mirrored preview), offers a switch, and the conversation is labelled', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, TWO);
  assert.equal(e.personBox.hidden, false);
  assert.match(e.personText.textContent, /^Translated Person A on the left of your camera view, the person moving most like a signer\. That is a guess, not a recognition\. 2 people were in view\.$/);
  assert.deepEqual(e.personSwitch.children.map(b => b.textContent), ['Translate Person B instead']);
  assert.equal(e.personsDetail.textContent, '2 in view · translated Person A');
  assert.equal(app.run('state.history[0].label'), 'Person A');
  e.personSwitch.children[0].listeners.click();                                  // the switch
  assert.deepEqual(app.worker.sent.at(-1), {type: 'retranslate', request_id: app.run('state.id'), person: 'Person B'});
  assert.equal(app.run('state.phase'), 'processing'); assert.equal(e.personBox.hidden, true);
  app.reply(1, {...RESULT, text: 'goodbye', persons: {...TWO, mode: 'manual', chosen: 'Person B'}});
  assert.match(e.personText.textContent, /^Translated Person B on the right of your camera view, as you chose\./);
  assert.equal(app.run('state.history.length'), 1, 'a switch replaces that utterance’s entry');
  assert.equal(app.run('state.history[0].text'), 'goodbye'); assert.equal(app.run('state.history[0].label'), 'Person B');
  assert.deepEqual(e.personSwitch.children.map(b => b.textContent), ['Translate Person A instead']);
  e.copySessionButton.listeners.click();
});

test('one person in a session of one: no person box and no label; a later single-person utterance in a two-person session is labelled', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, {mode: 'single', chosen: 'Person A', count: 1, session_people: 1, people: [{label: 'Person A', frames: 30, share: null, eligible: true, score: null, x: 0.5}]});
  assert.equal(e.personBox.hidden, true); assert.equal(app.run('state.history[0].label'), null);
  assert.equal(e.personsDetail.textContent, '1 in view · translated Person A');
  app.run('resetCurrent()');
  await finishBrowser(app, {mode: 'single', chosen: 'Person B', count: 1, session_people: 2, people: [{label: 'Person B', frames: 30, share: null, eligible: true, score: null, x: 0.3}]});
  assert.equal(e.personBox.hidden, false);
  assert.equal(e.personText.textContent, 'Person B on the right of your camera view. 1 person was in view.');
  assert.equal(e.personSwitch.children.length, 0, 'nobody else to switch to');
  assert.equal(app.run('state.history[0].label'), 'Person B');
});

test('no person is picked when nobody was in view for most of the utterance; a declined result still offers the other person', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, {mode: 'default', chosen: null, count: 2, session_people: 2, people: TWO.people.map(p => ({...p, eligible: false, share: 0.5}))});
  assert.match(e.personText.textContent, /^2 people were in view\. Nobody stayed in view long enough to compare their movement, so the translator followed the person it was already following\.$/);
  assert.equal(e.personSwitch.children.length, 0);
  app.run('resetCurrent()');
  await finishBrowser(app, TWO, {text: '', accepted: false, reason: 'declined: low confidence'});
  assert.equal(e.resultContent.hidden, true); assert.equal(e.personBox.hidden, false);
  assert.deepEqual(e.personSwitch.children.map(b => b.textContent), ['Translate Person B instead']);
  assert.equal(app.run('state.history.length'), 1, 'a declined result adds nothing to the conversation');
});

test('Clear conversation and turning the camera off tell the worker to forget who was in view; the Python service is not sent the message', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, TWO);
  e.clearSessionButton.listeners.click();
  assert.deepEqual(app.worker.sent.at(-1), {type: 'clear_session'});
  assert.equal(app.run('state.history.length'), 0);
  app.worker.sent.length = 0;
  e.cameraOffButton.listeners.click();
  assert.equal(app.worker.sent.some(m => m.type === 'clear_session'), true);
  const server = await setup(); await server.finish({text: 'hello', accepted: true, reason: 'ready', confidence: 0.5});
  assert.equal(server.elements.personBox.hidden, true);
  server.elements.clearSessionButton.listeners.click();
  assert.equal(server.sockets.flatMap(s => s.sent).some(m => m.type === 'clear_session'), false);
});

test('in the browser the multi-person warning explains the choice; the Python service keeps its one-signer warning', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.startBrowser();
  app.reply(0, {type: 'pose', buffer_len: 1, person_count: 2, has_person: true, pose_ms: 140});
  assert.match(e.captureWarning.textContent, /picks the person moving most like a signer and says who/);
  const server = await setup(); await server.start(); server.ack(1, {person_count: 2});
  assert.match(server.elements.captureWarning.textContent, /follows one signer/);
  assert.match(html, /id="peopleNote" hidden/);                    // hidden in the markup, shown only when the in-browser engine starts
  assert.equal(e.peopleNote.hidden, false);
  assert.match(e.privacyNote.textContent, /kept until your next utterance, Clear or turning the camera off/);
});

test('the person note: declined switch wording, nothing counted, the mirror boundaries, and the note sits below the result, not beside it', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, {...TWO, mode: 'manual', chosen: 'Person B'}, {text: '', accepted: false, reason: 'declined: low confidence'});
  assert.match(e.personText.textContent, /^Person B on the right of your camera view was picked \(as you chose\), but the model was not confident enough to translate it\. 2 people were in view\.$/);
  assert.doesNotMatch(e.personText.textContent, /^Translated/);
  app.run('resetCurrent()');
  await finishBrowser(app, {mode: 'single', chosen: null, count: 0, session_people: 3, people: []});
  assert.equal(e.personBox.hidden, true, 'nobody counted: nothing to say (no "null" sentence)');
  const said = (x) => { app.run('resetCurrent()'); return finishBrowser(app, {mode: 'single', chosen: 'Person A', count: 1, session_people: 2,
    people: [{label: 'Person A', frames: 30, share: null, eligible: true, score: null, x}]}).then(() => e.personText.textContent); };
  assert.match(await said(0.61), /on the left of your camera view/);     // raw right of centre = left in the mirrored preview
  assert.match(await said(0.6), /in the middle of your camera view/);
  assert.match(await said(0.5), /in the middle/);
  assert.match(await said(0.4), /in the middle/);
  assert.match(await said(0.39), /on the right of your camera view/);
  assert.match(await said(null), /^Person A\. 1 person was in view\.$/);
  assert.match(html, /<\/div>\s*<div id="personBox" class="person-box" role="status" aria-live="polite" hidden>/, 'a sibling of the result area, with its own live region');
});

test('Clear conversation hides a note whose switch could no longer work; a switch removes that utterance from the conversation at once', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, TWO);
  assert.equal(e.personBox.hidden, false); assert.equal(e.personSwitch.children.length, 1);
  e.clearSessionButton.listeners.click();
  assert.equal(e.personBox.hidden, true); assert.equal(e.personSwitch.children.length, 0);
  app.run('resetCurrent()');
  await finishBrowser(app, TWO);
  assert.equal(app.run('state.history.length'), 1);
  e.personSwitch.children[0].listeners.click();
  assert.equal(app.run('state.history.length'), 0, 'the earlier entry no longer matches what the visitor is shown');
  app.reply(1, {...RESULT, text: '', accepted: false, reason: 'declined: low confidence', persons: {...TWO, mode: 'manual', chosen: 'Person B'}});
  assert.equal(app.run('state.history.length'), 0, 'a declined switch adds nothing');
});

test('a translation the worker answers with "cancelled" releases the page, so the next utterance can start (was stuck for good)', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await finishBrowser(app, TWO);
  e.personSwitch.children[0].listeners.click();                       // the switch is running
  const id = app.run('state.id');
  assert.equal(app.run('state.inflight'), id);
  e.clearButton.listeners.click();                                     // Cancel; the worker finishes silently, then says so
  assert.equal(app.run('state.inflight'), id, 'still waiting for the worker');
  app.worker.emit({channel: 1, type: 'cancelled', request_id: id});
  assert.equal(app.run('state.inflight'), null);
  app.run('startSigning()');
  assert.notEqual(e.captureMessage.textContent, 'Your previous utterance is still finishing. Please wait a moment.');
  assert.equal(app.worker.sent.at(-1).type, 'start_recording');
});

test('after Clear conversation a late result still names the people but offers no switch; the next utterance offers it again', async () => {
  const app = await setup(STATIC); const e = app.elements; app.ready();
  await app.startBrowser(); app.poseReply(1); app.advance(); app.run('sendFrame()'); app.poseReply(2); app.run('finishSigning()');
  e.clearSessionButton.listeners.click();                             // Clear conversation while the translation is still running
  assert.equal(app.run('state.peopleCleared'), true);
  app.reply(1, {...RESULT, persons: TWO});
  assert.equal(e.personBox.hidden, false); assert.match(e.personText.textContent, /^Translated Person A/);
  assert.equal(e.personSwitch.children.length, 0, 'the worker has forgotten Person B, so a switch could not work');
  app.run('resetCurrent()'); app.run('startSigning()');
  assert.equal(app.run('state.peopleCleared'), false);
});

test('the live text says the roster is kept even with one person, and that the switch depends on who else was in view', async () => {
  const privacy = fs.readFileSync(path.join(__dirname, '../static/privacy.html'), 'utf8');
  assert.match(privacy, /This happens even when only one person is in view: each person’s label and last shoulder position are kept/);
  assert.match(privacy, /with one person in view, nothing from the utterance is kept after Finish/);
  assert.match(html, /when someone else was in view for most of the utterance, offers a switch/);
  const app = await setup(STATIC); app.ready();
  assert.match(app.elements.privacyNote.textContent, /kept even with one person in view/);
});
