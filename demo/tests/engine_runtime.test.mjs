// engine.js browserRuntime with a scratch ONNX Runtime (sessions are stand-ins that advance a fake clock), the real
// loader.js over a fake site + Cache Storage, real cv.js and pose.js pre-processing. Covers protocol v3's check /
// init / load / forget: no network before consent, the manifest pin, cached flags and uncached-only byte counts,
// progress that counts downloaded bytes only, the probe (detector + 2 RTMW crops on a 960x720 frame), a refused
// device keeping nothing, and DINOv2 created before the ensemble is registered. Parity lives in the other tests.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { MODEL_BASE, RUNTIME } from '../static/engine/config.js';
import { browserRuntime, createEngine } from '../static/engine/engine.js';
import { LIVE } from '../static/engine/features.js';
import { MEAN, STD } from '../static/engine/pose.js';
import { createLoader, modelFiles } from '../static/engine/loader.js';

const sha = (b) => createHash('sha256').update(b).digest('hex');
const enc = new TextEncoder();
const tick = () => new Promise((r) => setImmediate(r));

// ---- a fake site: the real runtime files from node_modules (their pinned hashes must hold), fake models elsewhere.
const require = createRequire(import.meta.url);
const ortDist = (f) => require.resolve('onnxruntime-web').replace(/dist\/.*$/, `dist/${f}`);
const RUNTIME_FILES = { ort: ortDist('ort.webgpu.bundle.min.mjs'), ortWasm: ortDist('ort-wasm-simd-threaded.asyncify.wasm') };

const B = { det: enc.encode('DET'), pose: enc.encode('POSE'), dino: enc.encode('DINO'),
  vocab: enc.encode(JSON.stringify(['<unk>', '<pad>', '<bos>', '<eos>', 'hello', '.'])) };
const member = (name) => {
  const e = enc.encode(`ENC ${name}`), d = enc.encode(`DEC ${name}`), dir = `translators/${name}/`;
  B[`${dir}encoder.onnx`] = e; B[`${dir}decoder.onnx`] = d; B[`${dir}vocab.json`] = B.vocab;
  return { name, fusion: name !== 'w4', encoder: `${dir}encoder.onnx`, decoder: `${dir}decoder.onnx`, vocab: `${dir}vocab.json`,
    sha256: { encoder: sha(e), decoder: sha(d), vocab: sha(B.vocab) }, bytes: { encoder: e.length, decoder: d.length, vocab: B.vocab.length } };
};
const REV = 'ed25f3a31f01632728cabb09d1542f84ab7b0056';
function makeManifest(dinoRevision = REV) {
  return { version: 1, preprocess: LIVE, decode: { max_len: 5, pad: 1, bos: 2, eos: 3 },
    pose: { detector: { file: 'pose/det.onnx', sha256: sha(B.det), bytes: B.det.length }, pose: { file: 'pose/pose.onnx', sha256: sha(B.pose), bytes: B.pose.length } },
    dinov2: { hf_model: 'onnx-community/dinov2-small', revision: dinoRevision, path: 'onnx/model.onnx', sha256: sha(B.dino), bytes: B.dino.length },
    models: {
      single: { label: 'S', beam: 10, lp: 0.7, rgb: false, tau: { 0.5: 0.7 }, evidence: {}, members: [member('w4')] },
      ensemble: { label: 'E', beam: 10, lp: 0.3, rgb: true, tau: { 0.5: 0.9 }, evidence: {}, members: [member('f1'), member('f2')] } } };
}

function site(manifestBytes) {
  const files = { [`${MODEL_BASE}manifest.json`]: manifestBytes };
  const m = JSON.parse(new TextDecoder().decode(manifestBytes));
  for (const name of Object.keys(m.models)) {
    for (const f of modelFiles(m, name)) {
      files[f.url] = f.key in RUNTIME_FILES ? new Uint8Array(readFileSync(RUNTIME_FILES[f.key]))
        : f.key.startsWith('pose.') ? B[f.key.slice(5) === 'detector' ? 'det' : 'pose'] : f.key === 'dinov2' ? B.dino : B[f.key];
    }
  }
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push(url);
    const body = files[url];
    if (!body) return new Response('missing', { status: 404 });
    return new Response(body.slice(), { headers: { 'content-length': String(body.length) } });
  };
  return { fetch, calls };
}

function cacheStorage() {
  const entries = new Map();
  const cache = {
    async match(key) { return entries.has(key) ? new Response(entries.get(key).slice()) : undefined; },
    async put(key, res) { entries.set(key, new Uint8Array(await res.arrayBuffer())); },
    async delete(key) { return entries.delete(key.url ?? key); },
    async keys() { return [...entries.keys()].map((url) => ({ url })); },
  };
  return { entries, caches: { open: async () => cache } };
}

// ---- a scratch ONNX Runtime: the engine imports it from the "verified ort bytes" through URL.createObjectURL.
const fake = { clock: 0, detMs: 37.25, estMs: 21.5, runs: { det: 0, est: 0 }, creates: {}, releases: 0, failDino: 0, detInputs: [], estInputs: [] };
globalThis.__fakeOrtCreate = async (bytes) => {
  const kind = new TextDecoder().decode(bytes).split(' ')[0];
  fake.creates[kind] = (fake.creates[kind] ?? 0) + 1;
  if (kind === 'DINO' && fake.failDino-- > 0) throw new Error('WebAssembly.Memory(): could not allocate memory');
  const run = kind === 'DET' ? async (feeds) => {
    fake.runs.det++; fake.clock += fake.detMs; fake.detInputs.push(feeds.x);
    return { dets: { data: new Float32Array(5), dims: [1, 1, 5] } };
  } : async (feeds) => {
    fake.runs.est++; fake.clock += fake.estMs; fake.estInputs.push(feeds.x);
    return { sx: { data: new Float32Array(133 * 384), dims: [1, 133, 384] }, sy: { data: new Float32Array(133 * 512), dims: [1, 133, 512] } };
  };
  return { inputNames: ['x'], outputNames: kind === 'DET' ? ['dets'] : ['sx', 'sy'], run, release: async () => { fake.releases++; } };
};
const FAKE_ORT = 'data:text/javascript,' + encodeURIComponent(
  'export const env = { wasm: { numThreads: 4 } };' +
  'export class Tensor { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } }' +
  'export const InferenceSession = { create: (...a) => globalThis.__fakeOrtCreate(...a) };');
URL.createObjectURL = () => FAKE_ORT;
Object.assign(globalThis, { OffscreenCanvas: class {}, createImageBitmap: () => {}, crossOriginIsolated: true });

function setup({ manifest = makeManifest(), store = cacheStorage(), pin } = {}) {
  const bytes = enc.encode(JSON.stringify(manifest)), s = site(bytes);
  globalThis.fetch = s.fetch;
  const rt = browserRuntime({ loader: createLoader({ fetch: s.fetch, caches: store.caches }), now: () => fake.clock,
    manifestSha256: pin ?? sha(bytes) });
  const out = [];
  const handle = createEngine((m) => out.push(m), rt, () => fake.clock);
  return { rt, out, handle, store, calls: s.calls, manifest };
}
const quiet = async (fn) => { const e = console.error; console.error = () => {}; try { return await fn(); } finally { console.error = e; } };
// Bytes a model still downloads: one file per sha256, minus what is cached.
const bytesOf = (m, name, pick, cached = new Set()) => [...new Map(modelFiles(m, name).filter(pick).map((f) => [f.sha256, f])).values()]
  .filter((f) => !cached.has(f.sha256)).reduce((a, f) => a + f.bytes, 0);
const shared = (f) => /^(ort|ortWasm|pose\.)/.test(f.key);
const has = (f) => shared(f);                              // in memory after the single model: runtimes and pose

test('check: capabilities with no network at all; neither WebGPU nor isolation is unsupported', async () => {
  const h = setup();
  await h.handle({ type: 'check' });
  assert.deepEqual(h.out, [{ type: 'capabilities', webgpu: false, isolated: true }]);   // Node: no navigator.gpu
  assert.equal(h.calls.length, 0);
  globalThis.crossOriginIsolated = false;
  try {
    const g = setup();
    await g.handle({ type: 'check' });
    assert.equal(g.out[0].type, 'unsupported');
    assert.match(g.out[0].reason, /neither WebGPU nor multi-threaded WebAssembly/);
    assert.equal(g.calls.length, 0);
  } finally { globalThis.crossOriginIsolated = true; }
});

test('init: the manifest must match its pinned sha256 and pin DINOv2 to a commit; nothing else is fetched first', async () => {
  const bad = setup({ pin: '0'.repeat(64) });
  await quiet(() => bad.handle({ type: 'init' }));
  assert.deepEqual(bad.out, [{ type: 'engine_error', stage: 'init',
    error: 'The model manifest is not the version this page was built for (SHA-256 mismatch), so nothing was loaded.' }]);
  assert.deepEqual(bad.calls, [`${MODEL_BASE}manifest.json`]);
  for (const rev of ['main', REV.slice(1), REV.toUpperCase(), null]) {
    const h = setup({ manifest: makeManifest(rev) });
    await quiet(() => h.handle({ type: 'init' }));
    assert.deepEqual(h.out, [{ type: 'engine_error', stage: 'init',
      error: 'The model manifest does not pin DINOv2 to a commit (dinov2.revision), so nothing was loaded.' }], String(rev));
  }
});

test('init / load: cached flags, uncached-only sizes, progress in downloaded bytes, the probe label; a second visit downloads nothing', async () => {
  const store = cacheStorage(), first = setup({ store }), m = first.manifest;
  await first.handle({ type: 'init' });
  const e = first.out.shift();
  assert.equal(e.type, 'engine');
  assert.deepEqual(e.runtime, { webgpu: false, isolated: true });
  const all = (name) => bytesOf(m, name, () => true);
  assert.deepEqual([e.models.single.cached, e.models.single.download_bytes, e.models.single.shared_download_bytes],
    [false, bytesOf(m, 'single', (f) => !shared(f)), bytesOf(m, 'single', shared)]);
  Object.assign(fake, { runs: { det: 0, est: 0 }, detInputs: [], estInputs: [] });
  await first.handle({ type: 'load', model: 'single' });
  const progress = first.out.filter((x) => x.type === 'progress'), loaded = first.out.at(-1);
  assert.deepEqual(progress.at(-1), { type: 'progress', model: 'single', loaded: all('single'), total: all('single') });
  assert.ok(progress.every((x) => x.total === all('single')));
  assert.deepEqual(loaded, { type: 'loaded', model: 'single', providers: { pose: 'wasm', translator_single: 'wasm', threads: 4 },
    pose_probe_ms: 80.2, pose_probe: 'detector + 2 pose crops' });   // 37.25 + 2 x 21.5 = 80.25 -> 80.2, as Python rounds
  // The probe: warm-up + timed pass, each = detector once + RTMW twice; the frame is 960x720 (letterboxed to 640x480).
  assert.deepEqual(fake.runs, { det: 2, est: 4 });
  const x = fake.detInputs[1].data, plane = 640 * 640;
  assert.deepEqual([x[479 * 640], x[480 * 640], x[plane - 1]], [128, 114, 114]);   // rows below 480 are padding
  const inside = (c) => Math.fround((128 - MEAN[c]) / STD[c]), crop = 192 * 256;   // no warpAffine border in either crop
  for (const t of fake.estInputs) assert.ok(t.data.every((v, i) => v === inside(Math.floor(i / crop))));
  assert.equal(first.store.entries.size, modelFiles(m, 'single').length);

  const again = setup({ store });                          // a later visit (new worker, same Cache Storage)
  await again.handle({ type: 'init' });
  const models = again.out.shift().models;
  assert.deepEqual([models.single.cached, models.single.download_bytes, models.single.shared_download_bytes], [true, 0, 0]);
  const inSingle = new Set(modelFiles(m, 'single').map((f) => f.sha256));   // incl. the vocabulary every member shares
  assert.deepEqual([models.ensemble.cached, models.ensemble.download_bytes, models.ensemble.shared_download_bytes],
    [false, bytesOf(m, 'ensemble', (f) => !shared(f), inSingle), 0]);
  assert.equal(models.ensemble.download_bytes, 4 * 6 + B.dino.length);   // 'ENC f1' 'DEC f1' 'ENC f2' 'DEC f2' + DINOv2
  const before = again.calls.length;
  await again.handle({ type: 'load', model: 'single' });
  assert.deepEqual(again.out.filter((x) => x.type === 'progress'), [{ type: 'progress', model: 'single', loaded: 0, total: 0 }]);
  assert.equal(again.out.at(-1).type, 'loaded');
  assert.equal(again.calls.length, before);                // loaded from this browser: no network
});

test('probe over the limit: sessions released, every cached file deleted, unsupported; later loads refused without a download', async () => {
  const store = cacheStorage(), h = setup({ store });
  store.entries.set('https://model-cache.invalid/sha256/' + 'a'.repeat(64), new Uint8Array(3));   // an older visit's file
  Object.assign(fake, { estMs: 400, detMs: 10, releases: 0 });
  try {
    await h.handle({ type: 'init' });
    await h.handle({ type: 'load', model: 'single' });
    const reason = 'Pose estimation takes about 810 ms per camera frame on this device (limit 800 ms), so translation is not offered here. The downloaded files were removed from this browser.';
    assert.deepEqual(h.out.at(-1), { type: 'unsupported', model: 'single', reason });
    assert.equal(store.entries.size, 0);
    assert.equal(fake.releases, 2);                        // detector and pose sessions
    const calls = h.calls.length;
    await h.handle({ type: 'load', model: 'ensemble' });
    assert.deepEqual(h.out.at(-1), { type: 'unsupported', model: 'ensemble', reason });
    assert.equal(h.calls.length, calls);
    assert.equal(store.entries.size, 0);
    assert.equal(h.rt.ready('single'), false);
  } finally { Object.assign(fake, { estMs: 21.5, detMs: 37.25 }); }
});

test('ensemble: DINOv2 is created before the translator is registered; a failed DINOv2 leaves it not ready and the retry completes it', async () => {
  const h = setup();
  await h.handle({ type: 'init' });
  Object.assign(fake, { failDino: 1, creates: {} });
  await quiet(() => h.handle({ type: 'load', model: 'ensemble' }));
  assert.deepEqual(h.out.at(-1), { type: 'engine_error', stage: 'load', model: 'ensemble', error: 'WebAssembly.Memory(): could not allocate memory' });
  assert.equal(h.rt.ready('ensemble'), false);
  assert.equal(fake.creates.ENC, undefined);               // no translator session was created before DINOv2 failed
  h.handle({ type: 'start_recording', request_id: 'r', model: 'ensemble' });
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'That translation model is not available.', request_id: 'r' });
  await h.handle({ type: 'load', model: 'ensemble' });
  assert.equal(h.out.at(-1).type, 'loaded');
  assert.deepEqual(h.out.at(-1).providers, { pose: 'wasm', dinov2: 'wasm', translator_ensemble: 'wasm', threads: 4 });
  assert.deepEqual([fake.creates.DINO, fake.creates.ENC, fake.creates.DET], [2, 2, 1]);
  assert.equal(h.rt.ready('ensemble'), true);
  const creates = JSON.stringify(fake.creates);
  await h.handle({ type: 'load', model: 'ensemble' });     // loaded: the early return creates nothing
  assert.equal(h.out.at(-1).type, 'loaded');
  assert.equal(JSON.stringify(fake.creates), creates);
});

test('forget: every cached model and runtime file is deleted and nothing is cached afterwards', async () => {
  const store = cacheStorage(), h = setup({ store });
  await h.handle({ type: 'init' });
  await h.handle({ type: 'load', model: 'single' });
  assert.ok(store.entries.size > 0);
  await h.handle({ type: 'forget' });
  assert.deepEqual(h.out.at(-1), { type: 'forgotten' });
  assert.equal(store.entries.size, 0);
  const before = h.calls.length;
  await h.handle({ type: 'load', model: 'ensemble' });     // still usable until the page terminates the worker ...
  assert.equal(h.out.at(-1).type, 'loaded');
  assert.equal(store.entries.size, 0);                     // ... but nothing is written back
  const got = h.calls.slice(before);                       // nothing cached now: the members' shared vocabulary once
  assert.deepEqual(got.filter((u) => u.endsWith('vocab.json')), [`${MODEL_BASE}translators/f1/vocab.json`]);
  const p = h.out.filter((x) => x.type === 'progress').at(-1);
  assert.deepEqual([p.loaded, p.total], [bytesOf(h.manifest, 'ensemble', (f) => !has(f)), bytesOf(h.manifest, 'ensemble', (f) => !has(f))]);
  await tick();
});

test('the fake site serves the pinned runtime files (so the real loader verifies them)', () => {
  for (const [key, file] of Object.entries(RUNTIME_FILES)) assert.equal(sha(readFileSync(file)), RUNTIME[key].sha256, key);
});
