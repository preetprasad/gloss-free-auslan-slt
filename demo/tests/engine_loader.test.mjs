// loader.js (verified, cached, streamed model downloads) and the engine.js worker protocol, with fakes only:
// no models, no network, no browser. The real pieces behind them are covered by engine_{pose,features,rgb,
// translate}.test.mjs; the in-browser run is the lead's G6.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { HF_BASE, MANIFEST_SHA256, MODEL_BASE, RUNTIME } from '../static/engine/config.js';
import { createEngine, round1, UserError } from '../static/engine/engine.js';
import { checkModelBase, createLoader, modelFiles, sha256hex } from '../static/engine/loader.js';

const sha = (b) => createHash('sha256').update(b).digest('hex');
const bytes = (n, seed = 7) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 255);

// A server that streams `body` in `chunk`-byte pieces and records every request.
function server(files, { chunk = 1000, length = true } = {}) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push({ url, method: init.method ?? 'GET' });
    const body = files[url];
    if (!body) return new Response('missing', { status: 404 });
    const headers = length ? { 'content-length': String(body.length) } : {};
    if (init.method === 'HEAD') return new Response(null, { headers });
    let o = 0;
    const stream = new ReadableStream({ pull(c) { if (o >= body.length) c.close(); else { c.enqueue(body.slice(o, o + chunk)); o += chunk; } } });
    return new Response(stream, { headers });
  };
  return { fetch, calls };
}

// Cache Storage with one named cache, as the browser's (match / put / delete by URL).
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

test('sha256hex == node:crypto', async () => {
  for (const n of [0, 1, 55, 64, 100000]) assert.equal(await sha256hex(bytes(n)), sha(bytes(n)));
});

test('a checksum mismatch is rejected and nothing is cached', async () => {
  const body = bytes(5000), { fetch } = server({ '/m/a.onnx': body }), store = cacheStorage();
  const loader = createLoader({ fetch, caches: store.caches });
  await assert.rejects(loader.bytes({ url: '/m/a.onnx', sha256: sha(bytes(5000, 8)) }), /Checksum mismatch for \/m\/a\.onnx; the file was not used\./);
  assert.equal(store.entries.size, 0);
  await assert.rejects(loader.bytes({ url: '/m/missing.onnx', sha256: sha(body) }), /HTTP 404/);
});

test('progress is streamed, verified bytes are returned, and a second load is a cache hit with no network or progress', async () => {
  const body = bytes(4500), file = { url: '/m/a.onnx', sha256: sha(body) };
  const { fetch, calls } = server({ '/m/a.onnx': body }), store = cacheStorage();
  const loader = createLoader({ fetch, caches: store.caches });
  assert.equal(await loader.has(file.sha256), false);
  const seen = [];
  const got = await loader.bytes(file, (loaded, total) => seen.push([loaded, total]));
  assert.deepEqual(seen, [[1000, 4500], [2000, 4500], [3000, 4500], [4000, 4500], [4500, 4500]]);
  assert.equal(sha(got), file.sha256);
  assert.equal(await loader.has(file.sha256), true);
  assert.deepEqual([...store.entries.keys()], [`https://model-cache.invalid/sha256/${file.sha256}`]);
  const again = createLoader({ fetch: async () => { throw new Error('network used on a cache hit'); }, caches: store.caches });
  const hits = [];
  assert.deepEqual(await again.bytes(file, (l, t) => hits.push([l, t])), got);
  assert.deepEqual(hits, []);                              // progress reports network bytes only
  assert.equal(calls.length, 1);
});

test('a corrupted cache entry is dropped and downloaded again', async () => {
  const body = bytes(3000), file = { url: '/m/a.onnx', sha256: sha(body) };
  const { fetch, calls } = server({ '/m/a.onnx': body }), store = cacheStorage();
  store.entries.set(`https://model-cache.invalid/sha256/${file.sha256}`, bytes(3000, 9));
  const got = await createLoader({ fetch, caches: store.caches }).bytes(file);
  assert.equal(sha(got), file.sha256);
  assert.equal(calls.length, 1);
  assert.equal(sha(store.entries.get(`https://model-cache.invalid/sha256/${file.sha256}`)), file.sha256);
});

test('prune keeps the listed hashes and drops the rest (an old model version)', async () => {
  const [a, b] = [bytes(100, 1), bytes(100, 2)], store = cacheStorage();
  const loader = createLoader({ fetch: server({ '/a': a, '/b': b }).fetch, caches: store.caches });
  await loader.bytes({ url: '/a', sha256: sha(a) }); await loader.bytes({ url: '/b', sha256: sha(b) });
  await loader.prune(new Set([sha(a)]));
  assert.deepEqual([await loader.has(sha(a)), await loader.has(sha(b))], [true, false]);
  await createLoader({ caches: null }).prune(new Set());   // no Cache Storage: nothing to do
});

test('clear() deletes every cached file, including one being written, and caches nothing afterwards', async () => {
  const [a, b, c] = [bytes(100, 1), bytes(100, 2), bytes(100, 3)], store = cacheStorage();
  let releasePut, gated = true;                            // the first cache write waits for the test
  const put = store.caches;
  const slow = { open: async () => { const x = await put.open(); return { ...x, put: async (k, r) => {
    if (gated) { gated = false; await new Promise((ok) => { releasePut = ok; }); }
    return x.put(k, r);
  } }; } };
  const loader = createLoader({ fetch: server({ '/a': a, '/b': b, '/c': c }).fetch, caches: put });
  await loader.bytes({ url: '/a', sha256: sha(a) });
  const writing = createLoader({ fetch: server({ '/b': b }).fetch, caches: slow });
  const pending = writing.bytes({ url: '/b', sha256: sha(b) });
  while (!releasePut) await new Promise((r) => setImmediate(r));   // b is downloaded and its cache write has started
  const cleared = writing.clear();
  releasePut();
  assert.equal(sha(await pending), sha(b));
  await cleared;
  assert.equal(store.entries.size, 0);                     // a (older) and b (in flight) are both gone
  assert.equal(sha(await writing.bytes({ url: '/b', sha256: sha(b) })), sha(b));   // still verified and usable ...
  assert.equal(store.entries.size, 0);                     // ... but no longer cached
  await loader.bytes({ url: '/c', sha256: sha(c) });       // another loader instance (another worker) is unaffected
  assert.equal(store.entries.size, 1);
});

test('MODEL_BASE: this site, or a commit-pinned Hugging Face revision, nothing else', () => {
  const pinned = `https://huggingface.co/preetprasad/auslan-slt-live/resolve/${'0123456789abcdef'.repeat(2)}01234567/`;
  for (const [base, origin] of [['/models/', 'https://auslan-research.vercel.app'], ['/models/', undefined],
    ['http://127.0.0.1:8000/models/', 'http://127.0.0.1:8000'], [pinned, 'https://auslan-research.vercel.app'], [pinned, undefined]]) {
    assert.doesNotThrow(() => checkModelBase(base, origin), `${base} from ${origin}`);
  }
  for (const base of ['https://huggingface.co/preetprasad/auslan-slt-live/resolve/main/', pinned.replace('https:', 'http:'),
    pinned.slice(0, -1), pinned.replace('/resolve/', '/blob/'), pinned.replace('0123456789abcdef', '0123456789ABCDEF'),
    pinned.replace('huggingface.co', 'huggingface.co.evil.example'), '//evil.example/models/', 'https://evil.example/models/']) {
    assert.throws(() => checkModelBase(base, 'https://auslan-research.vercel.app'), /neither this site nor a commit-pinned Hugging Face revision/, base);
  }
  assert.throws(() => checkModelBase('//evil.example/models/', undefined), /neither this site/);
  assert.doesNotThrow(() => checkModelBase(MODEL_BASE));
});

test('MANIFEST_SHA256 pins the exported manifest (re-pin config.js whenever the export changes)', () => {
  assert.equal(sha(readFileSync(new URL('../../results/live_onnx/manifest.json', import.meta.url))), MANIFEST_SHA256);
});

// Python round(x, 1) on the exact binary value, ties to even (BigInt arithmetic, independent of engine.js).
function pyRound1(x) {
  if (!Number.isFinite(x) || x === 0) return x;
  const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x);
  const bits = dv.getBigUint64(0), ex = Number((bits >> 52n) & 0x7ffn), frac = bits & ((1n << 52n) - 1n);
  const m = ex ? frac | (1n << 52n) : frac, e = (ex || 1) - 1075;
  const num = m * 10n << BigInt(Math.max(e, 0)), den = 1n << BigInt(Math.max(-e, 0));
  let q = num / den; const r = num % den;
  if (2n * r > den || (2n * r === den && q % 2n)) q += 1n;
  const v = Number(q) / 10;
  return bits >> 63n ? -v : v;
}

test('round1 == Python round(x, 1): half to even on exact ties, exact decimal value otherwise', () => {
  // Python 3 (repo .venv) printed these: [round(v, 1) for v in ...].
  const py = [[0.15, 0.1], [0.25, 0.2], [0.35, 0.3], [0.45, 0.5], [0.05, 0.1], [24.25, 24.2], [24.35, 24.4], [97000 / 4000, 24.2],
    [1.25, 1.2], [-0.25, -0.2], [-0.15, -0.1], [0.75, 0.8], [123.45, 123.5], [41.26, 41.3], [152.25, 152.2], [5.55, 5.5], [2.5, 2.5]];
  for (const [x, want] of py) { assert.equal(round1(x), want, String(x)); assert.equal(pyRound1(x), want, `reference ${x}`); }
  assert.ok(Object.is(round1(-1e-9), -0) && Object.is(round1(0), 0));
  let rnd = 12345;
  const next = () => (rnd = (Math.imul(rnd, 1103515245) + 12345) >>> 0) / 2 ** 32;
  const xs = [];
  for (let i = 0; i < 20000; i++) xs.push(next() * 2000, Math.round(next() * 20000) / 4, Math.round(next() * 20000) / 20,
    (Math.floor(next() * 300) + 1) * 1000 / (next() * 12000 + 1), -next() * 50);
  let ties = 0;
  for (const x of xs) {
    if (Number.isInteger(x * 4) && (x * 4) % 2) ties++;
    assert.equal(round1(x), pyRound1(x), String(x));
  }
  assert.ok(ties > 5000, `ties exercised: ${ties}`);
  assert.equal(round1(Math.round(97) * 1000 / 4000), 24.2);   // the review's capture_fps case (app.py prints 24.2)
});

test('the declared size only sizes the buffer: too small, too large or absent, the hash still decides', async () => {
  const body = bytes(2500), { fetch } = server({ '/m/a': body }, { length: false });
  const loader = createLoader({ fetch, caches: null });   // no Cache Storage (e.g. a private window): still works
  for (const size of [0, 1000, 2500, 9000]) assert.equal(sha(await loader.bytes({ url: '/m/a', sha256: sha(body), bytes: size })), sha(body), `size ${size}`);
  assert.equal(await loader.size({ url: '/m/a', bytes: 77 }), 77);
  const withLength = createLoader(server({ '/m/a': body })).size({ url: '/m/a' });
  assert.equal(await withLength, 2500);                    // HEAD Content-Length when the manifest has no size
});

test('modelFiles lists the runtimes, pose models, members in recorded order and DINOv2, with their hashes', () => {
  const manifest = JSON.parse(readFileSync(new URL('../../results/live_onnx/manifest.json', import.meta.url)));
  const single = modelFiles(manifest, 'single');
  assert.deepEqual(single.map((f) => f.key), ['ort', 'ortWasm', 'pose.detector', 'pose.pose',
    'translators/w4A_s202/encoder.onnx', 'translators/w4A_s202/decoder.onnx', 'translators/w4A_s202/vocab.json']);
  assert.equal(single[2].url, `${MODEL_BASE}pose/yolox_m.onnx`);
  assert.equal(single[2].sha256, manifest.pose.detector.sha256);
  assert.equal(single[0].sha256, RUNTIME.ort.sha256);
  const ens = modelFiles(manifest, 'ensemble');
  assert.deepEqual(ens.filter((f) => f.key.endsWith('encoder.onnx')).map((f) => f.key.split('/')[1]),
    ['vlp80', 'rgb_fusion', 'vlp40', 'ctc_s202', 'fus_s101', 'fus_s202']);
  const d = manifest.dinov2, dino = ens.at(-1);
  assert.equal(dino.key, 'dinov2');
  assert.equal(dino.url, d.file ? MODEL_BASE + d.file : `${HF_BASE}${d.hf_model}/resolve/${d.revision}/${d.path}`);
  assert.equal(dino.sha256, d.sha256);
  for (const f of ens) assert.match(f.sha256, /^[0-9a-f]{64}$/, f.key);
});

// ---- engine.js protocol with a fake runtime (pose / translation replaced by deterministic stand-ins).
const MODELS = { single: { label: 'S', threshold: 0.7, coverage: 0.5, needs_frames: false }, ensemble: { label: 'E', threshold: 0.9, coverage: 0.5, needs_frames: true } };
const JPEG = (k) => `data:image/jpeg;base64,${btoa(String.fromCharCode(0xff, 0xd8, k))}`;

function harness(overrides = {}) {
  const out = [], gates = [];
  let clock = 0;
  const runtime = {
    init: async () => ({ models: MODELS, max_frames: 300, runtime: { webgpu: true, isolated: true } }),
    ready: (m) => m in MODELS,
    load: async (model, progress) => { progress(5, 10); progress(10, 10); return { providers: { pose: 'webgpu' }, pose_probe_ms: 120 }; },
    // Pose waits until the test releases it, so the order of work is observable.
    extract: (jpeg, track) => new Promise((resolve) => gates.push(() => { clock += 150; track.n = (track.n ?? 0) + 1; resolve({ keypoints: Float32Array.of(jpeg[2]), count: 1 }); })),
    translate: async (model, keypoints, times, jpegs) => ({ text: 'hello there.', confidence: 0.5, translate_ms: 41.26, rgb_ms: jpegs.length ? 900 : 0,
      frames_used: keypoints.length, seen: { keypoints: [...keypoints].map((k) => k[0]), times, jpegs: jpegs.length } }),
    ...overrides,
  };
  const handle = createEngine((m) => out.push(m), runtime, () => clock);
  const tick = () => new Promise((r) => setImmediate(r));
  return { out, handle, runtime, release: async () => { gates.shift()(); await tick(); await tick(); }, tick,
    replies: (type) => out.filter((m) => m.type === type), async ready() { await handle({ type: 'init' }); } };
}

test('engine: frames are posed one at a time in arrival order; Finish translates what was captured', async () => {
  const h = harness(); await h.ready();
  assert.equal(h.out[0].type, 'engine');
  h.handle({ type: 'start_recording', request_id: 'r1', model: 'single' });
  assert.deepEqual(h.out.at(-1), { channel: 1, type: 'recording_started', request_id: 'r1' });
  [0, 40, 80].forEach((at, k) => h.handle({ type: 'frame', request_id: 'r1', captured_at: at, data: JPEG(k + 1) }));
  await h.tick();
  assert.equal(h.replies('pose').length, 0);
  await h.release();
  assert.deepEqual(h.replies('pose').map((m) => m.buffer_len), [1]);   // the second waits for the first
  await h.release(); await h.release();
  assert.deepEqual(h.replies('pose').at(-1), { channel: 0, type: 'pose', request_id: 'r1', has_person: true, person_count: 1, buffer_len: 3, pose_ms: 150 });
  h.handle({ type: 'translate', request_id: 'r1' }); await h.tick(); await h.tick();
  const r = h.replies('result')[0];
  assert.deepEqual(r, { channel: 1, type: 'result', request_id: 'r1', text: 'hello there.', confidence: 0.5, threshold: 0.7, coverage_target: 0.5,
    accepted: true, reason: 'ready', mode: 'model', model: 'single', model_label: 'S', capture_fps: 25, frames: 3, frames_used: 3,
    translate_ms: 41.3, rgb_ms: 0, persons: { mode: 'single', chosen: null, count: 0, session_people: 0, people: [] } });
});

test('engine: the ensemble keeps the JPEGs for image features; the single model does not', async () => {
  const seen = {};
  const h = harness({ translate: async (model, k, times, jpegs) => { seen[model] = jpegs.map((j) => j[2]); return { text: 'a.', confidence: 0.1, translate_ms: 1, rgb_ms: 0, frames_used: 2 }; } });
  await h.ready();
  for (const [model, id] of [['single', 's'], ['ensemble', 'e']]) {
    h.handle({ type: 'start_recording', request_id: id, model });
    [0, 40].forEach((at, k) => h.handle({ type: 'frame', request_id: id, captured_at: at, data: JPEG(k + 5) }));
    await h.tick(); await h.release(); await h.release();
    h.handle({ type: 'translate', request_id: id }); await h.tick(); await h.tick();
  }
  assert.deepEqual(seen, { single: [], ensemble: [5, 6] });
});

test('engine: a cancelled or replaced utterance gets frame_ignored and no result', async () => {
  const h = harness(); await h.ready();
  h.handle({ type: 'start_recording', request_id: 'old', model: 'single' });
  h.handle({ type: 'frame', request_id: 'old', captured_at: 0, data: JPEG(1) }); await h.tick();
  h.handle({ type: 'stop_recording', request_id: 'old' });                         // while that frame is being posed
  assert.deepEqual(h.out.at(-1), { channel: 1, type: 'recording_stopped', request_id: 'old' });
  h.handle({ type: 'start_recording', request_id: 'new', model: 'single' });
  h.handle({ type: 'frame', request_id: 'old', captured_at: 40, data: JPEG(2) });   // late frame of the old utterance
  await h.release();
  await h.tick();
  assert.deepEqual(h.out.filter((m) => m.channel === 0).map((m) => [m.type, m.request_id]), [['frame_ignored', 'old'], ['frame_ignored', 'old']]);
  h.handle({ type: 'frame', request_id: 'new', captured_at: 0, data: JPEG(3) }); await h.tick(); await h.release();
  h.handle({ type: 'frame', request_id: 'new', captured_at: 40, data: JPEG(4) }); await h.tick(); await h.release();
  let finish;
  h.runtime.translate = () => new Promise((r) => { finish = r; });
  h.handle({ type: 'translate', request_id: 'new' }); await h.tick(); await h.tick();
  h.handle({ type: 'stop_recording', request_id: 'new' });                          // Cancel while translating
  finish({ text: 'stale.', confidence: 0.1, translate_ms: 1, rgb_ms: 0, frames_used: 2 }); await h.tick(); await h.tick();
  assert.equal(h.replies('result').length, 0);
});

test('engine: start_recording while a translation runs is refused with app.py\'s wording; the old utterance is kept', async () => {
  const h = harness(); await h.ready();
  const run = async (id) => {
    h.handle({ type: 'start_recording', request_id: id, model: 'single' });
    [0, 40].forEach((at, k) => h.handle({ type: 'frame', request_id: id, captured_at: at, data: JPEG(k + 1) }));
    await h.tick(); await h.release(); await h.release();
  };
  await run('a');
  let finish;
  h.runtime.translate = (model, keypoints) => new Promise((r) => { finish = () => r({ text: 'done.', confidence: 0.1, translate_ms: 1, rgb_ms: 0, frames_used: keypoints.length }); });
  h.handle({ type: 'translate', request_id: 'a' }); await h.tick(); await h.tick();
  h.handle({ type: 'start_recording', request_id: 'b', model: 'single' });
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'Your previous utterance is still finishing. Please wait a moment.', request_id: 'b' });
  h.handle({ type: 'frame', request_id: 'b', captured_at: 0, data: JPEG(3) });   // queued behind the translation
  finish(); await h.tick(); await h.tick();
  assert.deepEqual([h.replies('result').at(-1).request_id, h.replies('result').at(-1).text], ['a', 'done.']);   // not replaced
  assert.deepEqual(h.out.at(-1), { channel: 0, type: 'frame_ignored', request_id: 'b' });
  h.handle({ type: 'stop_recording', request_id: 'a' });
  await run('c');                                          // translation finished: a new utterance starts
  assert.deepEqual(h.replies('recording_started').map((m) => m.request_id), ['a', 'c']);
  assert.equal(h.replies('pose').at(-1).request_id, 'c');
  // A cancelled translation still holds the session until it ends (app.py: s.task is not cancelled by Cancel).
  h.handle({ type: 'translate', request_id: 'c' }); await h.tick(); await h.tick();
  h.handle({ type: 'stop_recording', request_id: 'c' });
  h.handle({ type: 'start_recording', request_id: 'd', model: 'single' });
  assert.equal(h.out.at(-1).error, 'Your previous utterance is still finishing. Please wait a moment.');
  finish(); await h.tick(); await h.tick();
  assert.equal(h.replies('result').length, 1);             // c was cancelled: no result
  h.handle({ type: 'start_recording', request_id: 'd', model: 'single' });
  assert.deepEqual(h.out.at(-1), { channel: 1, type: 'recording_started', request_id: 'd' });
});

test('engine: a pose run that threw is an empty frame; five in a row are a runtime engine_error', async () => {
  const errors = [], logged = console.error;
  console.error = (e) => errors.push(e);
  try {
    let fail = [];
    const h = harness({ extract: async (jpeg) => fail.shift()
      ? { keypoints: new Float32Array(399), count: 0, error: new Error('WebGPU device lost') }
      : { keypoints: Float32Array.of(jpeg[2]), count: jpeg[2] ? 1 : 0 } });
    await h.ready();
    h.handle({ type: 'start_recording', request_id: 'r', model: 'single' });
    const send = async (at, k) => { h.handle({ type: 'frame', request_id: 'r', captured_at: at, data: JPEG(k) }); await h.tick(); await h.tick(); };
    fail = [true, true, true, true, false];                // four throws, then a good frame: the count restarts
    for (let i = 0; i < 5; i++) await send(40 * i, i + 1);
    assert.deepEqual(h.replies('pose').map((m) => [m.has_person, m.buffer_len]), [[false, 1], [false, 2], [false, 3], [false, 4], [true, 5]]);
    fail = [true, true, true, true];
    for (let i = 5; i < 9; i++) await send(40 * i, i + 1);
    assert.equal(h.replies('engine_error').length, 0);
    assert.equal(h.replies('pose').at(-1).buffer_len, 9);  // each failed frame is kept, empty, as service.py keeps it
    fail = [true];
    await send(400, 11);
    const msg = 'Pose estimation stopped working in this browser. Reload the page to try again.';
    assert.deepEqual(h.out.slice(-2), [{ type: 'engine_error', stage: 'runtime', error: msg }, { channel: 0, error: msg, request_id: 'r' }]);
    assert.equal(errors.length, 9);                        // every throw is logged
    await send(440, 12);                                   // the utterance was cleared
    assert.deepEqual(h.out.at(-1), { channel: 0, type: 'frame_ignored', request_id: 'r' });
    h.handle({ type: 'translate', request_id: 'r' }); await h.tick(); await h.tick();
    assert.deepEqual(h.out.at(-1), { channel: 1, error: 'Start a new utterance before translating.', request_id: 'r' });
  } finally { console.error = logged; }
});

test('engine: decline rules and the service error messages', async () => {
  const cases = [[{ text: 'hmm.', confidence: 0.71 }, false, 'declined: low confidence', ''], [{ text: '  ', confidence: 0.2 }, false, 'declined: empty', ''],
    [{ text: 'x.', confidence: NaN }, false, 'declined: low confidence', ''], [{ text: 'yes.', confidence: 0.7 }, true, 'ready', 'yes.']];
  for (const [o, accepted, reason, text] of cases) {
    const h = harness({ translate: async () => ({ translate_ms: 1, rgb_ms: 0, frames_used: 2, ...o }) }); await h.ready();
    h.handle({ type: 'start_recording', request_id: 'r', model: 'single' });
    [0, 40].forEach((at, k) => h.handle({ type: 'frame', request_id: 'r', captured_at: at, data: JPEG(k) }));
    await h.tick(); await h.release(); await h.release();
    h.handle({ type: 'translate', request_id: 'r' }); await h.tick(); await h.tick();
    const r = h.replies('result')[0];
    assert.deepEqual([r.accepted, r.reason, r.text], [accepted, reason, text], JSON.stringify(o));
    if (!Number.isFinite(o.confidence)) assert.equal(r.confidence, null);
  }
  const h = harness({ translate: async () => { throw new UserError('Keep one signer in view for the whole utterance, then try again.'); } });
  await h.ready();
  h.handle({ type: 'translate', request_id: 'none' }); await h.tick(); await h.tick();
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'Start a new utterance before translating.', request_id: 'none' });
  h.handle({ type: 'start_recording', request_id: 'r', model: 'single' });
  h.handle({ type: 'frame', request_id: 'r', captured_at: 0, data: 'data:image/png;base64,AAAA' }); await h.tick(); await h.tick();
  assert.deepEqual(h.out.at(-1), { channel: 0, error: 'Send a JPEG camera frame.', request_id: 'r' });
  h.handle({ type: 'frame', request_id: 'r', captured_at: 10, data: JPEG(1) }); await h.tick(); await h.tick();
  assert.deepEqual(h.out.at(-1), { channel: 0, error: 'Invalid camera timing. Please start a new utterance.', request_id: 'r' });
  h.handle({ type: 'frame', request_id: 'r', captured_at: 10151, data: JPEG(1) }); await h.tick(); await h.tick();
  assert.equal(h.out.at(-1).error, 'Invalid camera timing. Please start a new utterance.');
  h.handle({ type: 'frame', request_id: 'r', captured_at: 20000 - 9000, data: 'data:image/jpeg;base64,' + 'A'.repeat(1333338) }); await h.tick();
  assert.equal(h.out.at(-1).error, 'Invalid camera timing. Please start a new utterance.');   // timing is checked first, as app.py
  h.handle({ type: 'translate', request_id: 'r' }); await h.tick(); await h.tick();
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'Too few frames were captured. Please try again.', request_id: 'r' });
  h.handle({ type: 'start_recording', request_id: 'r2', model: 'single' });
  [0, 40].forEach((at, k) => h.handle({ type: 'frame', request_id: 'r2', captured_at: at, data: JPEG(k) }));
  await h.tick(); await h.release(); await h.release();
  h.handle({ type: 'translate', request_id: 'r2' }); await h.tick(); await h.tick();
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'Keep one signer in view for the whole utterance, then try again.', request_id: 'r2' });
  h.handle({ type: 'start_recording', request_id: 'r3', model: 'nope' });
  assert.deepEqual(h.out.at(-1), { channel: 1, error: 'That translation model is not available.', request_id: 'r3' });
});

test('engine: check / init / load / forget report capabilities, progress, providers and failures by stage', async () => {
  const quiet = console.error; console.error = () => {};
  const h = harness({ check: async () => ({ webgpu: false, isolated: true }), forget: async () => {} });
  await h.handle({ type: 'check' });
  assert.deepEqual(h.out.shift(), { type: 'capabilities', webgpu: false, isolated: true });
  await h.ready();
  await h.handle({ type: 'load', model: 'single' });
  assert.deepEqual(h.out.slice(1), [{ type: 'progress', model: 'single', loaded: 5, total: 10 }, { type: 'progress', model: 'single', loaded: 10, total: 10 },
    { type: 'loaded', model: 'single', providers: { pose: 'webgpu' }, pose_probe_ms: 120 }]);
  const slow = harness({ load: async () => { throw new UserError('Pose estimation takes about 2000 ms per camera frame on this device (limit 800 ms), so translation is not offered here.'); } });
  await slow.ready(); await slow.handle({ type: 'load', model: 'single' });
  assert.deepEqual(slow.out.at(-1), { type: 'unsupported', model: 'single', reason: 'Pose estimation takes about 2000 ms per camera frame on this device (limit 800 ms), so translation is not offered here.' });
  const net = harness({ load: async () => { throw new Error('Could not download /models/pose/yolox_m.onnx (HTTP 503).'); } });
  await net.ready(); await net.handle({ type: 'load', model: 'single' });
  assert.deepEqual(net.out.at(-1), { type: 'engine_error', stage: 'load', model: 'single', error: 'Could not download /models/pose/yolox_m.onnx (HTTP 503).' });
  const old = harness({ check: async () => { throw new UserError('This browser does not provide OffscreenCanvas, which the in-browser translator needs.'); } });
  await old.handle({ type: 'check' });
  assert.deepEqual(old.out, [{ type: 'unsupported', reason: 'This browser does not provide OffscreenCanvas, which the in-browser translator needs.' }]);
  const gone = harness({ init: async () => { throw new Error('The model manifest could not be loaded (HTTP 404).'); } });
  await gone.handle({ type: 'init' });
  assert.deepEqual(gone.out, [{ type: 'engine_error', stage: 'init', error: 'The model manifest could not be loaded (HTTP 404).' }]);
  await h.handle({ type: 'forget' });
  assert.deepEqual(h.out.at(-1), { type: 'forgotten' });
  const stuck = harness({ forget: async () => { throw new Error('cache unavailable'); } });
  await stuck.handle({ type: 'forget' });
  assert.deepEqual(stuck.out, [{ type: 'engine_error', stage: 'runtime', error: 'cache unavailable' }]);
  console.error = quiet;
});
