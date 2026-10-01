// Web Worker entry of the in-browser translator: the demo/app.py socket protocol over postMessage, running
// demo/service.py ModelService in the visitor's browser through the parity-tested ports (pose.js, track.js via
// pose.js, features.js, rgb.js, translate.js, cv.js). Nothing leaves the device; nothing is stored but the
// SHA-256-verified model files in Cache Storage (loader.js).
//
// Protocol v4 (v3 plus persons: the result's `persons`, {type: 'retranslate', request_id, person} and {type: 'clear_session'}).
// Page -> worker: {type: 'check'} (capabilities only, no network) | {type: 'init'} (only after the
// visitor consents: manifest, sizes, cache state) | {type: 'load', model} | {type: 'forget'} (delete every cached
// file) | the service messages start_recording, frame (JPEG data URL + captured_at), stop_recording, translate, each
// with request_id.
// Worker -> page: {type: 'capabilities' | 'engine' | 'progress' | 'loaded' | 'forgotten', ...}; failures
// {type: 'unsupported', reason, model?} and {type: 'engine_error', error, stage: 'init' | 'load' | 'runtime', model?};
// and the service replies with `channel` (0 = the pose socket, 1 = the translate socket): pose / frame_ignored /
// recording_started / recording_stopped / result / cancelled (a translation that finished after its utterance was cancelled or
// replaced: no result, but the page may stop waiting) / {error, request_id}. Replies carry the request_id they answer; the
// page drops stale ones.
import { COVERAGE, EP, MANIFEST_SHA256, MAX_FRAMES, MIN_DEVICE_MEMORY_GB, MODEL_BASE, PERSON_CHOICE, POSE_MS_LIMIT } from './config.js';
import { bgrFromRGBA } from './cv.js';
import { assertLive, prepareClip } from './features.js';
import { checkModelBase, createLoader, modelFiles, sha256hex } from './loader.js';
import { Session } from './persons.js';
import { createPose } from './pose.js';
import { rgbFeatures, sampleIndices } from './rgb.js';
import { loadTranslator } from './translate.js';

const F = 399;
const MAX_IMAGE_BYTES = 1_000_000, MAX_IMAGE_PIXELS = 1280 * 960;   // demo/service.py decode_frame
const JPEG_PREFIX = 'data:image/jpeg;base64,';
const STILL_FINISHING = 'Your previous utterance is still finishing. Please wait a moment.';   // app.py start_recording
const POSE_BROKEN = 'Pose estimation stopped working in this browser. Reload the page to try again.';
const POSE_FAILURES = 5;   // consecutive frames whose pose run THREW; one such frame stays an empty frame (service.py)
export const POSE_PROBE = 'detector + 2 pose crops';

/**
 * Python round(x, 1), as app.py rounds the telemetry: the exact binary value to one decimal, exact ties to even
 * (24.25 -> 24.2, 0.15 -> 0.1). An exact tie is x = odd / 4 (x * 4 is exact); anything else is toFixed's exact nearest.
 */
export function round1(x) {
  const q = x * 4;
  if (Number.isInteger(q) && q % 2) { const n = Math.floor(x * 10); return (n % 2 ? n + 1 : n) / 10; }
  return Number(x.toFixed(1));
}

/** A message for the visitor, sent as app.py sends a ValueError's text. */
export class UserError extends Error {}

function jpegBytes(value) {                 // decode_frame's checks, before any image decoding
  if (typeof value !== 'string' || !value.startsWith(JPEG_PREFIX)) throw new UserError('Send a JPEG camera frame.');
  const encoded = value.slice(JPEG_PREFIX.length);
  if (encoded.length > Math.floor(MAX_IMAGE_BYTES * 4 / 3) + 4) throw new UserError('Camera frame is too large.');
  let bin;
  try { bin = atob(encoded); } catch { throw new UserError('The camera frame could not be read.'); }
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

/**
 * The protocol, independent of the models: `runtime` does check / init / load / forget / extract / translate.
 * Pose and translation run one at a time in arrival order (never dropped); a reply for an utterance that was
 * cancelled or replaced meanwhile is withheld or sent as frame_ignored, as app.py does. While a translation runs
 * (even a cancelled one) start_recording is refused, as app.py refuses it while the session's task is running.
 */
export function createEngine(post, runtime, now = () => performance.now(), { choice = PERSON_CHOICE } = {}) {
  let models = null, cur = null, work = Promise.resolve(), loads = Promise.resolve(), translating = 0, poseFailures = 0;
  const session = new Session();            // who has been in view this session (labels and last positions only)
  const reply = (channel, m) => post({ channel, ...m });
  const serial = (fn) => { const p = work.then(fn); work = p.catch(() => {}); return p; };

  async function frame({ request_id: id, captured_at: at, data }) {
    const u = cur;
    if (!u || u.id !== id || !u.recording) return reply(0, { type: 'frame_ignored', request_id: id });
    if (u.keypoints.length >= MAX_FRAMES) throw new UserError('Recording limit reached. Please try a shorter utterance.');
    if (typeof at !== 'number' || !Number.isFinite(at) || at < 0 || at > 10_150 || at - u.lastAt < 20) {
      throw new UserError('Invalid camera timing. Please start a new utterance.');
    }
    u.lastAt = at;
    const t0 = now(), jpeg = jpegBytes(data);
    const { keypoints, count, error, all, picked } = await runtime.extract(jpeg, u.track);
    if (!error) poseFailures = 0;
    else {                                  // the frame stays empty; a pose runtime that keeps throwing is broken
      console.error(error);
      if (++poseFailures >= POSE_FAILURES) {
        poseFailures = 0;
        post({ type: 'engine_error', stage: 'runtime', error: POSE_BROKEN });
        if (cur !== u) return reply(0, { type: 'frame_ignored', request_id: id }); // never clear a newer utterance
        cur = null;
        return reply(0, { error: POSE_BROKEN, request_id: id });
      }
    }
    if (cur !== u || !u.recording) return reply(0, { type: 'frame_ignored', request_id: id });
    u.keypoints.push(keypoints); u.times.push(at);
    u.people.step(all, picked);
    if (models[u.model].needs_frames) u.jpegs.push(jpeg);   // the ensemble's image features need the frames after Finish
    reply(0, { type: 'pose', request_id: id, has_person: count >= 1, person_count: count, buffer_len: u.keypoints.length,
      pose_ms: round1(now() - t0) });
  }

  // What the page is told about who was in view and who was translated. `chosen` is a label or null.
  const personsInfo = (sel, chosenId, mode) => ({
    mode: !sel.multi ? 'single' : chosenId === null ? 'default' : mode === 'manual' ? 'manual' : sel.by,
    chosen: sel.people.find((p) => p.id === chosenId)?.label ?? (sel.multi ? null : sel.people[0]?.label ?? null),
    count: sel.people.length, session_people: session.labels,
    people: sel.people.map((p) => ({ label: p.label, frames: p.frames, share: p.share, eligible: p.eligible, score: p.score, x: p.x })),
  });

  async function run(u, id, keypoints, times, jpegs, model, persons) {
    let out;
    try { out = await runtime.translate(model, keypoints, times, jpegs); } catch (e) {
      if (cur !== u) return reply(1, { type: 'cancelled', request_id: id });
      if (!(e instanceof UserError)) console.error(e);
      return reply(1, { error: e instanceof UserError ? e.message : 'Translation could not be completed. Please try again.', request_id: id });
    }
    if (cur !== u) return reply(1, { type: 'cancelled', request_id: id });   // cancelled or replaced while translating
    const info = models[model], score = out.confidence, empty = !out.text.trim();
    const accepted = !empty && Number.isFinite(score) && score <= info.threshold;
    const span = Math.max(times.at(-1) - times[0], 1);
    reply(1, { type: 'result', request_id: id, text: accepted ? out.text : '', confidence: Number.isFinite(score) ? score : null,
      threshold: info.threshold, coverage_target: info.coverage, accepted,
      reason: accepted ? 'ready' : empty ? 'declined: empty' : 'declined: low confidence',
      mode: 'model', model, model_label: info.label, capture_fps: round1((times.length - 1) * 1000 / span),
      frames: times.length, frames_used: out.frames_used, translate_ms: round1(out.translate_ms), rgb_ms: round1(out.rgb_ms), persons });
  }

  async function translate({ request_id: id }) {
    const u = cur;
    if (!u || u.id !== id || !u.recording) throw new UserError('Start a new utterance before translating.');
    u.recording = false;
    if (u.keypoints.length < 2) { cur = null; throw new UserError('Too few frames were captured. Please try again.'); }
    const { keypoints, times, jpegs, model } = u;
    // Two or more people in view: the person with the most signing-like movement (PERSON_CHOICE) is translated. Otherwise
    // the single-signer stream is used exactly as before.
    const sel = u.people.select(times, choice), chosen = sel.multi ? sel.chosen : null;
    // Buffers are released now. When two or more people were in view their keypoints (and, for the ensemble, the frames)
    // stay until the next utterance, Clear, camera off or the page closes, so that a switch of person can re-run.
    Object.assign(u, { keypoints: [], times: [], jpegs: [], track: {},
      retained: sel.multi ? { times, jpegs: models[model].needs_frames ? jpegs : [], model } : null });
    const persons = personsInfo(sel, chosen, choice);
    const use = chosen !== null ? u.people.keypointsFor(chosen) : keypoints;
    if (!sel.multi) u.people = null;
    return run(u, id, use, times, jpegs, model, persons);
  }

  async function retranslate({ request_id: id, person }) {
    const u = cur;
    if (!u || u.id !== id || u.recording || !u.retained) throw new UserError('That utterance is no longer available. Please sign it again.');
    const { times, jpegs, model } = u.retained;
    const sel = u.people.select(times, u.people.idOf(person) ?? -1);
    if (sel.chosen === null) throw new UserError('That person was not in view for enough of the utterance to translate.');
    return run(u, id, u.people.keypointsFor(sel.chosen), times, jpegs, model, personsInfo(sel, sel.chosen, 'manual'));
  }

  const failed = (e, stage) => {
    if (e instanceof UserError) return { type: 'unsupported', reason: e.message };
    console.error(e);
    return { type: 'engine_error', error: e.message, stage };
  };

  return function handle(msg) {
    const id = msg.request_id;
    switch (msg.type) {
      case 'check':
        return runtime.check().then((c) => post({ type: 'capabilities', ...c }), (e) => post(failed(e, 'init')));
      case 'init':
        return runtime.init().then((r) => { models = r.models; post({ type: 'engine', ...r }); }, (e) => post(failed(e, 'init')));
      case 'load':                         // one download at a time; session creation waits for pose/translation
        loads = loads.then(() => runtime.load(msg.model, (loaded, total) => post({ type: 'progress', model: msg.model, loaded, total }), serial)
          .then((r) => post({ type: 'loaded', model: msg.model, ...r }), (e) => post({ ...failed(e, 'load'), model: msg.model })));
        return loads;
      case 'forget':                       // the page then terminates this worker
        return runtime.forget().then(() => post({ type: 'forgotten' }), (e) => post(failed(e, 'runtime')));
      case 'start_recording':               // app.py: the running translation keeps the session until it finishes
        if (translating) return reply(1, { error: STILL_FINISHING, request_id: id });
        if (!models || !runtime.ready(msg.model)) return reply(1, { error: 'That translation model is not available.', request_id: id });
        cur = { id, model: msg.model, recording: true, keypoints: [], times: [], jpegs: [], track: {}, lastAt: -100,
          people: session.utterance(), retained: null };
        return reply(1, { type: 'recording_started', request_id: id });
      case 'stop_recording':
        if (cur?.id === id) cur = null;
        return reply(1, { type: 'recording_stopped', request_id: id });
      case 'frame':
        return serial(() => frame(msg)).catch((e) => {
          if (e instanceof UserError) return reply(0, { error: e.message, request_id: id });
          console.error(e); cur = null;      // app.py: an unexpected capture failure clears the session
          reply(0, { error: 'Capture could not be completed. Please try again.' });
        });
      case 'translate':
        translating++;
        return serial(() => translate(msg)).catch((e) => reply(1, { error: e.message, request_id: id })).finally(() => { translating--; });
      case 'retranslate':                   // the same utterance, another person (only while its keypoints are retained)
        translating++;
        return serial(() => retranslate(msg)).catch((e) => reply(1, { error: e instanceof UserError ? e.message
          : 'Translation could not be completed. Please try again.', request_id: id })).finally(() => { translating--; });
      case 'clear_session':                 // Clear conversation / camera off: forget who was in view, drop a retained utterance
        session.clear();
        if (cur && !cur.recording) { cur.retained = null; cur.people = null; if (!translating) cur = null; }
        return;
      default:
        return reply(1, { error: 'Unsupported message.', request_id: id });
    }
  };
}

// Decoded as demo/service.py decode_frame does with PIL: Chrome 152 createImageBitmap RGB == PIL RGB on 29/29 studio
// frames (docs/DEMO_DEPLOYMENT.md, smoke test). No colour-space conversion.
async function decodeJPEG(bytes) {
  let bm;
  try { bm = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), { colorSpaceConversion: 'none' }); } catch {
    throw new UserError('The camera frame could not be read.');
  }
  try {
    const c = new OffscreenCanvas(bm.width, bm.height), g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(bm, 0, 0);
    return g.getImageData(0, 0, bm.width, bm.height);
  } finally { bm.close(); }
}

const blobURL = (bytes) => URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' }));

// The probe frame: capture-sized (the studio sends at most 960 x 720) with two people fully in view (their padded
// RTMW crops lie inside the frame), because most studio frames hold more than one person (24 of the 37 G4 validation
// frames) and RTMW runs once per person.
const PROBE_FRAME = { width: 960, height: 720 };
const PROBE_BOXES = Float32Array.of(150, 120, 390, 600, 570, 120, 810, 600);

/** ms of one busy frame's pose work (detector + 2 RTMW crops) on `p` (pose.js DWPose), timed after a warm-up. */
export async function poseProbe(p, now = () => performance.now()) {
  const img = { data: new Uint8Array(PROBE_FRAME.width * PROBE_FRAME.height * 3).fill(128), ...PROBE_FRAME, channels: 3 };
  const once = async () => { await p.detect(img); await p.estimate(img, PROBE_BOXES); };
  await once();                             // WebGPU compiles its shaders on the first run
  const t0 = now();
  await once();
  return now() - t0;
}

/**
 * The real runtime: downloads through loader.js, then the phase-1 modules exactly as their parity tests use them.
 * manifestSha256 is a test seam (fake manifests); the page always uses the pinned config value.
 */
export function browserRuntime({ loader = createLoader(), now = () => performance.now(), manifestSha256 = MANIFEST_SHA256 } = {}) {
  let manifest, ort, pose, dino, caps = null, webgpu = false, probeMs = null, refused = null;
  const translators = {}, providers = {};
  const has = (key) => key === 'ort' || key === 'ortWasm' ? !!ort
    : key.startsWith('pose.') ? !!pose : key === 'dinov2' ? !!dino : false;

  // One execution provider at a time, in EP order, so the provider that actually loaded is known.
  async function withEP(eps, create) {
    let last;
    for (const ep of eps) {
      if (ep === 'webgpu' && !webgpu) continue;
      // logSeverityLevel 3: errors only, not ORT's load-time warnings (unused initialisers, shape ops kept on CPU).
      try { return [await create({ executionProviders: [ep], logSeverityLevel: 3 }), ep]; } catch (e) { last = e; }
    }
    throw last ?? new Error('no execution provider');
  }

  const release = (p) => Promise.all([p.det.release?.(), p.est.release?.()]);
  const loaded = () => ({ providers: { ...providers }, pose_probe_ms: probeMs, pose_probe: POSE_PROBE });
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  // One entry per sha256: the ensemble's six members share one vocabulary file, which is fetched and counted once.
  const unique = (files, seen = new Set()) => files.filter((f) => !seen.has(f.sha256) && seen.add(f.sha256));

  async function check() {                 // capabilities only: no network
    const need = [['WebAssembly', typeof WebAssembly === 'object'], ['OffscreenCanvas', typeof OffscreenCanvas === 'function'],
      ['createImageBitmap', typeof createImageBitmap === 'function'], ['Web Crypto', !!globalThis.crypto?.subtle]];
    for (const [name, ok] of need) if (!ok) throw new UserError(`This browser does not provide ${name}, which the in-browser translator needs.`);
    const mem = globalThis.navigator?.deviceMemory;
    if (mem && mem < MIN_DEVICE_MEMORY_GB) {
      throw new UserError(`This device reports ${mem} GB of memory; the in-browser translator needs at least ${MIN_DEVICE_MEMORY_GB} GB.`);
    }
    webgpu = EP.pose.includes('webgpu') && !!(await globalThis.navigator?.gpu?.requestAdapter().catch(() => null));
    const isolated = !!globalThis.crossOriginIsolated;
    if (!webgpu && !isolated) {
      throw new UserError('This browser offers neither WebGPU nor multi-threaded WebAssembly, so pose estimation would take over 2 s per camera frame. Translation is not offered on this device.');
    }
    return (caps = { webgpu, isolated });
  }

  return {
    ready: (model) => !!translators[model],
    check,

    async init() {
      if (!caps) await check();
      checkModelBase(MODEL_BASE);
      const res = await fetch(`${MODEL_BASE}manifest.json`, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`The model manifest could not be loaded (HTTP ${res.status}).`);
      const raw = new Uint8Array(await res.arrayBuffer());
      if ((await sha256hex(raw)) !== manifestSha256) {
        throw new Error('The model manifest is not the version this page was built for (SHA-256 mismatch), so nothing was loaded.');
      }
      manifest = JSON.parse(new TextDecoder().decode(raw));
      if (!/^[0-9a-f]{40}$/.test(manifest.dinov2?.revision ?? '')) {
        throw new Error('The model manifest does not pin DINOv2 to a commit (dinov2.revision), so nothing was loaded.');
      }
      assertLive(manifest.preprocess);
      await loader.prune(new Set(Object.keys(manifest.models).flatMap((m) => modelFiles(manifest, m).map((f) => f.sha256))));
      const shared = (f) => /^(ort|ortWasm|pose\.)/.test(f.key);   // runtimes and pose: downloaded once for both models
      const models = {};
      for (const [name, spec] of Object.entries(manifest.models)) {
        const files = await Promise.all(unique(modelFiles(manifest, name)).map(async (f) => ({ ...f, hit: await loader.has(f.sha256) })));
        const uncached = async (list) => sum(await Promise.all(list.map((f) => f.hit ? 0 : loader.size(f))));
        models[name] = { label: spec.label, threshold: spec.tau[String(COVERAGE)], coverage: COVERAGE, beam: spec.beam,
          length_penalty: spec.lp, needs_frames: spec.rgb, ...spec.evidence, cached: files.every((f) => f.hit),
          download_bytes: await uncached(files.filter((f) => !shared(f))), shared_download_bytes: await uncached(files.filter(shared)) };
      }
      return { models, max_frames: MAX_FRAMES, runtime: { ...caps } };
    },

    async load(model, onProgress, serial) {
      if (refused) throw new UserError(refused);
      const rgb = manifest.models[model].rgb;
      if (translators[model] && (!rgb || dino)) return loaded();
      const files = modelFiles(manifest, model).filter((f) => !has(f.key));   // what this model still needs
      const need = unique(files), hits = await Promise.all(need.map((f) => loader.has(f.sha256)));
      // Progress counts network bytes only: 0 of 0 when everything comes from Cache Storage.
      let total = sum(await Promise.all(need.map((f, i) => hits[i] ? 0 : loader.size(f)))), done = 0, last = -Infinity;
      const bySha = new Map();
      for (const f of need) {
        let fetched = 0;                    // stays 0 on a cache hit (loader.bytes reports network bytes only)
        bySha.set(f.sha256, await loader.bytes(f, (n) => {
          fetched = n;
          if (now() - last < 100) return;   // at most ten progress messages a second
          last = now(); onProgress(done + n, Math.max(total, done + n));
        }));
        done += fetched; total = Math.max(total, done);
      }
      onProgress(done, total);
      const got = new Map(files.map((f) => [f.key, bySha.get(f.sha256)]));
      await serial(async () => {
        if (!ort) {
          const mod = await import(/* @vite-ignore */ blobURL(got.get('ort')));   // the verified bytes, not a second fetch
          mod.env.wasm.wasmBinary = got.get('ortWasm');
          // One session creation at a time: onnxruntime-web 1.30 refuses concurrent WebGPU session creation
          // ("another WebGPU EP inference session is being created"), and pose.js creates its two in parallel.
          let creating = Promise.resolve();
          const create = (...a) => { const p = creating.then(() => mod.InferenceSession.create(...a)); creating = p.catch(() => {}); return p; };
          ort = { ...mod, InferenceSession: { create } };
        }
        if (!pose) {
          // The probe runs inside the provider loop, so a provider that creates sessions but fails to run falls back.
          [pose, providers.pose] = await withEP(EP.pose, async (sessionOptions) => {
            const p = await createPose({ ort, detector: got.get('pose.detector'), pose: got.get('pose.pose'), sessionOptions });
            try { probeMs = round1(await poseProbe(p, now)); } catch (e) { await release(p); throw e; }
            return p;
          });
          if (probeMs > POSE_MS_LIMIT) {    // a refused device keeps nothing: every cached file is deleted first
            refused = `Pose estimation takes about ${Math.round(probeMs)} ms per camera frame on this device (limit ${POSE_MS_LIMIT} ms), so translation is not offered here. The downloaded files were removed from this browser.`;
            await release(pose); pose = null; delete providers.pose;
            await loader.clear();
            throw new UserError(refused);
          }
        }
        // DINOv2 before the translator is registered, so an ensemble is never reported ready without its image features.
        if (rgb && !dino) [dino, providers.dinov2] = await withEP(EP.dinov2, (o) => ort.InferenceSession.create(got.get('dinov2'), o));
        [translators[model], providers[`translator_${model}`]] = await withEP(EP.translator, (sessionOptions) =>
          loadTranslator({ ort, manifest, model, fetchBytes: async (rel) => got.get(rel), sessionOptions }));
      });
      providers.threads = ort.env.wasm.numThreads;
      return loaded();
    },

    /** Delete every cached model and runtime file (the page's "Remove downloaded models"). */
    async forget() { await loader.clear(); },

    async extract(jpeg, track) {
      const img = await decodeJPEG(jpeg);
      if (img.width * img.height > MAX_IMAGE_PIXELS) throw new UserError('Use a camera resolution of 1280 × 960 or lower.');
      return pose.extract(bgrFromRGBA(img), track);
    },

    // ModelService.translate: 25 fps grid + visibility check + normalize_clip, RGB features for the ensemble, decode.
    async translate(model, keypoints, times, jpegs) {
      let clip;
      try { clip = prepareClip(keypoints, times); } catch (e) { throw new UserError(e.message); }
      let rgb = null, R = 0, rgbMs = 0;
      if (manifest.models[model].rgb) {
        const t0 = now(), pick = sampleIndices(clip.idx.length), frames = [], kps = new Float32Array(pick.length * F);
        for (const [i, j] of pick.entries()) {
          frames.push(await decodeJPEG(jpegs[clip.idx[j]]));
          kps.set(clip.kps.subarray(j * F, (j + 1) * F), i * F);
        }
        rgb = await rgbFeatures(frames, kps, { ort, session: dino });
        R = pick.length; rgbMs = now() - t0;
      }
      const t0 = now(), out = await translators[model].translate(clip.feats, clip.T, rgb, R);
      return { text: out.text, confidence: out.confidence, translate_ms: now() - t0, rgb_ms: rgbMs, frames_used: clip.idx.length };
    },
  };
}

if (typeof WorkerGlobalScope === 'function' && globalThis instanceof WorkerGlobalScope) {
  const handle = createEngine((m) => postMessage(m), browserRuntime());
  onmessage = (e) => handle(e.data);
}
