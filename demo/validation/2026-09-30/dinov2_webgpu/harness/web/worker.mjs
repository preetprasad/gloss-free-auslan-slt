// DINOv2 WebGPU-vs-WASM harness worker (PREREG_DINOV2_WEBGPU_2026-09-30). Served at /static/engine/dgpu/, so '../x.js'
// are the SHIPPED engine modules, unchanged. Loads onnxruntime-web and DINOv2 as engine.js browserRuntime().load does:
// loader.bytes (sha256-verified) -> blob import of the verified ort bytes, env.wasm.wasmBinary = verified asyncify wasm,
// one session creation at a time, session options { executionProviders: [ep], logSeverityLevel: 3 }. The manifest is
// fetched and pinned exactly as init() does. Each clip: the SHIPPED rgbFeatures once per session on the same frames and
// keypoints; only session.run is timed separately (a proxy session that forwards to the real one).
// Messages: page -> {type:'run', start, end, redo, localRuntime}; worker -> {type:'status'|'done'|'error', ...}.
import { MANIFEST_SHA256, MODEL_BASE, RUNTIME } from '../config.js';
import { createLoader, modelFiles, sha256hex } from '../loader.js';
import { rgbFeatures, sampleIndices } from '../rgb.js';
import { compare, parseClip, toFrames } from './clip.mjs';

const EPS = ['webgpu', 'wasm'];
const LOCAL_RUNTIME = { ort: '/ortdist/ort.webgpu.bundle.min.mjs', ortWasm: '/ortdist/ort-wasm-simd-threaded.asyncify.wasm' };
const loader = createLoader();
const post = (m) => postMessage(m);
const blobURL = (bytes) => URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' }));
let ort = null, sessions = null, setupInfo = null, warmed = false;

async function upload(name, body) {
  const r = await fetch(`/__upload/${name}`, { method: 'POST', headers: { 'X-Dev-Upload': '1' }, body });
  if (!r.ok) throw new Error(`upload ${name}: HTTP ${r.status}`);
}

async function setup(localRuntime) {
  if (sessions) return;
  const t0 = performance.now();
  // localRuntime: the same files from demo/node_modules (no network); loader.bytes still refuses any other sha256.
  const rt = (k) => (localRuntime ? { ...RUNTIME[k], url: LOCAL_RUNTIME[k] } : RUNTIME[k]);
  const ortBytes = await loader.bytes(rt('ort')), wasm = await loader.bytes(rt('ortWasm'));
  const mod = await import(/* @vite-ignore */ blobURL(ortBytes));
  mod.env.wasm.wasmBinary = wasm;
  let creating = Promise.resolve();
  const create = (...a) => { const p = creating.then(() => mod.InferenceSession.create(...a)); creating = p.catch(() => {}); return p; };
  ort = { ...mod, InferenceSession: { create } };

  const res = await fetch(`${MODEL_BASE}manifest.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`manifest HTTP ${res.status}`);
  const raw = new Uint8Array(await res.arrayBuffer());
  if ((await sha256hex(raw)) !== MANIFEST_SHA256) throw new Error('manifest sha256 != config.js MANIFEST_SHA256');
  const manifest = JSON.parse(new TextDecoder().decode(raw));
  const dino = modelFiles(manifest, 'ensemble').find((f) => f.key === 'dinov2');
  const bytes = await loader.bytes(dino);                 // the shipped, hash-checked DINOv2 ONNX

  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  // GPUAdapterInfo is a platform object: copy its fields explicitly (a spread gives {}).
  const gpuInfo = adapter ? Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map((k) => [k, adapter.info?.[k] ?? null])) : null;
  const made = {}, created = {};
  for (const ep of EPS) {                                 // one at a time; no fallback: the arm is exactly this EP
    const t1 = performance.now();
    try {
      made[ep] = await ort.InferenceSession.create(bytes, { executionProviders: [ep], logSeverityLevel: 3 });
      created[ep] = { provider: ep, ok: true, create_ms: performance.now() - t1 };
    } catch (e) { created[ep] = { provider: null, ok: false, error: String(e?.message ?? e) }; }
  }
  setupInfo = { id: `setup_${Date.now()}`, created, ort_version: ort.env.versions?.web ?? null, runtime_source: localRuntime ? 'local /ortdist (sha256-pinned)' : 'jsDelivr (config.js RUNTIME)',
    threads: ort.env.wasm.numThreads, isolated: globalThis.crossOriginIsolated, ua: navigator.userAgent,
    hardware_concurrency: navigator.hardwareConcurrency, gpu_adapter: gpuInfo,
    dinov2: { url: dino.url, sha256: dino.sha256 }, session_options: { logSeverityLevel: 3 }, setup_ms: performance.now() - t0 };
  await upload(`${setupInfo.id}.json`, JSON.stringify(setupInfo));
  post({ type: 'status', setup: setupInfo });
  if (!created.webgpu.ok || !created.wasm.ok) throw new Error(`session creation failed: ${JSON.stringify(created)}`);
  sessions = made;                                        // only a complete pair counts as set up
}

async function fetchClip(cid) {
  const r = await fetch(`/dgpu/frames/${cid}`);
  if (!r.ok) throw new Error(`frames ${cid}: HTTP ${r.status} ${await r.text()}`);
  const c = parseClip(await r.arrayBuffer());
  const want = sampleIndices(c.T);                         // the shipped JS sampling must agree with the Python sampling
  if (JSON.stringify(want) !== JSON.stringify(c.idx)) throw new Error(`${cid}: sampleIndices(${c.T}) != payload idx`);
  return { cid, ...c, frames: toFrames(c) };
}

// A session proxy that times run() and forwards everything rgbFeatures reads.
const timed = (s, t) => ({ inputNames: s.inputNames, outputNames: s.outputNames,
  run: async (feeds) => { const t0 = performance.now(); const r = await s.run(feeds); t.run_ms = performance.now() - t0; return r; } });

async function run({ start = 0, end = 800, redo = false, localRuntime = true }) {
  await setup(localRuntime);
  const { clips, done } = await (await fetch('/dgpu/clips')).json();
  const skip = new Set(redo ? [] : done);
  const todo = clips.map((cid, i) => ({ cid, i })).slice(start, end).filter((c) => !skip.has(c.cid));
  const st = { start, end, todo: todo.length, done: 0, skipped: end - start - todo.length, last: null, min_cos: 1, max_abs: 0,
    zero_mismatch: 0, ms: { webgpu: [], wasm: [] }, t0: Date.now() };
  post({ type: 'status', run: { ...st, ms: undefined, elapsed_s: 0 } });
  let next = todo.length ? fetchClip(todo[0].cid) : null;
  for (let k = 0; k < todo.length; k++) {
    const clip = await next;
    next = k + 1 < todo.length ? fetchClip(todo[k + 1].cid) : null;   // one clip prefetched, never more
    const inputs = [clip.frames, clip.kps];
    if (!warmed) {                                          // WebGPU compiles its shaders on the first run
      for (const ep of EPS) await rgbFeatures(...inputs, { ort, session: sessions[ep] });
      warmed = true;
    }
    const order = k % 2 ? ['wasm', 'webgpu'] : ['webgpu', 'wasm'], out = {}, t = { webgpu: {}, wasm: {} };
    for (const ep of order) {
      const t0 = performance.now();
      out[ep] = await rgbFeatures(...inputs, { ort, session: timed(sessions[ep], t[ep]) });
      t[ep].total_ms = performance.now() - t0;
    }
    const cmp = compare(out.webgpu, out.wasm);
    await upload(`${clip.cid}.wasm.f32`, out.wasm);
    await upload(`${clip.cid}.gpu.f32`, out.webgpu);
    await upload(`${clip.cid}.json`, JSON.stringify({ clip: clip.cid, manifest_index: todo[k].i, setup: setupInfo.id,
      providers: { webgpu: setupInfo.created.webgpu.provider, wasm: setupInfo.created.wasm.provider },
      n: clip.n, T: clip.T, video_frames: clip.nVideo, kps_frames: clip.nKps, idx: clip.idx, width: clip.W, height: clip.H,
      order, timing: t, min_cos: cmp.min_cos, max_abs: cmp.max_abs, zero_mismatch: cmp.zero_mismatch,
      images: cmp.rows }));                                 // json last: its presence marks the clip complete
    st.done++; st.last = clip.cid; st.zero_mismatch += cmp.zero_mismatch;
    if (cmp.min_cos !== null) st.min_cos = Math.min(st.min_cos, cmp.min_cos);
    st.max_abs = Math.max(st.max_abs, cmp.max_abs);
    for (const ep of EPS) st.ms[ep].push(t[ep].run_ms);
    const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
    post({ type: 'status', run: { ...st, ms: undefined, median_run_ms: { webgpu: med(st.ms.webgpu), wasm: med(st.ms.wasm) },
      elapsed_s: (Date.now() - st.t0) / 1000 } });
  }
  post({ type: 'done', start, end, done: st.done, skipped: st.skipped });
}

onmessage = async (e) => {
  try {
    if (e.data.type === 'run') await run(e.data);
  } catch (err) { post({ type: 'error', error: String(err?.stack ?? err) }); }
};
