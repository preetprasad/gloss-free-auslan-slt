// In-browser engine configuration: where model files and the pinned runtimes come from, per-stage execution
// providers, and the device limits. One place to change for deployment.

// Model artefacts (results/live_onnx/, manifest.json at the root), published at a PINNED Hugging Face commit, never a
// branch; engine.js refuses any other cross-origin base (loader.js checkModelBase). On localhost, demo/serve_static.py
// serves the local export instead, so parity runs need no network. Commit 54a4eebb = upload of 30 Sep 2026.
const LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(globalThis.location?.hostname);
export const MODEL_BASE = LOCAL ? '/models/'
  : 'https://huggingface.co/preetprasad/auslan-slt-live/resolve/54a4eebbae5d84edc674e95b878a317e1bd187fb/';

// sha256 of results/live_onnx/manifest.json, checked before the manifest is parsed: the manifest carries every other
// file's sha256, so this pin makes the whole download chain verifiable. Re-pin whenever the export changes
// (demo/tests/engine_loader.test.mjs fails until it matches the file).
export const MANIFEST_SHA256 = 'ccc3256e5113b204b99203a9ab09a499d5e3872b076fa64b39e5bee1b01e7f91';

// DINOv2 when manifest.dinov2 has no `file` (public onnx-community/dinov2-small at the manifest's pinned revision).
// On localhost, serve_static.py mirrors the local Hugging Face cache at /hf/.
export const HF_BASE = LOCAL ? '/hf/' : 'https://huggingface.co/';

const ORT_VERSION = '1.30.0';   // == demo/package.json onnxruntime-web / onnxruntime-node (the tested build)
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// jsDelivr files, byte-identical to the npm package in demo/node_modules (sha256 checked 2026-09-25).
// The WebGPU bundle carries the WebGPU and WASM execution providers; its WASM binary is the asyncify build.
export const RUNTIME = {
  ort: { url: `${ORT_DIST}ort.webgpu.bundle.min.mjs`, bytes: 117732,
    sha256: '0730816731ce2cdfb06f273da065d4a1c7091eb9d1756a83b2916c8a89d015e2' },
  ortWasm: { url: `${ORT_DIST}ort-wasm-simd-threaded.asyncify.wasm`, bytes: 26781914,
    sha256: '39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da' },
};

export const CACHE_NAME = 'auslan-models-v1';   // Cache Storage, entries keyed by sha256

// Execution providers per stage, tried in order, one at a time; the one that loaded is reported to the page.
// DINOv2 on WebGPU passed results/PREREG_DINOV2_WEBGPU_2026-09-30.md (Chrome, M3 Max): on the 800 validation clips, S6
// decoded from WebGPU features equals S6 from WASM features on 800/800 hypotheses (both 12.291, = the Python live
// recompute; min cosine 1 - 1.3e-10, max |diff| 1.3e-4 over 56,050 crops), and 10/10 studio end-to-end runs match the
// Python service on text and decision; session.run median 258 vs 5,214 ms per clip. The translator stays on WASM: G6
// shows WebGPU also keeps its parity (800/800 identical hypotheses for both models; p50 41.5 vs 65.2 ms per clip single,
// 287.3 vs 439.7 ms ensemble), but every studio end-to-end run so far used the translator on WASM.
export const EP = { pose: ['webgpu', 'wasm'], translator: ['wasm'], dinov2: ['webgpu', 'wasm'] };

// Device limits. Pose runs on every captured frame, so after Finish the visitor waits about
// (pose_ms / frame interval - 1) s per second signed: pose_ms / 40 - 1 with 25 fps capture, but app.js sends every
// camera frame at least 30 ms apart, so a 30 fps camera sends 30 frames/s and the wait is pose_ms / 33.3 - 1. The
// Python service runs ~200 ms/frame (~4 s per second signed at 25 fps); 800 ms is ~19 s per second signed at 25 fps
// (~23 s at 30 fps), past which the demo is not usable. The limit applies to the load-time probe (engine.js: detector
// + 2 RTMW crops on a 960x720 frame, after a warm-up), not to live frames. Single-thread WASM pose measured 2,224
// ms/frame (Node), so a device without WebGPU and without cross-origin isolation (no WASM threads) is refused at the
// `check` step, before any download.
export const POSE_MS_LIMIT = 800;
export const MIN_DEVICE_MEMORY_GB = 4;          // navigator.deviceMemory, where the browser reports it

// Service defaults mirrored from demo/app.py Settings: 50% answer rate, 300 camera frames (10 s at 30 fps).
export const COVERAGE = 0.5;
export const MAX_FRAMES = 300;

// Who is translated when two or more people are in view (persons.js): 'activity' = the person moving most like a signer
// (activity.js), 'online' = the person the single-signer tracker already follows (labelled, switchable). The pre-registered rule
// in results/person_labels/PREREG_PERSON_LABELS_2026-09-30.md was met on DEV and HELD-OUT (results/person_labels/outcome/):
// activity is no worse than the shipped tracker on the ground-truth clips, which is all the rule asked; it is NOT shown to be better.
// With one person in view neither applies.
export const PERSON_CHOICE = 'activity';
