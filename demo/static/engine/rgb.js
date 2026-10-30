// DINOv2 crop features for the RGB-fusion ensemble members: a line-for-line port of demo/rgb_features.py
// (crop helpers from results/dwpose_experiment/extract_rgb_feats.py). Parity gate G5: demo/tests/engine_rgb.test.mjs.
import { resize, rgbFromRGBA } from './cv.js';
// Dependencies are passed in, so the same code runs in the browser worker and in Node:
//   ort     onnxruntime-web / onnxruntime-node (only ort.Tensor is used)
//   session InferenceSession for manifest.dinov2 (onnx-community/dinov2-small: last_hidden_state, token 0 is
//           transformers' pooler_output; a graph that outputs pooler_output [N,384] directly also works)

export const FRAMES = 24;                       // RGBFeatures(frames=24)
const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406].map(Math.fround), STD = [0.229, 0.224, 0.225].map(Math.fround);  // IMN, float32 tensors
const THR = Math.fround(0.3);                   // numpy compares the float32 confidences with 0.3 in float32
const SLOTS = [[0, 133], [91, 112], [112, 133]]; // signer, left hand, right hand

// numpy round(): half to even.
const rint = (x) => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };

// np.linspace(0, n - 1, min(frames, n)).round().astype(int)
export function sampleIndices(n, frames = FRAMES) {
  const num = Math.min(frames, n);
  if (num <= 1) return num === 1 ? [0] : [];
  const step = (n - 1) / (num - 1);
  return Array.from({ length: num }, (_, i) => rint(i === num - 1 ? n - 1 : i * step));
}

// crop_box(xy[a:b], cf[a:b]) on one frame's (133,3) float32 keypoints. Mixed precision as in numpy 1.26:
// min/max and the two sums/differences in float32, the rest in float64, int() truncates toward zero.
export function cropBox(k, a = 0, b = 133, thr = THR, pad = 1.4) {
  let n = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let j = a; j < b; j++) {
    if (!(k[3 * j + 2] > thr)) continue;
    const x = k[3 * j], y = k[3 * j + 1];
    n++; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  if (n < 3) return null;
  const cx = Math.fround(x0 + x1) / 2, cy = Math.fround(y0 + y1) / 2;
  const s = Math.max(Math.fround(x1 - x0), Math.fround(y1 - y0)) * pad / 2 + 1;
  const int = (v) => Math.trunc(v) + 0;  // Python int(): toward zero, and never -0
  return [int(cx - s), int(cy - s), int(cx + s), int(cy + s)];
}

// get_crop: clamp to the frame, slice, cv2.resize to 224x224 (INTER_LINEAR, cv.js). frame: RGB { data, width, height }.
// Returns the crop's HWC bytes, or null.
export function getCrop(frame, box) {
  if (box === null) return null;
  const x0 = Math.max(0, box[0]), y0 = Math.max(0, box[1]), x1 = Math.min(frame.width, box[2]), y1 = Math.min(frame.height, box[3]);
  if (x1 <= x0 || y1 <= y0) return null;
  const w = x1 - x0, h = y1 - y0, roi = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) roi.set(frame.data.subarray(((y0 + y) * frame.width + x0) * 3, ((y0 + y) * frame.width + x1) * 3), y * w * 3);
  return resize({ data: roi, width: w, height: h, channels: 3 }, SIZE, SIZE).data;
}

// RGBFeatures.__call__. frames: RGBA ImageData-like {data, width, height} for the sampled indices only (the browser's
// decoded JPEGs; RGBA->RGB here is the Python BGR->RGB); kps: Float32Array of the matching (n,133,3) keypoints.
// Returns Float32Array (n, 1152): per frame [signer | left hand | right hand] x 384, empty slots zero.
export async function rgbFeatures(frames, kps, { ort, session }) {
  const n = frames.length, CH = SIZE * SIZE, IMG = 3 * CH, zero = new Uint8Array(3 * CH);
  const x = new Float32Array(3 * n * IMG), slots = [];
  for (let f = 0; f < n; f++) {
    const rgb = rgbFromRGBA(frames[f]), k = kps.subarray(f * 399, (f + 1) * 399);
    for (const [a, b] of SLOTS) {
      const c = getCrop(rgb, cropBox(k, a, b)), px = c ?? zero, o = slots.length * IMG;
      slots.push(c !== null);
      // HWC uint8 -> NCHW float32: (u8 / 255 - mean) / std, each step rounded to float32 as numpy/torch do.
      for (let p = 0; p < CH; p++)
        for (let ch = 0; ch < 3; ch++) x[o + ch * CH + p] = Math.fround(Math.fround(px[3 * p + ch] / 255) - MEAN[ch]) / STD[ch];
    }
  }
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [3 * n, 3, SIZE, SIZE]) });
  const out = res[session.outputNames[0]], D = out.dims.at(-1), stride = out.dims.length === 3 ? out.dims[1] * D : D;
  const feats = new Float32Array(3 * n * D);
  slots.forEach((ok, i) => { if (ok) feats.set(out.data.subarray(i * stride, i * stride + D), i * D); });
  return feats;
}
