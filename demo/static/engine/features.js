// Translator input features: src/slt/data.py::normalize_clip for the live configuration only, and the
// demo/service.py::ModelService.translate preamble (25 fps grid, stack, signer-visibility check).
// numpy 1.26 float32 arithmetic is reproduced exactly: pairwise sums, float32 mean/var/sqrt, Math.fround per op.
// Parity: demo/tests/engine_features.test.mjs (fixtures from demo/tests/fixtures/make_feature_fixtures.py).
import { sampleIndices } from './rgb.js';   // np.linspace(0, n - 1, num).round() (half to even), shared
import { gridIndices } from './track.js';   // service.grid_indices, parity-tested in engine_track.test.mjs

export { gridIndices };

const f = Math.fround;
const K = 133, F = 3 * K, XY = 2 * K;
const EPS = f(1e-6);   // numpy casts the Python float 1e-6 to float32 against float32 arrays (compare and add)

// The one configuration ported. normalize_clip branches outside it (bbox norm, conf threshold, hand-local, velocity,
// temporal pooling, subsets) are not implemented, so any other manifest must fail here rather than decode wrongly.
export const LIVE = Object.freeze({
  coord_norm: 'frame', num_keypoints: 133, keypoint_subset: 'all', use_confidence: true, use_velocity: false,
  conf_threshold: 0, max_src_frames: 256, temporal_pool: 1, hand_local_norm: false, drop_zero_frames: true,
  feature_dim: 399,
});

export function assertLive(pre) {
  for (const [k, v] of Object.entries(LIVE)) {
    if (pre[k] !== v) throw new Error(`features.js implements normalize_clip for ${k}=${v} only; manifest has ${pre[k]}`);
  }
}

// np.add.reduce of a contiguous float32 row (numpy pairwise_sum: 8 accumulators, blocks of 128). Same as track.js.
function pairwiseSum(a, off, n) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f(res + a[off + i]);
    return res;
  }
  if (n <= 128) {
    const r = [a[off], a[off + 1], a[off + 2], a[off + 3], a[off + 4], a[off + 5], a[off + 6], a[off + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = f(r[j] + a[off + i + j]);
    let res = f(f(f(r[0] + r[1]) + f(r[2] + r[3])) + f(f(r[4] + r[5]) + f(r[6] + r[7])));
    for (; i < n; i++) res = f(res + a[off + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f(pairwiseSum(a, off, n2) + pairwiseSum(a, off + n2, n - n2));
}

// np.abs(frame).sum() > 1e-6 over one frame's 133 x 3 values (drop_zero_frames; also the service's visibility test).
const absRow = new Float32Array(F);
function frameHasSignal(kps, t) {
  for (let j = 0, o = t * F; j < F; j++) absRow[j] = Math.abs(kps[o + j]);
  return pairwiseSum(absRow, 0, F) > EPS;
}

/**
 * normalize_clip(kps, zeros BBOX, live cfg). kps: Float32Array(T * 133 * 3), per keypoint (x, y, score).
 * Returns { feats: Float32Array(T' * 399), T: T' }; per keypoint (x_n, y_n, score). Throws if no frame has signal.
 */
export function normalizeClip(kps, T, maxFrames = LIVE.max_src_frames) {
  if (kps.length !== T * F) throw new Error(`normalizeClip: expected ${T * F} values, got ${kps.length}`);
  const rows = [], xy = new Float32Array(XY), sq = new Float32Array(XY);
  for (let t = 0; t < T; t++) {
    if (!frameHasSignal(kps, t)) continue;                       // feats[frame_ok]
    const o = t * F;
    for (let k = 0; k < K; k++) { xy[2 * k] = kps[o + 3 * k]; xy[2 * k + 1] = kps[o + 3 * k + 1]; }
    // coord_norm 'frame': m = mean, s = std (ddof 0) + 1e-6, all float32 as numpy computes them
    const m = f(pairwiseSum(xy, 0, XY) / XY);
    for (let j = 0; j < XY; j++) { const d = f(xy[j] - m); sq[j] = f(d * d); }
    const s = f(f(Math.sqrt(f(pairwiseSum(sq, 0, XY) / XY))) + EPS);
    const row = new Float32Array(F);                            // stores round to float32: (x - m) / s in float32
    for (let k = 0; k < K; k++) {
      row[3 * k] = f(xy[2 * k] - m) / s;
      row[3 * k + 1] = f(xy[2 * k + 1] - m) / s;
      row[3 * k + 2] = kps[o + 3 * k + 2];
    }
    rows.push(row);
  }
  if (!rows.length) throw new Error('normalize_clip: all frames invalid, so this clip carries no input.');
  const keep = rows.length > maxFrames ? sampleIndices(rows.length, maxFrames) : rows.map((_, i) => i);
  const feats = new Float32Array(keep.length * F);
  keep.forEach((r, i) => feats.set(rows[r], i * F));
  return { feats, T: keep.length };
}

/**
 * ModelService.translate up to the translator: nearest captured frame on the 25 fps grid, the < 80% visibility
 * refusal, then normalize_clip. keypoints: one Float32Array(399) per captured frame (zeros when no signer was found);
 * times: captured_at in ms. Returns { idx, kps, feats, T }; kps is the stacked grid clip (the RGB stage samples it).
 */
export function prepareClip(keypoints, times) {
  const idx = gridIndices(times);
  const kps = new Float32Array(idx.length * F);
  idx.forEach((i, t) => kps.set(keypoints[i], t * F));
  let visible = 0;
  for (let t = 0; t < idx.length; t++) visible += frameHasSignal(kps, t);
  if (visible / idx.length < 0.8) throw new Error('Keep one signer in view for the whole utterance, then try again.');
  return { idx, kps, ...normalizeClip(kps, idx.length) };
}
