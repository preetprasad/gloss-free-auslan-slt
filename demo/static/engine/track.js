// Signer tracking and 25 fps resampling, ported verbatim from demo/service.py (_torso, seed_index, track_index,
// grid_indices). float32 arithmetic is reproduced where numpy does it in float32 (Math.fround), float64 elsewhere.
// Detections: kp Float32Array(n*133*2) (x, y), sc Float32Array(n*133).

import { fma } from './cv.js';

export const CONF = 0.3;        // keypoint confidence floor
export const GATE = 1.6;        // re-seed beyond this many shoulder widths
export const L_SH = 5, R_SH = 6;
export const FRAME_MS = 40.0;   // 25 fps training grid
const K = 133;
const f = Math.fround;

// np.float32 1-D sum as numpy computes it (pairwise_sum: 8 accumulators, blocks of 128).
function pairwiseSum(a, off, n) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f(res + a[off + i]);
    return res;
  }
  if (n <= 128) {
    const r = Array.from({ length: 8 }, (_, j) => a[off + j]);
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

/** sc[i].mean() for a float32 score row. */
export function scoreMean(sc, i) {
  return f(pairwiseSum(sc, i * K, K) / K);
}

/** Shoulder midpoint (float32 pair), shoulder width (double of a float32 norm) and validity, for detection i. */
export function torso(kp, sc, i) {
  const s = i * K, k = i * K * 2;
  if (sc[s + L_SH] < CONF || sc[s + R_SH] < CONF) return { mid: null, wid: null, ok: false };
  const lx = kp[k + 2 * L_SH], ly = kp[k + 2 * L_SH + 1], rx = kp[k + 2 * R_SH], ry = kp[k + 2 * R_SH + 1];
  const mid = [f(lx + rx) / 2, f(ly + ry) / 2];
  const wid = norm2(f(lx - rx), f(ly - ry));
  return { mid, wid: wid > 1e-3 ? wid : null, ok: wid > 1e-3 };
}

// np.linalg.norm of a float32 2-vector: sqrt(x.dot(x)) in float32.
function norm2(dx, dy) {
  return f(Math.sqrt(f(f(dx * dx) + f(dy * dy))));
}

/** Frame-0 rule: the largest confident torso, tie-broken toward the centre of the frame. */
export function seedIndex(kp, sc, n, [W, H]) {
  let best = 0, bestScore = -1e18;
  for (let i = 0; i < n; i++) {
    const { mid, wid, ok } = torso(kp, sc, i);
    if (!ok) continue;
    const centrality = 1.0 - Math.abs(mid[0] - W / 2.0) / (W / 2.0);
    const score = wid * (1.0 + 0.35 * centrality) + 0.02 * scoreMean(sc, i);
    if (score > bestScore) { best = i; bestScore = score; }
  }
  return best;
}

/** Frame-t rule: nearest confident torso to the previous accepted one, else re-seed. Returns [index, reseeded]. */
export function trackIndex(kp, sc, n, prevMid, prevWid, frameWH) {
  if (prevMid == null) return [seedIndex(kp, sc, n, frameWH), true];
  let best = null, bestD = 1e18;
  for (let i = 0; i < n; i++) {
    const { mid, ok } = torso(kp, sc, i);
    if (!ok) continue;
    const d = norm2(f(mid[0] - prevMid[0]), f(mid[1] - prevMid[1])) / Math.max(prevWid, 1e-3);
    if (d < bestD) { best = i; bestD = d; }
  }
  if (best === null) return [seedIndex(kp, sc, n, frameWH), true];
  if (bestD > GATE) return [seedIndex(kp, sc, n, frameWH), true];   // track lost: a cut, or the signer left frame
  return [best, false];
}

/** Nearest captured frame on the 25 fps grid (np.arange + argmin, first index on ties). */
export function gridIndices(times) {
  const t0 = times[0], stop = times[times.length - 1] + 1e-6;
  const len = Math.max(0, Math.ceil((stop - t0) / FRAME_MS));
  const second = t0 + FRAME_MS, delta = second - t0;
  const out = new Array(len);
  for (let g = 0; g < len; g++) {
    // numpy's DOUBLE_fill: buffer[i] = start + i*delta, which clang contracts into one FMA (1 ulp apart at fractional
    // t0, enough to flip a half-way frame: engine_track.test.mjs, t0 = 237.7).
    const grid = g === 0 ? t0 : g === 1 ? second : fma(g, delta, t0);
    let best = 0, bestD = Infinity;
    for (let j = 0; j < times.length; j++) {
      const d = Math.abs(times[j] - grid);
      if (d < bestD) { best = j; bestD = d; }
    }
    out[g] = best;
  }
  return out;
}
