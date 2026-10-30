// Tracker v2, ported exactly from results/tracker_v2/tracker_v2.py (whole clip at once: the studio is utterance-final).
// Pure functions, no DOM. Parity: results/tracker_v2/port/eval_port.mjs (every frame of the 60 ground-truth clips).
//
// Input: frames[t] = {n, kp: Float32Array(n*133*2) (x, y), sc: Float32Array(n*133), box: Float32Array(n*4) xyxy},
// or undefined / n = 0 when nothing was detected; W, H = frame size. Output: Int16Array chosen per frame, -1 = masked.
//
// Numerics. The reference runs on float32 arrays under numpy 2.x (NEP 50: a Python float meeting a float32 is cast to
// float32), so box arithmetic, IoU, the 1 - IoU link test, areas, centrality and hand motion are float32 there, and
// Math.fround reproduces them here. A few values leave float32: builtin max/min returning the Python literal (0.0,
// 1.0), and np.mean over a list holding any Python float is then a float64 mean. Those cases carry a `py` flag so
// the next operation takes the same precision numpy would. Assignment costs are summed in float64
// (np.asarray(cost, float)). Measured 2026-09-30: with fround removed (plain float64) the port still matches all
// 4,104 frames x 2 weight sets of the 60 ground-truth clips and the random fixture, but fails a keypoint score of
// exactly float32(0.3) (tests/engine_tracklets.test.mjs), which numpy does not count; so the emulation stays.

const f = Math.fround;
const K = 133;
export const CONF = 0.3, MIN_KP = 5, DUP_IOU = 0.6, LINK_IOU = 0.3, MAX_GAP = 5, EXT_IOU = 0.3;
export const DEFAULT = Object.freeze({ w_cov: 1.0, w_centre: 1.0, w_area: 1.0, w_hands: 1.0 });
export const TUNED = Object.freeze({ w_cov: 0.0, w_centre: 0.0, w_area: 1.0, w_hands: 0.0 });  // box area alone
const F_CONF = f(CONF), F_DUP = f(DUP_IOU), F_LINK = f(LINK_IOU), F_EXT = f(EXT_IOU);

// numpy pairwise_sum (8 accumulators, blocks of 128); r = Math.fround for float32 arrays, identity for float64.
function pairwiseSum(a, off, n, r) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = r(res + a[off + i]);
    return res;
  }
  if (n <= 128) {
    const acc = [];
    for (let j = 0; j < 8; j++) acc.push(a[off + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) acc[j] = r(acc[j] + a[off + i + j]);
    let res = r(r(r(acc[0] + acc[1]) + r(acc[2] + acc[3])) + r(r(acc[4] + acc[5]) + r(acc[6] + acc[7])));
    for (; i < n; i++) res = r(res + a[off + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return r(pairwiseSum(a, off, n2, r) + pairwiseSum(a, off + n2, n - n2, r));
}

// np.mean over a list of {v, py}: float32 array unless any element is a Python float (then float64).
function listMean(xs) {
  const py = xs.some((x) => x.py), v = xs.map((x) => x.v);
  return py ? pairwiseSum(v, 0, v.length, (x) => x) / v.length : f(pairwiseSum(v, 0, v.length, f) / v.length);
}

/** iou(a, b) on float32 boxes a = A[ai*4..], b = B[bi*4..]. Returns {v, py}; py = the Python 0.0 (degenerate union). */
export function iou(A, ai, B, bi) {
  const a = ai * 4, b = bi * 4;
  const x0 = Math.max(A[a], B[b]), y0 = Math.max(A[a + 1], B[b + 1]);
  const x1 = Math.min(A[a + 2], B[b + 2]), y1 = Math.min(A[a + 3], B[b + 3]);
  const dx = f(x1 - x0), dy = f(y1 - y0);
  const inter = dx > 0 && dy > 0 ? f(dx * dy) : 0;
  const ua = f(f(f(f(A[a + 2] - A[a]) * f(A[a + 3] - A[a + 1])) + f(f(B[b + 2] - B[b]) * f(B[b + 3] - B[b + 1]))) - inter);
  return ua > 0 ? { v: f(inter / ua), py: false } : { v: 0, py: true };
}

const lt = (x, thr, fthr) => (x.py ? x.v < thr : x.v < fthr);
const ge = (x, thr, fthr) => (x.py ? x.v >= thr : x.v >= fthr);

/** Indices of usable detections in one frame, duplicates of one person merged. */
export function candidates(fr) {
  const n = fr ? fr.n : 0, ok = [], mean = [];
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let k = 0; k < 23; k++) if (fr.sc[i * K + k] > F_CONF) c++;
    if (c >= MIN_KP) { ok.push(i); mean[i] = f(pairwiseSum(fr.sc, i * K, K, f) / K); }
  }
  ok.sort((i, j) => mean[j] - mean[i]);            // stable, like list.sort(key=-mean)
  const keep = [];
  for (const i of ok) if (keep.every((j) => lt(iou(fr.box, i, fr.box, j), DUP_IOU, F_DUP))) keep.push(i);
  return keep;
}

/** scipy.optimize.linear_sum_assignment by brute force, with tracker_v2.py's tie-break: the permutation minimising
 *  (sum of costs in index order, permutation tuple), i.e. the first minimum in lexicographic order. cost: rows of
 *  float64 numbers. Returns [rows, cols], rows ascending. */
export function linearSumAssignment(cost) {
  const r = cost.length, c = cost[0].length, tr = r > c;
  const m = tr ? c : r, pool = tr ? r : c, at = tr ? (j, p) => cost[p][j] : (i, p) => cost[i][p];
  let best = null, bestSum = Infinity;
  const p = [], used = new Array(pool).fill(false);
  (function dfs(s) {
    if (p.length === m) { if (best === null || s < bestSum) { bestSum = s; best = p.slice(); } return; }
    for (let q = 0; q < pool; q++) {
      if (used[q]) continue;
      used[q] = true; p.push(q);
      dfs(s + at(p.length - 1, q));
      p.pop(); used[q] = false;
    }
  })(0);
  if (!tr) return [best.map((_, i) => i), best];
  const o = best.map((row, j) => [row, j]).sort((x, y) => x[0] - y[0]);
  return [o.map((x) => x[0]), o.map((x) => x[1])];
}

/** Link per-frame candidates into tracklets. Each tracklet: {ts: [frames ascending], ds: [detection index]}. */
export function tracklets(frames, cands) {
  const tracks = [];
  let live = [];
  const last = (k) => tracks[k].ts.length - 1;
  cands.forEach((cs, t) => {
    live = live.filter((k) => t - tracks[k].ts[last(k)] <= MAX_GAP + 1);
    let fresh;
    if (live.length && cs.length) {
      const ious = live.map((k) => cs.map((c) => iou(frames[tracks[k].ts[last(k)]].box, tracks[k].ds[last(k)], frames[t].box, c)));
      const f64 = ious.some((row) => row.some((x) => x.py));          // np.array(...) is float64 if any Python float
      const cost = ious.map((row) => row.map((x) => (x.py ? 1 - x.v : f(1 - x.v))));
      const [rr, cc] = linearSumAssignment(cost);
      const used = new Set();
      rr.forEach((i, q) => {
        const j = cc[q];
        if (f64 ? 1 - cost[i][j] >= LINK_IOU : f(1 - cost[i][j]) >= F_LINK) {
          tracks[live[i]].ts.push(t); tracks[live[i]].ds.push(cs[j]); used.add(j);
        }
      });
      fresh = cs.filter((_, j) => !used.has(j));
    } else fresh = cs.slice();
    for (const d of fresh) { tracks.push({ ts: [t], ds: [d] }); live.push(tracks.length - 1); }
  });
  return tracks;
}

/** Per-tracklet evidence that it is the signer (coverage, centrality, box area, hand motion), as in features(). */
export function features(track, frames, W, H, T) {
  const cen = [], area = [], mot = [];
  const fWH = f(W * H), fW2 = f(W / 2.0);
  let prev = null;
  track.ts.forEach((t, q) => {
    const d = track.ds[q], fr = frames[t], b = d * 4, B = fr.box;
    const hgt = f(B[b + 3] - B[b + 1]), bhPy = !(hgt > 1.0), bh = bhPy ? 1.0 : hgt;
    area.push({ v: f(f(f(B[b + 2] - B[b]) * bh) / fWH), py: false });
    const rel = f(Math.abs(f(f(f(B[b] + B[b + 2]) / 2) - fW2)) / fW2);
    cen.push(rel < 1.0 ? { v: f(1 - rel), py: false } : { v: 0.0, py: true });
    const hs = [];
    for (let j = 91; j < 133; j++) hs.push(fr.sc[d * K + j] > F_CONF);
    if (prev !== null) {
      const norms = [];
      for (let j = 91; j < 133; j++) {
        if (!(hs[j - 91] && prev.hs[j - 91])) continue;
        const o = (d * K + j) * 2, po = (prev.d * K + j) * 2;
        const dx = f(fr.kp[o] - prev.kp[po]), dy = f(fr.kp[o + 1] - prev.kp[po + 1]);
        norms.push(f(Math.sqrt(f(f(dx * dx) + f(dy * dy)))));
      }
      if (norms.length) {
        const v = 10.0 * f(pairwiseSum(norms, 0, norms.length, f) / norms.length);
        const x = bhPy ? { v: v / bh, py: true } : { v: f(f(v) / bh), py: false };
        mot.push(x.v < 1.0 ? x : { v: 1.0, py: true });
      }
    }
    prev = { hs, d, kp: fr.kp };
  });
  return { cov: track.ts.length / T, centre: listMean(cen), area: listMean(area), hands: mot.length ? listMean(mot) : 0.0 };
}

export function score(track, frames, W, H, T, w = DEFAULT) {
  const x = features(track, frames, W, H, T);
  return w.w_cov * x.cov + w.w_centre * x.centre + w.w_area * x.area + w.w_hands * x.hands;
}

/** The signer's detection index per frame (-1 = masked), as tracker_v2.choose(d, w)[1]. */
export function choose(frames, W, H, w = DEFAULT) {
  const T = frames.length, chosen = new Int16Array(T).fill(-1);
  const tr = tracklets(frames, frames.map(candidates));
  if (!tr.length) return chosen;
  const S = tr.map((k) => score(k, frames, W, H, T, w));
  let best = 0;
  for (let k = 1; k < tr.length; k++) if (S[k] > S[best]) best = k;
  const path = new Int16Array(T).fill(-1);
  tr[best].ts.forEach((t, q) => { path[t] = tr[best].ds[q]; });
  const others = tr.map((_, k) => k).filter((k) => k !== best).sort((x, y) => (S[y] < S[x] ? -1 : S[y] > S[x] ? 1 : 0));
  for (const k of others) {
    const { ts, ds } = tr[k];
    const shared = ts.filter((t) => path[t] >= 0);
    const fresh = ts.map((t, q) => [t, ds[q]]).filter(([t]) => path[t] < 0);
    if (!fresh.length) continue;
    let same;
    if (shared.length) {
      const m = listMean(shared.map((t) => iou(frames[t].box, path[t], frames[t].box, ds[ts.indexOf(t)])));
      same = m >= EXT_IOU;                            // float(np.mean(...)) >= 0.3 in Python floats
    } else {
      const t0 = fresh[0][0], t1 = fresh[fresh.length - 1][0], links = [];
      let tb = -1, ta = -1;
      for (let t = 0; t < T; t++) if (path[t] >= 0) { if (t < t0) tb = t; if (t > t1 && ta < 0) ta = t; }
      if (tb >= 0) links.push(iou(frames[tb].box, path[tb], frames[t0].box, fresh[0][1]));
      if (ta >= 0) links.push(iou(frames[ta].box, path[ta], frames[t1].box, fresh[fresh.length - 1][1]));
      let mx = links[0];
      for (const l of links.slice(1)) if (l.v > mx.v) mx = l;     // builtin max: first maximum
      same = links.length > 0 && ge(mx, EXT_IOU, F_EXT);
    }
    if (same) for (const [t, d] of fresh) path[t] = d;
  }
  return path;
}
