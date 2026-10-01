// Signing-activity score of one tracked person over one utterance: the 8 kinematic features of
// results/signing_activity/signing_activity.py::features (KIN) and the frozen logistic model in activity_model.js.
// float64 throughout (the Python reference is run on float64 copies in the parity fixtures); the score only ranks the
// people in view, so parity is to ~1e-9, not bit-exact. A missing frame is an all-zero row: its scores are 0, so its
// shoulders are not confident and every feature that needs them is NaN for that frame.
// Parity: demo/tests/engine_activity.test.mjs (fixtures from demo/tests/fixtures/make_activity_fixtures.py).
import { ACTIVITY_MODEL } from './activity_model.js';

export const CONF = 0.3;          // keypoint confidence floor (nonsigner_gate.CONF)
export const FPS = 25;            // the 25 fps training grid
export const RAISED = 0.8;        // a wrist counts as raised above (shoulder line - 0.8 shoulder widths)
export const FEATURES = ACTIVITY_MODEL.features;
const K = 133, F = 3 * K;
const LEFT = { root: 91, first: 92 }, RIGHT = { root: 112, first: 113 }, NFINGERS = 20;

const finite = Number.isFinite;

/** np.nanmedian of an array with NaN holes (NaN when nothing is finite). */
export function nanmedian(a) {
  const v = a.filter(finite).sort((p, q) => p - q), n = v.length;
  return n === 0 ? NaN : n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/** np.nanpercentile(a, q), linear interpolation (NaN when nothing is finite). */
export function nanpercentile(a, q) {
  const v = a.filter(finite).sort((p, r) => p - r), n = v.length;
  if (n === 0) return NaN;
  const pos = (n - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

const fmax = (a, b) => (finite(a) ? (finite(b) ? Math.max(a, b) : a) : b);   // np.fmax: a NaN operand is ignored
const fmin = (a, b) => (finite(a) ? (finite(b) ? Math.min(a, b) : a) : b);

/**
 * The 8 kinematic features of one person's clip. kps: Float32Array(T * 133 * 3), per keypoint (x, y, score), on the
 * 25 fps grid. Speeds are in shoulder widths per second; heights and distances in shoulder widths.
 */
export function kinematics(kps, T) {
  if (kps.length !== T * F) throw new Error(`kinematics: expected ${T * F} values, got ${kps.length}`);
  const px = (t, j) => kps[t * F + 3 * j], py = (t, j) => kps[t * F + 3 * j + 1], pc = (t, j) => kps[t * F + 3 * j + 2];
  const ok = new Array(T), mid = new Array(T), ws = new Array(T).fill(NaN);
  for (let t = 0; t < T; t++) {
    const w = Math.hypot(px(t, 5) - px(t, 6), py(t, 5) - py(t, 6));
    ok[t] = pc(t, 5) >= CONF && pc(t, 6) >= CONF && w > 1e-3;
    mid[t] = [(px(t, 5) + px(t, 6)) / 2, (py(t, 5) + py(t, 6)) / 2];
    if (ok[t]) ws[t] = w;
  }
  // z(t, j): keypoint j centred on the shoulders, in shoulder widths (NaN where the frame's shoulders are not confident)
  const z = (t, j) => (ok[t] ? [(px(t, j) - mid[t][0]) / ws[t], (py(t, j) - mid[t][1]) / ws[t]] : [NaN, NaN]);
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const step = (t) => t >= 1 && ok[t] && ok[t - 1];              // both frames of the step have confident shoulders
  const conf = (t, j) => pc(t, j) >= CONF;

  const speed = (j, seen) => {                                    // per-frame joint speed; NaN unless both frames qualify
    const out = new Array(T).fill(NaN);
    for (let t = 1; t < T; t++) if (step(t) && seen(t) && seen(t - 1)) out[t] = dist(z(t, j), z(t - 1, j)) * FPS;
    return out;
  };
  const s9 = speed(9, (t) => conf(t, 9)), s10 = speed(10, (t) => conf(t, 10));
  const wristSpeed = s9.map((a, t) => fmax(a, s10[t]));

  const fingerMotion = ({ root, first }) => {                     // mean speed of the 20 fingers relative to the hand root
    const seen = (t) => { let s = 0; for (let i = 0; i < NFINGERS; i++) s += pc(t, first + i); return s / NFINGERS >= CONF; };
    const rel = (t, i) => { const a = z(t, first + i), r = z(t, root); return [a[0] - r[0], a[1] - r[1]]; };
    const out = new Array(T).fill(NaN);
    for (let t = 1; t < T; t++) {
      if (!(step(t) && seen(t) && seen(t - 1))) continue;
      let s = 0;
      for (let i = 0; i < NFINGERS; i++) s += dist(rel(t, i), rel(t - 1, i));
      out[t] = s / NFINGERS * FPS;
    }
    return out;
  };
  const fl = fingerMotion(LEFT), fr = fingerMotion(RIGHT);
  const finger = fl.map((a, t) => fmax(a, fr[t]));

  const height = new Array(T).fill(NaN), face = new Array(T).fill(NaN);   // + = above the shoulder line
  for (let t = 0; t < T; t++) {
    if (!ok[t]) continue;
    let h = NaN, d = NaN;
    for (const j of [9, 10]) {
      if (!conf(t, j)) continue;
      const zj = z(t, j);
      h = fmax(h, -zj[1]);
      if (conf(t, 0)) d = fmin(d, dist(zj, z(t, 0)));
    }
    height[t] = h; face[t] = d;
  }
  let raised = 0, nOk = 0;
  for (let t = 0; t < T; t++) {
    if (!ok[t]) continue;
    nOk++;
    if ((finite(height[t]) ? height[t] : -9) > -RAISED) raised++;
  }
  const body = new Array(T).fill(NaN);
  for (let t = 1; t < T; t++) if (step(t)) body[t] = dist(mid[t], mid[t - 1]) / ws[t] * FPS;
  const head = speed(0, (t) => conf(t, 0));

  return {
    wrist_speed_med: nanmedian(wristSpeed), wrist_speed_p90: nanpercentile(wristSpeed, 90),
    finger_motion_med: nanmedian(finger), hand_height_med: nanmedian(height),
    raised_share: nOk ? raised / nOk : NaN, hand_face_dist_med: nanmedian(face),
    body_motion_med: nanmedian(body), head_motion_med: nanmedian(head),
  };
}

/** The frozen model: intercept + sum(coef * (impute(x) - mean) / scale); a NaN feature takes the fitted median. */
export function scoreFeatures(feat, m = ACTIVITY_MODEL) {
  let s = m.intercept;
  m.features.forEach((name, i) => {
    const v = finite(feat[name]) ? feat[name] : m.median[i];
    s += m.coef[i] * (v - m.mean[i]) / m.scale[i];
  });
  return s;
}

/** Higher = moves more like a signer, relative to the other people in the same view. */
export function activityScore(kps, T) {
  const features = kinematics(kps, T);
  return { score: scoreFeatures(features), features };
}
