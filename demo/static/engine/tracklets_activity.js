// Gate C variant of tracker v2 for the live studio (results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md, last addendum).
// Keeps v2's tracklets (IoU linking, duplicate merge, continuations, masking) but chooses the signer by the studio's
// signing-activity score instead of box area: area picks the signer only because Auslan-Daily crops are built around
// the signer. Not wired into the engine. Scored by results/tracker_v2/port/eval_variant.mjs + score_variant.py.
//
// Rule (every constant is an existing one; none is tuned):
//  1. Person paths: tracklets are taken longest first (v2's coverage feature; ties by creation order). Each path is
//     seeded by the longest unused tracklet and grown with v2's continuation test (tracklets.js choose(): mean IoU with
//     the path on shared frames >= EXT_IOU, else IoU with the path's nearest frame before/after >= EXT_IOU). A tracklet
//     that passes the test joins the path even when it adds no frame (a duplicate of the same person), so it cannot
//     seed a second "person".
//  2. A path counts as a person if it has a detection on >= max(MIN_TRACK_FRAMES, MIN_TRACK_SHARE * T) frames, and is
//     eligible if it has a confident torso on >= MIN_VISIBLE of the 25 fps grid frames (persons.js constants).
//  3. Choice: the eligible counted path with the highest activityScore (activity.js). If no counted path is eligible:
//     exactly one counted path -> that path (one person in view; the translator's own refusal decides downstream);
//     otherwise (nobody counted, or two or more and none eligible) -> decline, every frame masked. Box area is never used.
//  4. Frames where the chosen path has no detection are masked (-1), as v2 does.
import { candidates, tracklets, iou, EXT_IOU } from './tracklets.js';
import { MIN_TRACK_FRAMES, MIN_TRACK_SHARE, MIN_VISIBLE } from './persons.js';
import { gridIndices, torso, FRAME_MS } from './track.js';
import { activityScore } from './activity.js';

const K = 133, F = 3 * K;

/** v2's continuation test: does tracklet k continue the person on `path` (Int16Array, -1 = none)? */
function continues(frames, path, k) {
  const shared = k.ts.map((t, q) => [t, k.ds[q]]).filter(([t]) => path[t] >= 0);
  if (shared.length) {
    const m = shared.reduce((s, [t, d]) => s + iou(frames[t].box, path[t], frames[t].box, d).v, 0) / shared.length;
    return m >= EXT_IOU;
  }
  const t0 = k.ts[0], t1 = k.ts[k.ts.length - 1], links = [];
  let tb = -1, ta = -1;
  for (let t = 0; t < path.length; t++) if (path[t] >= 0) { if (t < t0) tb = t; if (t > t1 && ta < 0) ta = t; }
  if (tb >= 0) links.push(iou(frames[tb].box, path[tb], frames[t0].box, k.ds[0]).v);
  if (ta >= 0) links.push(iou(frames[ta].box, path[ta], frames[t1].box, k.ds[k.ds.length - 1]).v);
  return links.length > 0 && Math.max(...links) >= EXT_IOU;
}

/** Group tracklets into person paths. order: tracklet indices, seeds first. Returns [{path: Int16Array(T), members}]. */
export function personPaths(frames, tr, order) {
  const T = frames.length, left = order.slice(), out = [];
  while (left.length) {
    const seed = left.shift(), path = new Int16Array(T).fill(-1), members = [seed];
    tr[seed].ts.forEach((t, q) => { path[t] = tr[seed].ds[q]; });
    for (let i = 0; i < left.length;) {
      const k = tr[left[i]];
      if (continues(frames, path, k)) {
        k.ts.forEach((t, q) => { if (path[t] < 0) path[t] = k.ds[q]; });
        members.push(left.splice(i, 1)[0]);
      } else i++;
    }
    out.push({ path, members });
  }
  return out;
}

/**
 * frames as tracklets.js choose(); times = capture time (ms) per frame (default 25 fps).
 * Returns {chosen: Int16Array(T) (-1 = masked), by: 'activity' | 'single' | 'decline', people: [{frames, share, eligible, score}]}.
 */
export function chooseByActivity(frames, times = frames.map((_, t) => t * FRAME_MS)) {
  const T = frames.length;
  const tr = tracklets(frames, frames.map(candidates));
  const order = tr.map((_, k) => k).sort((a, b) => tr[b].ts.length - tr[a].ts.length || a - b);
  const need = Math.max(MIN_TRACK_FRAMES, MIN_TRACK_SHARE * T), idx = gridIndices(times), G = idx.length;
  const people = [];
  for (const { path } of personPaths(frames, tr, order)) {
    const n = path.reduce((s, d) => s + (d >= 0), 0);
    if (n < need) continue;
    const grid = new Float32Array(G * F);
    let vis = 0;
    idx.forEach((t, g) => {
      const d = path[t];
      if (d < 0) return;
      const fr = frames[t];
      if (torso(fr.kp, fr.sc, d).ok) vis++;
      for (let j = 0; j < K; j++) {
        grid[g * F + 3 * j] = fr.kp[(d * K + j) * 2]; grid[g * F + 3 * j + 1] = fr.kp[(d * K + j) * 2 + 1];
        grid[g * F + 3 * j + 2] = fr.sc[d * K + j];
      }
    });
    const share = vis / G, eligible = share >= MIN_VISIBLE;
    people.push({ path, frames: n, share, eligible, score: eligible ? activityScore(grid, G).score : null });
  }
  const ok = people.filter((p) => p.eligible);
  let pick = null, by = 'decline';
  if (ok.length) { pick = ok.reduce((b, p) => (p.score > b.score ? p : b)); by = 'activity'; }
  else if (people.length === 1) { pick = people[0]; by = 'single'; }
  return { chosen: pick ? pick.path : new Int16Array(T).fill(-1), by, people: people.map(({ path, ...p }) => ({ ...p, chosen: path === pick?.path })) };
}
