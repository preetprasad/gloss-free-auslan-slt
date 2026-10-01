// Gate C: run demo/static/engine/tracklets_activity.js (v2 tracklets + activity choice) on the stored all-detections of
// the 60 ground-truth clips and write, per clip, the detection index chosen on every frame (-1 = masked).
//   node eval_variant.mjs FLAT_BOX_DIR OUT.json          (FLAT_BOX_DIR from export_boxes.py)
// Also checks the path grouping against the parity-tested choose(): seeding personPaths in v2 TUNED score order must
// reproduce choose(TUNED) on every frame (exits non-zero otherwise). No ground-truth label is read here.
import { readFileSync, writeFileSync } from 'node:fs';
import { candidates, tracklets, score, choose, TUNED } from '../../../demo/static/engine/tracklets.js';
import { chooseByActivity, personPaths } from '../../../demo/static/engine/tracklets_activity.js';

const [dir, outPath] = process.argv.slice(2);
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

function loadClip(c) {           // as eval_port.mjs loadClip
  const m = JSON.parse(readFileSync(`${dir}/${c}.json`));
  const kp = f32(`${dir}/${c}.kp.f32`), sc = f32(`${dir}/${c}.sc.f32`), box = f32(`${dir}/${c}.box.f32`);
  const { T, N } = m, K = 133, frames = [];
  for (let t = 0; t < T; t++) {
    const n = m.NDET[t];
    frames.push(n ? { n, kp: kp.subarray(t * N * K * 2, (t * N + n) * K * 2), sc: sc.subarray(t * N * K, (t * N + n) * K),
      box: box.subarray(t * N * 4, (t * N + n) * 4) } : undefined);
  }
  return { frames, W: m.W, H: m.H };
}

const parts = JSON.parse(readFileSync(new URL('../gt_clips.json', import.meta.url)));
const res = {}, by = {};
let groupMismatch = 0;
for (const part of Object.values(parts)) for (const [c] of part) {
  const { frames, W, H } = loadClip(c);
  const tr = tracklets(frames, frames.map(candidates));
  if (tr.length) {
    const S = tr.map((k) => score(k, frames, W, H, frames.length, TUNED));
    const order = tr.map((_, k) => k).sort((a, b) => S[b] - S[a] || a - b);
    const ref = choose(frames, W, H, TUNED), got = personPaths(frames, tr, order)[0].path;
    for (let t = 0; t < ref.length; t++) if (ref[t] !== got[t]) groupMismatch++;
  }
  const r = chooseByActivity(frames);
  res[c] = { chosen: Array.from(r.chosen), by: r.by, people: r.people };
  by[r.by] = (by[r.by] ?? 0) + 1;
}
writeFileSync(outPath, JSON.stringify(res));
console.log(`${Object.keys(res).length} clips; choice made by ${JSON.stringify(by)}; grouping vs choose(TUNED) frame mismatches: ${groupMismatch}`);
process.exit(groupMismatch ? 1 : 0);
