// Run the shipped roster + activity code (demo/static/engine/persons.js, activity.js) on stored all-detections clips and
// write, per clip, the detection index each rule follows on every frame. No image is involved.
//   node eval_roster.mjs EXPORT_DIR OUT.json [--clips gt_clips.json]      (EXPORT_DIR from export_alldet.py)
//   node eval_roster.mjs --selftest
import { readFileSync, writeFileSync } from 'node:fs';
import { Session } from '../../demo/static/engine/persons.js';

const K = 133, DT = 40;   // 25 fps, the dataset's frame rate

export function runClip({ T, N, W, NDET, CHOSEN }, kp, sc) {
  const u = new Session().utterance();
  u.trace = [];
  for (let t = 0; t < T; t++) {
    const n = NDET[t];
    u.step(n ? { kp: kp.subarray(t * N * K * 2, t * N * K * 2 + n * K * 2), sc: sc.subarray(t * N * K, t * N * K + n * K), n, width: W } : undefined, CHOSEN[t]);
  }
  const times = Array.from({ length: T }, (_, i) => i * DT);
  const out = { multi: false, people: 0 };
  const follow = (sel) => {                       // detection index of the chosen person on every frame, -1 where absent
    if (!sel.multi || sel.chosen === null) return null;
    return u.trace.map((pairs) => pairs.find(([, id]) => id === sel.chosen)?.[0] ?? -1);
  };
  for (const mode of ['activity', 'online']) {
    const sel = u.select(times, mode);
    out.multi = sel.multi; out.people = sel.people.length;
    out[mode] = follow(sel);                        // null: not multi / nobody eligible -> the engine keeps the shipped stream
    out[`${mode}_by`] = sel.by ?? null;             // 'online' may have fallen back to 'activity' (the tracker followed nobody eligible)
    out[`${mode}_people`] = sel.people.map((p) => ({ label: p.label, share: p.share, eligible: p.eligible, score: p.score, chosen: p.id === sel.chosen }));
  }
  return out;
}

function load(dir, clip) {
  const j = JSON.parse(readFileSync(`${dir}/${clip}.json`));
  const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
  return [j, f32(`${dir}/${clip}.kp.f32`), f32(`${dir}/${clip}.sc.f32`)];
}

function selftest() {
  // two people, the right one signing; the shipped tracker (CHOSEN) follows the left one
  const T = 50, N = 2, kp = new Float32Array(T * N * K * 2), sc = new Float32Array(T * N * K).fill(0.9);
  for (let t = 0; t < T; t++) for (const [i, x] of [[0, 250], [1, 650]]) {
    const o = (t * N + i) * K * 2, put = (j, px, py) => { kp[o + 2 * j] = px; kp[o + 2 * j + 1] = py; };
    for (let j = 0; j < K; j++) put(j, x, 400);
    put(5, x + 55, 300); put(6, x - 55, 300); put(0, x, 230); put(9, x + 44, 500);
    const wy = i === 1 ? 290 + 60 * Math.sin(t * 1.3) : 500, wx = x - 44 + (i === 1 ? 50 * Math.cos(t * 0.9) : 0);
    put(10, wx, wy); for (let f = 0; f < 21; f++) { put(112 + f, wx, wy + (i === 1 ? f * 0.3 * Math.sin(t + f) : 0)); put(91 + f, x + 44, 500); }
  }
  const r = runClip({ T, N, W: 960, NDET: Array(T).fill(2), CHOSEN: Array(T).fill(0) }, kp, sc);
  if (!(r.multi && r.people === 2)) throw new Error('two people expected');
  if (!r.activity.every((i) => i === 1)) throw new Error(`activity should follow detection 1: ${r.activity.slice(0, 5)}`);
  if (!r.online.every((i) => i === 0)) throw new Error(`online should follow detection 0: ${r.online.slice(0, 5)}`);
  console.log('selftest passed');
}

if (process.argv[1].endsWith('eval_roster.mjs')) {
  if (process.argv.includes('--selftest')) selftest();
  else {
    const [dir, outPath] = process.argv.slice(2);
    const ci = process.argv.indexOf('--clips');
    const parts = JSON.parse(readFileSync(ci > 0 ? process.argv[ci + 1] : new URL('../tracker_v2/gt_clips.json', import.meta.url)));
    const res = {};
    for (const part of Object.values(parts)) for (const [clip] of part) res[clip] = runClip(...load(dir, clip));
    writeFileSync(outPath, JSON.stringify(res));
    const multi = Object.values(res).filter((r) => r.multi).length;
    console.log(`${Object.keys(res).length} clips, ${multi} with two or more people counted`);
  }
}
