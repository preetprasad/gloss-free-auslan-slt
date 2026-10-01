// Tracker v2 parity: demo/static/engine/tracklets.js vs results/tracker_v2/tracker_v2.py. The fixture holds the seed,
// a sha256 of each clip's float32 inputs and the Python choices; the inputs are regenerated here from the same 32-bit
// PRNG (results/tracker_v2/port/make_tracklets_fixture.py). Full parity on the 60 ground-truth clips is
// results/tracker_v2/port/eval_port.mjs (needs the local all-detections files).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { choose, linearSumAssignment, TUNED, DEFAULT } from '../static/engine/tracklets.js';

const FX = JSON.parse(readFileSync(new URL('./fixtures/tracklets/fixture.json', import.meta.url)));
const K = 133;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Mirrors gen_clip() in make_tracklets_fixture.py statement for statement.
function genClip(r, NMAX) {
  const T = 8 + Math.trunc(r() * 40), W = 320 + Math.trunc(r() * 960), H = 240 + Math.trunc(r() * 720);
  const P = 1 + Math.trunc(r() * 5), quant = r() < 0.4;
  const actors = [];
  for (let p = 0; p < P; p++) {
    const a = { x: r() * W * 0.8, y: r() * H * 0.5, w: 20 + r() * W * 0.4, h: 40 + r() * H * 0.5, vis: 0.4 + r() * 0.6, step: r() * 20 };
    if (p > 0 && r() < 0.2) for (const k of ['x', 'y', 'w', 'h']) a[k] = actors[p - 1][k];
    actors.push(a);
  }
  const KA = new Float32Array(T * NMAX * K * 2), SA = new Float32Array(T * NMAX * K), DB = new Float32Array(T * NMAX * 4);
  const ND = [];
  for (let t = 0; t < T; t++) {
    let boxes = [];
    for (const a of actors) {
      if (r() < a.vis) {
        a.x += (r() - 0.5) * a.step; a.y += (r() - 0.5) * a.step;
        const b = [a.x, a.y, a.x + a.w, a.y + a.h];
        boxes.push(quant ? b.map((v) => Math.trunc(v) + 0) : b);   // + 0: Python int() has no -0
      }
    }
    if (boxes.length && r() < 0.15) {
      const s = boxes[Math.trunc(r() * boxes.length)], o = (r() - 0.5) * 6;
      boxes.push([s[0] + o, s[1] + o, s[2] + o, s[3] + o]);
    }
    if (r() < 0.05) { const x = r() * W, y = r() * H; boxes.push([x, y, x, y + r() * 100]); }
    if (r() < 0.1) { const x = r() * W, y = r() * H; boxes.push([x, y, x + 1 + r() * 30, y + 1 + r() * 30]); }
    const keys = boxes.map(() => r());
    boxes = boxes.map((_, i) => i).sort((i, j) => keys[i] - keys[j] || i - j).map((i) => boxes[i]).slice(0, NMAX);
    ND.push(boxes.length);
    boxes.forEach((b, j) => {
      const s = (t * NMAX + j) * K;
      DB.set(b, (t * NMAX + j) * 4);
      for (let k = 0; k < K; k++) {
        SA[s + k] = r();
        KA[(s + k) * 2] = b[0] + r() * (b[2] - b[0]); KA[(s + k) * 2 + 1] = b[1] + r() * (b[3] - b[1]);
      }
      const u = r();
      if (u < 0.1) for (let k = 0; k < 23; k++) SA[s + k] = r() * 0.35;
      else if (u < 0.15) for (let k = 0; k < 23; k++) SA[s + k] = k % 2 ? 0.3 : 0.9;
    });
  }
  const sha = createHash('sha256');
  for (const a of [KA, SA, DB]) sha.update(new Uint8Array(a.buffer));
  const frames = ND.map((n, t) => (n ? {
    n, kp: KA.subarray(t * NMAX * K * 2, (t * NMAX + n) * K * 2), sc: SA.subarray(t * NMAX * K, (t * NMAX + n) * K),
    box: DB.subarray(t * NMAX * 4, (t * NMAX + n) * 4) } : undefined));
  return { T, W, H, frames, sha256: sha.digest('hex') };
}

test('tracker v2: choose() identical to tracker_v2.py on the random edge-case clips, TUNED and DEFAULT', () => {
  const r = mulberry32(FX.seed);
  assert.deepEqual([r(), r(), r(), r()], FX.first, 'PRNG differs from the Python generator');
  let frames = 0;
  FX.clips.forEach((exp, c) => {
    const g = genClip(r, FX.nmax);
    assert.deepEqual([g.T, g.W, g.H, g.sha256], [exp.T, exp.W, exp.H, exp.sha256], `clip ${c}: inputs differ from Python`);
    assert.deepEqual([...choose(g.frames, g.W, g.H, TUNED)], exp.TUNED, `clip ${c} TUNED`);
    assert.deepEqual([...choose(g.frames, g.W, g.H, DEFAULT)], exp.DEFAULT, `clip ${c} DEFAULT`);
    frames += g.T;
  });
  assert.equal(frames, FX.stats.frames);
  assert.ok(FX.stats.ties > 0 && FX.stats.rows_gt_cols > 0, 'fixture must exercise assignment ties and rows > cols');
});

test('tracker v2: the self-test scenario of tracker_v2.py (follow the signer, mask 10-14, never the bystander)', () => {
  const r = mulberry32(0), T = 30, W = 600, H = 900, frames = [];
  const u = (lo, hi) => lo + r() * (hi - lo);
  for (let t = 0; t < T; t++) {
    const dets = [{ box: [0, 300, 120, 850], kp: Array.from({ length: K }, () => [u(0, 120), u(300, 850)]) }];   // bystander, index 0
    if (!(t >= 10 && t < 15)) {
      const kp = Array.from({ length: K }, (_, k) => [u(200, 420), u(100, 850)].map((v) => v + (k >= 91 ? (r() - 0.5) * 16 : 0)));
      dets.push({ box: [200, 100, 420, 850], kp });
      dets.push({ box: [202, 102, 422, 852], kp: kp.map(([x, y]) => [x + 1, y + 1]) });              // duplicate of the signer
    }
    frames.push({ n: dets.length, kp: Float32Array.from(dets.flatMap((d) => d.kp.flat())),
      sc: new Float32Array(dets.length * K).fill(0.9), box: Float32Array.from(dets.flatMap((d) => d.box)) });
  }
  for (const w of [DEFAULT, TUNED]) {
    const ch = choose(frames, W, H, w);
    for (let t = 0; t < T; t++) {
      if (t >= 10 && t < 15) assert.equal(ch[t], -1, `frame ${t} must be masked`);
      else assert.ok(ch[t] === 1 || ch[t] === 2, `frame ${t} chose ${ch[t]}`);
    }
  }
});

test('tracker v2: assignment tie-break is the first minimum in lexicographic order, as tracker_v2.py', () => {
  assert.deepEqual(linearSumAssignment([[0.5, 0.5], [0.5, 0.5]]), [[0, 1], [0, 1]]);
  assert.deepEqual(linearSumAssignment([[1, 0, 0]]), [[0], [1]]);
  assert.deepEqual(linearSumAssignment([[0.2], [0.1], [0.1]]), [[1], [0]]);        // rows > cols
  assert.deepEqual(linearSumAssignment([[0, 1], [1, 0], [0, 0]]), [[0, 1], [0, 1]]);
  assert.deepEqual(choose([undefined, { n: 0 }], 100, 100, TUNED), Int16Array.from([-1, -1]));
});

test('tracker v2: keypoint scores exactly float32(0.3) do not count (numpy compares in float32, as tracker_v2.py)', () => {
  // tracker_v2.py gives [-1] with 4 body scores at 0.9 and the rest at float32(0.3), and [0] with 5. A float64 compare
  // (0.30000001 > 0.3) would count all 23 and choose the detection in both cases.
  for (const [strong, exp] of [[4, -1], [5, 0]]) {
    const sc = new Float32Array(K).fill(0.3);
    sc.fill(0.9, 0, strong);
    const fr = { n: 1, kp: new Float32Array(K * 2), sc, box: Float32Array.from([10, 10, 50, 90]) };
    assert.deepEqual([...choose([fr], 100, 100, TUNED)], [exp]);
  }
});
