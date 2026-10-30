// Tracker + 25 fps grid parity: demo/static/engine/track.js vs demo/service.py (_torso, seed_index, track_index,
// grid_indices). Inputs are regenerated from the same 32-bit PRNG as demo/tests/fixtures/make_pose_fixtures.py,
// so this needs only the committed manifest.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { torso, seedIndex, trackIndex, gridIndices, scoreMean } from '../static/engine/track.js';

const M = JSON.parse(readFileSync(new URL('./fixtures/pose/manifest.json', import.meta.url)));
const K = 133, f = Math.fround;

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

function below(x) {              // np.nextafter(np.float32(x), 0) for positive x
  const b = new Float32Array([x]);
  new Uint32Array(b.buffer)[0] -= 1;
  return b[0];
}

test('tracker: torso / seed_index / track_index / sc.mean() identical to service.py on 480 random frames', () => {
  const T = M.track, r = mulberry32(T.seed);
  assert.deepEqual([r(), r(), r(), r()], T.first, 'PRNG differs from the Python generator');
  let frames = 0, reseeds = 0;
  for (const c of T.cases) {
    const W = 320 + Math.trunc(r() * 960), H = 240 + Math.trunc(r() * 720);
    let cx = r() * W, cy = r() * H;
    const w = 20 + r() * 120;
    assert.equal(W, c.W); assert.equal(H, c.H);
    const track = {};
    for (const exp of c.frames) {
      const n = 1 + Math.trunc(r() * 4);
      let kp = new Float32Array(n * K * 2), sc = new Float32Array(n * K);
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < K; k++) sc[j * K + k] = r();
        let px, py, ww;
        if (j === 0) {
          const d = r() * (r() < 0.25 ? 3.2 : 0.8);
          const dx = r() * 2 - 1, dy = r() * 2 - 1;
          px = cx + d * w * dx; py = cy + d * w * dy;
          cx = px; cy = py;
          ww = w * (0.8 + 0.4 * r());
        } else {
          px = r() * W; py = r() * H; ww = 10 + r() * 150;
        }
        const a = (j * K + 5) * 2, b = (j * K + 6) * 2;
        kp[a] = px + ww / 2; kp[a + 1] = py + r() * 4 - 2;
        kp[b] = px - ww / 2; kp[b + 1] = py + r() * 4 - 2;
        const u = r();
        if (u < 0.08) sc[j * K + 5] = f(0.3);
        else if (u < 0.14) sc[j * K + 6] = below(0.3);
        else if (u < 0.18) { kp[b] = kp[a]; kp[b + 1] = kp[a + 1]; }
        else if (u < 0.20 && j > 0) { kp.copyWithin(j * K * 2, 0, K * 2); sc.copyWithin(j * K, 0, K); }
      }
      if (r() < 0.5) {
        const perm = Array.from({ length: n }, () => Math.trunc(r() * 1e9));
        const idx = [...perm.keys()].sort((p, q) => perm[p] - perm[q]);
        const kp2 = new Float32Array(kp.length), sc2 = new Float32Array(sc.length);
        idx.forEach((s, d) => { kp2.set(kp.subarray(s * K * 2, (s + 1) * K * 2), d * K * 2); sc2.set(sc.subarray(s * K, (s + 1) * K), d * K); });
        kp = kp2; sc = sc2;
      }
      assert.equal(n, exp.n);
      for (let j = 0; j < n; j++) {
        const t = torso(kp, sc, j);
        assert.deepEqual(t.mid === null ? null : [t.mid[0], t.mid[1], t.wid, t.ok], exp.torsos[j], `torso ${frames}/${j}`);
        assert.equal(scoreMean(sc, j), exp.sc_mean[j], `sc.mean ${frames}/${j}`);
      }
      assert.equal(seedIndex(kp, sc, n, [W, H]), exp.seed, `seed_index frame ${frames}`);
      const [i, reseeded] = trackIndex(kp, sc, n, track.mid ?? null, track.wid ?? null, [W, H]);
      assert.equal(i, exp.index, `track_index frame ${frames}`);
      assert.equal(reseeded, exp.reseeded);
      const t = torso(kp, sc, i);
      if (t.ok) { track.mid = t.mid; track.wid = t.wid; }
      assert.deepEqual(track.mid ? [track.mid[0], track.mid[1], track.wid] : null, exp.track);
      frames++; reseeds += reseeded;
    }
  }
  console.log(`tracker: ${frames} frames identical (${reseeds} re-seeds, ${frames - reseeds} tracked)`);
});

test('grid_indices identical to service.py on 200 capture timelines (incl. exact ties)', () => {
  const G = M.grid, r = mulberry32(G.seed);
  G.cases.forEach((exp, c) => {
    const n = 1 + Math.trunc(r() * 90);
    const t = [Math.trunc(r() * 1000)];
    for (let i = 0; i < n - 1; i++) {
      const gap = c % 3 ? 20 + Math.trunc(r() * 60) : 20 + Math.trunc(r() * 6) * 10;
      t.push(t[t.length - 1] + (c % 5 ? gap : gap + 0.5));
    }
    assert.equal(n, exp.n); assert.equal(t[t.length - 1], exp.t_last);
    assert.deepEqual(gridIndices(t), exp.idx, `case ${c}`);
  });
});

// D2: fractional capture times. numpy's arange fill is start + i*delta contracted to one FMA; the ties cases put frames
// exactly 20 ms either side of numpy's grid points, so an unfused start + i*delta picks the other frame on some of them.
test('grid_indices identical to service.py on fractional capture times (incl. the t0 = 237.7 case)', () => {
  const G = M.grid_frac, r = mulberry32(G.seed);
  G.random.forEach((exp, c) => {
    const n = 2 + Math.trunc(r() * 80);
    const t = [Math.trunc(r() * 1e8) / 1000];
    for (let i = 0; i < n - 1; i++) {
      const u = r();
      const gap = u < 0.6 ? 40.0 : u < 0.85 ? 20.0 : 20 + Math.trunc(r() * 60) + Math.trunc(r() * 10) / 10;
      t.push(t[t.length - 1] + gap);
    }
    assert.equal(n, exp.n); assert.equal(t[0], exp.t0); assert.equal(t[t.length - 1], exp.t_last);
    assert.deepEqual(gridIndices(t), exp.idx, `random case ${c}`);
  });
  const unfused = (t) => {        // the pre-fix formula, to show the ties separate it
    const t0 = t[0], n = Math.ceil((t[t.length - 1] + 1e-6 - t0) / 40), second = t0 + 40, d = second - t0;
    return Array.from({ length: n }, (_, g) => {
      const x = g === 0 ? t0 : g === 1 ? second : t0 + g * d;
      let b = 0;
      t.forEach((v, j) => { if (Math.abs(v - x) < Math.abs(t[b] - x)) b = j; });
      return b;
    });
  };
  let separated = 0;
  G.ties.forEach((c, k) => {
    assert.deepEqual(gridIndices(c.times), c.idx, `ties case ${k}${k ? '' : ' (t0 = 237.7)'}`);
    const wrong = JSON.stringify(unfused(c.times)) !== JSON.stringify(c.idx);
    assert.equal(wrong, c.unfused_wrong, `ties case ${k}: the unfused formula's disagreement differs from Python's`);
    separated += wrong;
  });
  assert.ok(G.ties[0].unfused_wrong && separated === G.ties_unfused_wrong && separated > 0);
  console.log(`grid fractional: ${G.random.length} random + ${G.ties.length} ties cases identical; an unfused start + i*delta ` +
    `would fail ${G.random_unfused_wrong} random and ${separated} ties cases`);
});
