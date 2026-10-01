// Parity of demo/static/engine/activity.js with results/signing_activity/signing_activity.py::features (the 8 kinematic
// features) and with the arithmetic of the frozen model. Fixtures: python demo/tests/fixtures/make_activity_fixtures.py
// (binaries under results/live_onnx/fixtures/activity, gitignored). The synthetic cases need no fixtures.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FEATURES, RAISED, activityScore, kinematics, nanmedian, nanpercentile, scoreFeatures } from '../static/engine/activity.js';
import { ACTIVITY_MODEL } from '../static/engine/activity_model.js';

const ROOT = new URL('../../', import.meta.url).pathname;
const BIN = join(ROOT, 'results/live_onnx/fixtures/activity');
const expected = JSON.parse(readFileSync(new URL('./fixtures/activity/expected.json', import.meta.url)));
const F = 133 * 3;

// A still upper body: shoulders 100 px apart at y = 100, nose above, wrists 1.8 widths below the shoulder line.
function still(T) {
  const k = new Float32Array(T * F);
  for (let t = 0; t < T; t++) {
    const set = (j, x, y, c = 0.9) => { k[t * F + 3 * j] = x; k[t * F + 3 * j + 1] = y; k[t * F + 3 * j + 2] = c; };
    for (let j = 0; j < 133; j++) set(j, 50, 150);
    set(5, 100, 100); set(6, 0, 100); set(0, 50, 40); set(9, 90, 280); set(10, 10, 280);
    for (let i = 0; i < 21; i++) { set(91 + i, 90, 280); set(112 + i, 10, 280); }
  }
  return k;
}

test('synthetic: a still person has no motion, hands 1.8 shoulder widths below the shoulder line', () => {
  const f = kinematics(still(50), 50);
  assert.equal(f.wrist_speed_med, 0);
  assert.equal(f.finger_motion_med, 0);
  assert.equal(f.body_motion_med, 0);
  assert.equal(f.head_motion_med, 0);
  assert.equal(f.raised_share, 0);
  assert.ok(Math.abs(f.hand_height_med + 1.8) < 1e-6, f.hand_height_med);
});

test('synthetic: an oscillating right wrist at the shoulder line is fast and raised', () => {
  const k = still(50);
  for (let t = 0; t < 50; t++) k[t * F + 3 * 10 + 1] = 100 + 20 * Math.sin(t);
  const f = kinematics(k, 50);
  assert.ok(f.wrist_speed_med > 1, f.wrist_speed_med);
  assert.equal(f.raised_share, 1);
  assert.ok(f.hand_height_med > -RAISED + 0.55, f.hand_height_med);
});

test('synthetic: frames without confident shoulders give NaN speeds and are ignored', () => {
  const k = still(50);
  for (let t = 1; t < 50; t += 2) { k[t * F + 3 * 5 + 2] = 0; k[t * F + 3 * 6 + 2] = 0; }   // every other frame
  const f = kinematics(k, 50);
  assert.ok(Number.isNaN(f.wrist_speed_med) && Number.isNaN(f.body_motion_med), 'no two consecutive confident frames');
  assert.equal(f.raised_share, 0);
  assert.throws(() => kinematics(new Float32Array(10), 3), /expected/);
});

test('nanmedian / nanpercentile follow numpy', () => {
  assert.equal(nanmedian([3, NaN, 1, 2]), 2);
  assert.equal(nanmedian([1, 2, 3, 4]), 2.5);
  assert.ok(Number.isNaN(nanmedian([NaN])) && Number.isNaN(nanpercentile([], 90)));
  assert.equal(nanpercentile([0, 10, 20, 30, 40], 90), 36);            // np.nanpercentile: pos = 3.6 -> 30 + 0.6 * 10
  assert.equal(nanpercentile([5], 90), 5);
});

test('the frozen model: intercept + standardised weights; a NaN feature takes the fitted median', () => {
  const m = ACTIVITY_MODEL;
  assert.deepEqual(FEATURES, m.features);
  assert.equal(m.features.length, 8);
  const atMean = Object.fromEntries(m.features.map((n, i) => [n, m.mean[i]]));
  assert.ok(Math.abs(scoreFeatures(atMean) - m.intercept) < 1e-12);
  const allNaN = Object.fromEntries(m.features.map((n) => [n, NaN]));
  const atMedian = Object.fromEntries(m.features.map((n, i) => [n, m.median[i]]));
  assert.equal(scoreFeatures(allNaN), scoreFeatures(atMedian));
  // one standard deviation up in a feature moves the score by exactly that feature's weight
  const up = { ...atMean, hand_height_med: m.mean[3] + m.scale[3] };
  assert.ok(Math.abs(scoreFeatures(up) - m.intercept - m.coef[3]) < 1e-12);
  assert.equal(m.test.scenes > 600, true);
});

test('a fast raised hand outranks a still one (the ranking the roster relies on)', () => {
  const still50 = still(50), signing = still(50);
  for (let t = 0; t < 50; t++) {
    signing[t * F + 3 * 10 + 1] = 90 + 30 * Math.sin(t * 1.3);
    signing[t * F + 3 * 10] = 20 + 15 * Math.cos(t * 0.9);
    for (let i = 0; i < 21; i++) { signing[t * F + 3 * (112 + i) + 1] = 90 + 30 * Math.sin(t * 1.3) + 2 * i * Math.sin(t); signing[t * F + 3 * (112 + i)] = 20 + 15 * Math.cos(t * 0.9); }
  }
  assert.ok(activityScore(signing, 50).score > activityScore(still50, 50).score);
});

const have = existsSync(BIN);
test('parity with signing_activity.features on real clips, first-25-frame windows and clips with lost frames',
  { skip: have ? false : 'fixtures not on this machine (run make_activity_fixtures.py)' }, () => {
    let worst = 0, nan = 0, scoreWorst = 0;
    for (const c of expected.cases) {
      const bytes = readFileSync(join(BIN, `${c.name}.f32`));
      const kps = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
      assert.equal(kps.length, c.T * F, c.name);
      const got = kinematics(kps, c.T);
      for (const name of FEATURES) {
        const want = c.features[name];
        if (want === null) { assert.ok(Number.isNaN(got[name]), `${c.name} ${name}: expected NaN, got ${got[name]}`); nan++; continue; }
        assert.ok(Number.isFinite(got[name]), `${c.name} ${name}: got ${got[name]}`);
        const d = Math.abs(got[name] - want) / Math.max(1, Math.abs(want));
        worst = Math.max(worst, d);
        assert.ok(d < 1e-9, `${c.name} ${name}: ${got[name]} vs ${want}`);
      }
      const s = activityScore(kps, c.T).score;
      scoreWorst = Math.max(scoreWorst, Math.abs(s - c.score));
      assert.ok(Math.abs(s - c.score) < 1e-8, `${c.name} score ${s} vs ${c.score}`);
    }
    console.log(`activity: ${expected.cases.length} cases identical (worst relative feature difference ${worst.toExponential(1)}, worst score difference ${scoreWorst.toExponential(1)}, ${nan} NaN features matched)`);
  });
