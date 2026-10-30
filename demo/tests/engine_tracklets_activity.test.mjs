// Gate C variant (engine/tracklets_activity.js): v2 tracklets, signer chosen by activity, never by box area.
// Full evaluation on the ground-truth clips: results/tracker_v2/port/eval_variant.mjs + score_variant.py.
import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseByActivity } from '../static/engine/tracklets_activity.js';

const K = 133;

// people: [{x, half, signs, from, to}] -> frames with a box per person (the bystander's box is the larger one).
function scene(T, people) {
  const frames = [];
  for (let t = 0; t < T; t++) {
    const on = people.filter((p) => t >= (p.from ?? 0) && t < (p.to ?? T)), n = on.length;
    const kp = new Float32Array(n * K * 2), sc = new Float32Array(n * K).fill(0.9), box = new Float32Array(n * 4);
    on.forEach((p, i) => {
      const put = (j, x, y) => { kp[(i * K + j) * 2] = x; kp[(i * K + j) * 2 + 1] = y; };
      for (let j = 0; j < K; j++) put(j, p.x, 400);
      put(5, p.x + 55, 300); put(6, p.x - 55, 300); put(0, p.x, 230); put(9, p.x + 44, 500);
      const wy = p.signs ? 290 + 60 * Math.sin(t * 1.3) : 500, wx = p.x - 44 + (p.signs ? 50 * Math.cos(t * 0.9) : 0);
      put(10, wx, wy);
      for (let f = 0; f < 21; f++) { put(112 + f, wx, wy + (p.signs ? f * 0.3 * Math.sin(t + f) : 0)); put(91 + f, p.x + 44, 500); }
      box.set([p.x - p.half, 150, p.x + p.half, 700], i * 4);
    });
    frames.push(n ? { n, kp, sc, box } : undefined);
  }
  return frames;
}

test('picks the signing person, not the larger box, and masks frames the signer is absent', () => {
  const fr = scene(50, [{ x: 250, half: 150, signs: false }, { x: 700, half: 90, signs: true, to: 45 }]);
  const r = chooseByActivity(fr);
  assert.equal(r.by, 'activity');
  for (let t = 0; t < 45; t++) assert.equal(r.chosen[t], 1, `frame ${t}`);
  for (let t = 45; t < 50; t++) assert.equal(r.chosen[t], -1, `frame ${t}`);
});

test('one person in view is followed; nobody eligible among two declines', () => {
  const one = chooseByActivity(scene(30, [{ x: 400, half: 100, signs: true, to: 20 }]));
  assert.equal(one.by, 'single');                       // 20/30 visible: not eligible, but the only person
  assert.equal(one.chosen.filter((d) => d === 0).length, 20);
  const two = chooseByActivity(scene(30, [{ x: 250, half: 100, signs: true, to: 20 }, { x: 700, half: 100, signs: false, from: 12 }]));
  assert.equal(two.by, 'decline');
  assert.ok(two.chosen.every((d) => d === -1));
});
