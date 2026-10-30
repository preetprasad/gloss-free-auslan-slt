// persons.js (who is in view, who is signing) and its use by engine.js, on synthetic scenes: no images, no models.
// Skeletons are minimal (shoulders, nose, wrists, fingers); a "signer" swings a wrist and its fingers, a "listener" sits still.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createEngine } from '../static/engine/engine.js';
import { LOST_FRAMES, MAX_PERSONS, MIN_VISIBLE, Session, labelOf } from '../static/engine/persons.js';

const K = 133, F = 3 * K, W = 960, H = 720, DT = 40;

/** One detection: shoulders `wid` px apart centred on (x, 300), nose above, wrists below; `signing` swings the right hand. */
function person(x, wid, t, signing) {
  const kp = new Float32Array(K * 2), sc = new Float32Array(K).fill(0.9);
  const put = (j, px, py) => { kp[2 * j] = px; kp[2 * j + 1] = py; };
  for (let j = 0; j < K; j++) put(j, x, 400);
  put(5, x + wid / 2, 300); put(6, x - wid / 2, 300); put(0, x, 230);
  let wx = x - wid * 0.4, wy = 300 + wid * 1.8;                             // right wrist (index 10), at rest
  if (signing) { wx += wid * 0.5 * Math.cos(t * 0.9); wy = 300 - wid * 0.1 + wid * 0.6 * Math.sin(t * 1.3); }
  put(9, x + wid * 0.4, 300 + wid * 1.8); put(10, wx, wy);
  for (let i = 0; i < 21; i++) {
    put(91 + i, x + wid * 0.4, 300 + wid * 1.8);
    put(112 + i, wx + (signing ? i * 0.3 * Math.sin(t + i) : 0), wy + (signing ? i * 0.3 * Math.cos(t * 1.1 + i) : 0));
  }
  return { kp, sc };
}

/** A frame's detections in the given order -> the `all` the pose stage returns. */
function frame(people) {
  const kp = new Float32Array(people.length * K * 2), sc = new Float32Array(people.length * K);
  people.forEach((p, i) => { kp.set(p.kp, i * K * 2); sc.set(p.sc, i * K); });
  return { kp, sc, n: people.length, width: W };
}

const times = (n) => Array.from({ length: n }, (_, i) => i * DT);
function run(session, scene, n = 50, picked = () => 0) {
  const u = session.utterance();
  for (let t = 0; t < n; t++) { const people = scene(t); u.step(people ? frame(people) : undefined, picked(t)); }
  return u;
}
// Left person listens, right person signs; the detector lists them in a different order on odd frames.
const conversation = (t) => {
  const left = person(250, 110, t, false), right = person(650, 110, t, true);
  return t % 2 ? [right, left] : [left, right];
};

test('one person in view: a label, no choice to make (the engine keeps the single-signer path)', () => {
  const s = new Session(), u = run(s, (t) => [person(400, 110, t, true)]);
  const sel = u.select(times(50));
  assert.equal(sel.multi, false);
  assert.deepEqual(sel.people.map((p) => p.label), ['Person A']);
  assert.equal(sel.chosen, null);
});

test('two people: the one moving like a signer is chosen, whatever order the detector lists them in', () => {
  const s = new Session(), u = run(s, conversation);
  const sel = u.select(times(50));
  assert.equal(sel.multi, true);
  assert.equal(sel.people.length, 2);
  const chosen = sel.people.find((p) => p.id === sel.chosen);
  assert.ok(chosen.x > 0.5, `the signer sits on the right of the raw frame, x = ${chosen.x}`);
  assert.ok(chosen.score > sel.people.find((p) => p.id !== sel.chosen).score);
  // labels go left to right in the mirrored preview: the raw-right person is on the preview's left, so Person A
  assert.equal(chosen.label, 'Person A');
  assert.equal(sel.people.find((p) => p.id !== sel.chosen).label, 'Person B');
  const kps = u.keypointsFor(sel.chosen);
  assert.equal(kps.length, 50);
  assert.ok(kps.every((k) => k[3 * 5] > 500), 'every frame holds the right-hand person');   // shoulder x
});

test('the same two people keep their labels across utterances, and labels are not reused for someone else', () => {
  const s = new Session();
  const first = run(s, conversation).select(times(50));
  const second = run(s, conversation).select(times(50));
  const by = (sel) => Object.fromEntries(sel.people.map((p) => [p.label, Math.round(p.x * 10)]));
  assert.deepEqual(by(second), by(first));
  const third = run(s, (t) => [person(250, 110, t, true), person(650, 110, t, false), person(450, 110, t, false)]).select(times(50));
  assert.deepEqual(third.people.map((p) => p.label).sort(), ['Person A', 'Person B', 'Person C']);
  assert.equal(s.labels, 3);
});

test('a person who leaves and comes back to the same place is the same person; elsewhere is a new one', () => {
  const s = new Session();
  const scene = (back) => (t) => (t < 15 ? [person(250, 110, t, false), person(650, 110, t, true)]
    : t < 15 + LOST_FRAMES + 10 ? [person(650, 110, t, true)] : [person(back, 110, t, false), person(650, 110, t, true)]);
  const n = 15 + LOST_FRAMES + 10 + 20;
  const same = run(s, scene(255), n).select(times(n));
  assert.equal(same.people.length, 2, 'the returning person is not counted as a third');
  const s2 = new Session();
  const other = run(s2, scene(520), n).select(times(n));
  assert.equal(other.people.length, 3, 'someone who returns far from where they left is a new person');
});

test('a short false detection is not a person and does not use up a label', () => {
  const s = new Session();
  const scene = (t) => [person(400, 110, t, true), ...(t >= 20 && t < 24 ? [person(120, 90, t, false)] : [])];
  const u = run(s, scene);
  assert.equal(u.select(times(50)).multi, false);
  assert.equal(s.people.length, 1, 'the blip was forgotten');
  const next = run(s, (t) => [person(400, 110, t, true), person(750, 110, t, false)]).select(times(50));
  assert.deepEqual(next.people.map((p) => p.label).sort(), ['Person A', 'Person B']);
});

test('only people in view for 80% of the utterance can be chosen, even if they move more', () => {
  const s = new Session();
  const scene = (t) => (t < 30 ? [person(250, 110, t, false), person(650, 110, t, true)] : [person(250, 110, t, false)]);
  const sel = run(s, scene, 50).select(times(50));
  assert.equal(sel.multi, true);
  const mover = sel.people.find((p) => !p.eligible);
  assert.ok(mover && mover.share < MIN_VISIBLE);
  assert.equal(sel.chosen, sel.people.find((p) => p.eligible).id, 'the person who was there throughout is chosen');
  const none = run(new Session(), (t) => (t % 3 ? [person(250, 110, t, true)] : [person(650, 110, t, true)]), 50).select(times(50));
  assert.equal(none.chosen, null, 'nobody was in view for 80% of the utterance: no choice, the engine keeps the single-signer stream');
});

test("'online' follows the person the shipped tracker followed; a number picks a person (a switch)", () => {
  const s = new Session();
  const u = run(s, conversation, 50, (t) => (t % 2 ? 1 : 0));            // the tracker follows the left person: index 1 on odd frames
  const online = u.select(times(50), 'online');
  const activity = u.select(times(50), 'activity');
  assert.notEqual(online.chosen, activity.chosen);
  assert.equal(online.people.find((p) => p.id === online.chosen).x < 0.5, true);
  const manual = u.select(times(50), activity.people.find((p) => p.id !== activity.chosen).id);
  assert.equal(manual.chosen, online.chosen);
  assert.equal(u.select(times(50), 9999).chosen, null);
  assert.equal(u.idOf('Person B'), activity.people.find((p) => p.label === 'Person B').id);
  assert.equal(u.idOf('Person Z'), undefined);
});

test('at most six people are followed per frame; extra detections are ignored, not mislabelled', () => {
  const s = new Session();
  const crowd = (t) => Array.from({ length: 9 }, (_, i) => person(60 + i * 100, 60 + i * 4, t, false));
  const sel = run(s, crowd).select(times(50));
  assert.equal(sel.people.length, MAX_PERSONS);
});

test('clear() forgets everyone and restarts the labels', () => {
  const s = new Session();
  run(s, conversation).select(times(50));
  assert.equal(s.people.length, 2);
  s.clear();
  assert.equal(s.people.length, 0);
  const sel = run(s, (t) => [person(300, 110, t, true)]).select(times(50));
  assert.deepEqual(sel.people.map((p) => p.label), ['Person A']);
  assert.equal(labelOf(0), 'Person A'); assert.equal(labelOf(25), 'Person Z'); assert.equal(labelOf(26), 'Person 27');
});

test('frames with no detection, a missing `all`, or unconfident shoulders are skipped without harm', () => {
  const s = new Session();
  const u = run(s, (t) => (t % 4 === 0 ? undefined : t % 4 === 1 ? [] : [person(400, 110, t, true)]));
  const sel = u.select(times(50));
  assert.equal(sel.people.length, 1);
  const bad = person(400, 110, 0, true); bad.sc[5] = 0;                       // a left shoulder that is not confident
  assert.equal(run(new Session(), () => [bad]).select(times(50)).people.length, 0);
});

// ---- the engine: a fake runtime that hands persons.js real detections
const MODELS = { single: { label: 'S', threshold: 0.7, coverage: 0.5, needs_frames: false }, ensemble: { label: 'E', threshold: 0.9, coverage: 0.5, needs_frames: true } };
const JPEG = (k) => `data:image/jpeg;base64,${btoa(String.fromCharCode(0xff, 0xd8, k))}`;
const tick = () => new Promise((r) => setImmediate(r));

function harness(scene, translated = [], choice = 'activity', gate = null) {
  const out = [];
  let frameNo = 0;
  const runtime = {
    init: async () => ({ models: MODELS, max_frames: 300, runtime: {} }),
    ready: () => true,
    extract: async (jpeg) => {
      const people = scene(frameNo++);
      if (!people) return { keypoints: new Float32Array(F), count: 0 };
      const all = frame(people);
      const kp = new Float32Array(F);                                        // the shipped tracker's stream: always detection 0
      for (let j = 0; j < K; j++) { kp[3 * j] = all.kp[2 * j]; kp[3 * j + 1] = all.kp[2 * j + 1]; kp[3 * j + 2] = all.sc[j]; }
      return { keypoints: kp, count: people.length, all, picked: 0 };
    },
    translate: async (model, keypoints, times, jpegs) => {
      if (gate) await gate;
      translated.push({ model, x: keypoints.map((k) => Math.round(k[3 * 5])), jpegs: jpegs.length, times: times.length });
      return { text: 'hello.', confidence: 0.5, translate_ms: 1, rgb_ms: 0, frames_used: keypoints.length };
    },
  };
  const handle = createEngine((m) => out.push(m), runtime, () => 0, { choice });
  return { out, handle, result: () => out.filter((m) => m.type === 'result').at(-1),
    async utterance(id, model = 'single', n = 30) {
      handle({ type: 'start_recording', request_id: id, model });
      for (let t = 0; t < n; t++) handle({ type: 'frame', request_id: id, captured_at: t * DT, data: JPEG(t) });
      await tick(); await tick();
      handle({ type: 'translate', request_id: id }); await tick(); await tick(); await tick();
    } };
}

test('engine: two people in view -> the signer is translated and named; one tap re-translates the other person', async () => {
  const translated = [];
  const h = harness(conversation, translated);
  await h.handle({ type: 'init' });
  await h.utterance('u1');
  const r = h.result();
  assert.equal(r.persons.mode, 'activity');
  assert.equal(r.persons.chosen, 'Person A');
  assert.equal(r.persons.count, 2);
  assert.equal(r.persons.session_people, 2);
  assert.deepEqual(r.persons.people.map((p) => p.label).sort(), ['Person A', 'Person B']);
  assert.ok(translated[0].x.every((x) => x > 500), 'the translator was given the right-hand (signing) person on every frame');
  h.handle({ type: 'retranslate', request_id: 'u1', person: 'Person B' }); await tick(); await tick(); await tick();
  const r2 = h.result();
  assert.equal(r2.persons.mode, 'manual'); assert.equal(r2.persons.chosen, 'Person B');
  assert.ok(translated[1].x.every((x) => x < 500), 'now the left-hand person');
  h.handle({ type: 'retranslate', request_id: 'u1', person: 'Person Z' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /not in view/);
});

test('engine: with one person in view nothing changes: the shipped single-signer stream is translated, nothing is retained', async () => {
  const translated = [];
  const h = harness((t) => [person(400, 110, t, true)], translated);
  await h.handle({ type: 'init' });
  await h.utterance('solo');
  assert.equal(h.result().persons.mode, 'single');
  assert.equal(h.result().persons.chosen, 'Person A');
  assert.equal(translated[0].x.length, 30);
  h.handle({ type: 'retranslate', request_id: 'solo', person: 'Person A' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /no longer available/);
});

test('engine: retained people are released by Clear, clear_session and the next utterance; the ensemble keeps frames only for a switch', async () => {
  const translated = [];
  const h = harness(conversation, translated);
  await h.handle({ type: 'init' });
  await h.utterance('e1', 'ensemble');
  h.handle({ type: 'retranslate', request_id: 'e1', person: 'Person B' }); await tick(); await tick(); await tick();
  assert.equal(translated.at(-1).jpegs, 30, 'the ensemble needs the frames of the utterance for the other person');
  h.handle({ type: 'stop_recording', request_id: 'e1' }); await tick();      // the page's Clear
  h.handle({ type: 'retranslate', request_id: 'e1', person: 'Person B' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /no longer available/);
  await h.utterance('e2', 'single');
  h.handle({ type: 'clear_session' }); await tick();                         // Clear conversation / camera off
  h.handle({ type: 'retranslate', request_id: 'e2', person: 'Person B' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /no longer available/);
  await h.utterance('e3', 'single');                                         // labels start again
  assert.equal(h.result().persons.session_people, 2);
  assert.equal(h.result().persons.chosen, 'Person A');
  h.handle({ type: 'start_recording', request_id: 'e4', model: 'single' }); await tick();
  h.handle({ type: 'retranslate', request_id: 'e3', person: 'Person B' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /no longer available/);
});

// ---- fixes found by the review of this build
test('one person detected twice is one person; the tracker following the dropped twin still counts for that person', () => {
  const twice = (t) => { const a = person(400, 110, t, true), b = person(415, 108, t, true); return t % 2 ? [b, a] : [a, b]; };
  const s = new Session(), u = run(s, twice, 50, (t) => (t % 2 ? 0 : 1));
  const sel = u.select(times(50), 'online');
  assert.equal(sel.multi, false, 'a duplicate box is not a second person');
  assert.equal(sel.people.length, 1);
  assert.equal(s.labels, 1);
  const next = run(s, (t) => [person(250, 110, t, true), person(650, 110, t, false)]).select(times(50));
  assert.deepEqual(next.people.map((p) => p.label).sort(), ['Person A', 'Person B'].sort(), 'labels are not inflated');
  // real neighbours (shoulders touching, one shoulder width apart) are still two people
  assert.equal(run(new Session(), (t) => [person(400, 110, t, true), person(510, 110, t, false)]).select(times(50)).multi, true);
  // the tracker followed the dropped twin on every frame: the person it stands for is what 'online' follows
  const u2 = run(new Session(), (t) => [person(400, 110, t, false), person(650, 110, t, true), person(410, 110, t, false)], 50, () => 2);
  const s2 = u2.select(times(50), 'online');
  assert.equal(s2.people.find((p) => p.id === s2.chosen).x < 0.5, true);
});

test('a session cleared while an utterance is recorded cannot merge two people into one track', () => {
  const s = new Session(), u = s.utterance();
  for (let t = 0; t < 40; t++) {
    if (t === 15) s.clear();                                            // Clear conversation in the middle of the recording
    const l = person(250, 110, t, false), r = person(650, 110, t, true);
    u.step(frame(t % 2 ? [r, l] : [l, r]), 0);
  }
  for (const [id, fr] of u.frames) {
    const xs = new Set(fr.filter(Boolean).map((k) => Math.round(k[3 * 5] / 300)));
    assert.equal(xs.size, 1, `person ${id} must be one physical person, not a splice of two`);
  }
  assert.equal(u.select(times(40)).chosen, null, 'nobody was tracked for 80% of the utterance, so nobody is chosen');
});

test('engine: a translation cancelled while it runs still answers, so the page stops waiting', async () => {
  const out = [];
  let release;
  const gate = new Promise((r) => { release = r; });
  const runtime = { init: async () => ({ models: MODELS, max_frames: 300, runtime: {} }), ready: () => true,
    extract: async () => ({ keypoints: new Float32Array(F), count: 0 }),
    translate: async () => { await gate; return { text: 'late.', confidence: 0.5, translate_ms: 1, rgb_ms: 0, frames_used: 2 }; } };
  const handle = createEngine((m) => out.push(m), runtime, () => 0);
  await handle({ type: 'init' });
  handle({ type: 'start_recording', request_id: 'c1', model: 'single' });
  for (let t = 0; t < 3; t++) handle({ type: 'frame', request_id: 'c1', captured_at: t * DT, data: JPEG(t) });
  await tick(); await tick();
  handle({ type: 'translate', request_id: 'c1' }); await tick();
  handle({ type: 'stop_recording', request_id: 'c1' }); await tick();      // Cancel while it translates
  release(); await tick(); await tick();
  assert.equal(out.filter((m) => m.type === 'result').length, 0);
  assert.deepEqual(out.filter((m) => m.type === 'cancelled'), [{ channel: 1, type: 'cancelled', request_id: 'c1' }]);
  handle({ type: 'start_recording', request_id: 'c2', model: 'single' });   // and a new utterance is accepted afterwards
  assert.equal(out.at(-1).type, 'recording_started');
});

test('engine: Clear conversation while a two-person utterance translates leaves nothing retained', async () => {
  let release; const gate = new Promise((r) => { release = r; });
  const h = harness(conversation, [], 'activity', gate);
  await h.handle({ type: 'init' });
  h.handle({ type: 'start_recording', request_id: 'k1', model: 'single' });
  for (let t = 0; t < 30; t++) h.handle({ type: 'frame', request_id: 'k1', captured_at: t * DT, data: JPEG(t) });
  await tick(); await tick();
  h.handle({ type: 'translate', request_id: 'k1' }); await tick();
  h.handle({ type: 'clear_session' });                                       // while the translation is still running
  release(); await tick(); await tick(); await tick();
  assert.equal(h.out.filter((m) => m.type === 'result').length, 1, 'the running translation still answers');
  h.handle({ type: 'retranslate', request_id: 'k1', person: 'Person B' }); await tick(); await tick();
  assert.match(h.out.at(-1).error, /no longer available/, 'nothing was kept for a switch');
});
