// Who is in view, and who is signing: a roster of the people detected while an utterance is recorded, followed frame to
// frame by where their shoulders are (the online tracker's distance and gate, extended to everyone in view), labelled
// Person A, B, ... for the session, and ranked at Finish by the signing-activity score (activity.js).
//
// Privacy: no image, face or appearance data is used or kept. The session roster holds a label and the last torso
// position per person, and lives in the worker until Session.clear(), camera off, or the page closes. Each utterance
// additionally holds the people's keypoints until the next utterance, Clear or camera off, only when two or more people
// were in view (so a switch of person can re-translate); a single-person utterance keeps nothing after Finish.
//
// Parameters are fixed a priori (results/person_labels/PREREG_PERSON_LABELS_2026-09-30.md); none is tuned on data.
import { GATE, gridIndices, scoreMean, torso } from './track.js';
import { activityScore } from './activity.js';

export const LOST_FRAMES = 25;        // a person unseen for this many captured frames is no longer "in view"
export const MIN_TRACK_FRAMES = 10;   // a person counts only if seen in at least max(10, 20% of) the captured frames
export const MIN_TRACK_SHARE = 0.2;
export const MIN_VISIBLE = 0.8;       // eligible to be translated: a confident torso on 80% of the 25 fps grid (stricter than features.js's any-signal refusal)
export const DUP_DIST = 0.5;          // two detections whose shoulder midpoints are closer than this many shoulder widths are one person
export const MAX_PERSONS = 6;         // the widest-shouldered people kept per frame (memory bound)
const K = 133, F = 3 * K;

export const labelOf = (n) => `Person ${n < 26 ? String.fromCharCode(65 + n) : n + 1}`;

/** The people seen so far in this session: {id, n (label number or null until first counted), mid, wid}. */
export class Session {
  constructor() { this.ids = 0; this.clear(); }
  // ids are never reused, so an utterance still being recorded when the session is cleared cannot mix two people's frames
  clear() { this.people = []; this.labels = 0; }
  utterance() { return new Utterance(this); }
}

// Greedy one-to-one nearest match of detections to people by shoulder-midpoint distance in that person's shoulder widths.
function match(dets, people, used) {
  const pairs = [];
  for (const d of dets) {
    if (d.person !== undefined) continue;
    for (const p of people) {
      if (used.has(p.id)) continue;
      const dist = Math.hypot(d.mid[0] - p.mid[0], d.mid[1] - p.mid[1]) / Math.max(p.wid, 1e-3);
      if (dist <= GATE) pairs.push({ dist, d, p });
    }
  }
  pairs.sort((a, b) => a.dist - b.dist || a.d.i - b.d.i || a.p.id - b.p.id);
  for (const { d, p } of pairs) {
    if (d.person !== undefined || used.has(p.id)) continue;
    d.person = p; used.add(p.id);
  }
}

export class Utterance {
  constructor(session) {
    this.s = session; this.t = 0;
    this.frames = new Map();     // person id -> sparse array over captured frames of Float32Array(399) (x, y, score per keypoint)
    this.seen = new Map();       // person id -> {last, count, sx, sw}: last frame seen, frames seen, sums of torso x and frame width
    this.online = [];            // per captured frame: id of the person the shipped online tracker followed, or -1
    this.created = new Set();    // ids first seen in this utterance (dropped at the end if they never counted as a person)
    this.trace = null;           // evaluation only: set to [] to record, per captured frame, [detection index, person id] pairs
  }

  /**
   * One captured frame. all = {kp: Float32Array(n * 133 * 2), sc: Float32Array(n * 133), n, width} for the n detections
   * (undefined when the pose stage found nobody or failed); pickedIndex = the detection the online tracker followed.
   */
  step(all, pickedIndex = -1) {
    const t = this.t++, s = this.s;
    this.online.push(-1);
    if (this.trace) this.trace[t] = [];
    if (!all || !all.n) return;
    const dets = [];
    for (let i = 0; i < all.n; i++) {
      const { mid, wid, ok } = torso(all.kp, all.sc, i);
      if (ok) dets.push({ i, mid, wid });
    }
    // One person detected twice (an offset duplicate box) is one person: keep the more confident detection of a close pair.
    dets.sort((a, b) => scoreMean(all.sc, b.i) - scoreMean(all.sc, a.i) || b.wid - a.wid || a.i - b.i);
    const kept = [], dupOf = new Map();
    for (const d of dets) {
      const twin = kept.find((k) => Math.hypot(d.mid[0] - k.mid[0], d.mid[1] - k.mid[1]) / Math.max(d.wid, k.wid) < DUP_DIST);
      if (twin) dupOf.set(d.i, twin); else kept.push(d);
    }
    dets.length = 0; dets.push(...kept.sort((a, b) => a.i - b.i));
    if (dets.length > MAX_PERSONS) { dets.sort((a, b) => b.wid - a.wid || a.i - b.i); dets.length = MAX_PERSONS; }
    const inView = (p) => this.seen.has(p.id) && t - this.seen.get(p.id).last <= LOST_FRAMES;
    const used = new Set();
    match(dets, s.people.filter(inView), used);                  // people already followed in this utterance first,
    match(dets, s.people.filter((p) => !inView(p)), used);       // then anyone seen earlier in the session, by where they were
    for (const d of dets) {
      if (d.person === undefined) { d.person = { id: s.ids++, n: null, mid: d.mid, wid: d.wid }; s.people.push(d.person); this.created.add(d.person.id); }
      const p = d.person;
      p.mid = d.mid; p.wid = d.wid;
      const k = new Float32Array(F);
      for (let j = 0; j < K; j++) { k[3 * j] = all.kp[(d.i * K + j) * 2]; k[3 * j + 1] = all.kp[(d.i * K + j) * 2 + 1]; k[3 * j + 2] = all.sc[d.i * K + j]; }
      if (!this.frames.has(p.id)) this.frames.set(p.id, []);
      this.frames.get(p.id)[t] = k;
      const r = this.seen.get(p.id) ?? { first: t, firstX: d.mid[0], last: t, count: 0, sx: 0, sw: 0 };
      r.last = t; r.count++; r.sx += d.mid[0]; r.sw += all.width ?? 0;
      this.seen.set(p.id, r);
      if (d.i === pickedIndex) this.online[t] = p.id;
      if (this.trace) this.trace[t].push([d.i, p.id]);
    }
    const twin = dupOf.get(pickedIndex);                        // the tracker followed the dropped twin of a kept detection
    if (twin?.person && this.online[t] === -1) this.online[t] = twin.person.id;
    if (this.trace) for (const [i, k] of dupOf) if (k.person) this.trace[t].push([i, k.person.id]);
  }

  /** People who count in this utterance: seen in at least max(10, 20% of) the captured frames, in labelling order. */
  substantial() {
    const need = Math.max(MIN_TRACK_FRAMES, MIN_TRACK_SHARE * this.t);
    // Order of first appearance; people who appear together go left to right in the (mirrored) preview, i.e. larger raw x first.
    const at = (p) => this.seen.get(p.id);
    return this.s.people.filter((p) => (at(p)?.count ?? 0) >= need)
      .sort((a, b) => at(a).first - at(b).first || at(b).firstX - at(a).firstX || a.id - b.id);
  }

  /**
   * At Finish. times = captured_at (ms) per captured frame. Returns
   *   {multi, chosen, by, people: [{id, label, frames, share, eligible, score, x}]}   (by: 'activity' | 'online' | 'manual')
   * multi = two or more people counted; only then is a person chosen here (otherwise the engine keeps the shipped
   * single-person path unchanged). mode: 'activity' = highest score among eligible people, 'online' = the person the
   * online tracker followed for most of the grid, or an id (manual switch).
   */
  select(times, mode = 'activity') {
    const s = this.s, people = this.substantial();
    for (const p of people) if (p.n === null) p.n = s.labels++;           // a label is given when someone first counts
    for (const id of this.created) {                                      // blips that never counted are forgotten
      const p = s.people.find((q) => q.id === id);
      if (p && p.n === null) s.people.splice(s.people.indexOf(p), 1);
    }
    const info = { multi: people.length >= 2, chosen: null, people: [] };
    if (!info.multi) return { ...info, people: people.map((p) => ({ id: p.id, label: labelOf(p.n), frames: this.seen.get(p.id).count, share: null, eligible: true, score: null, x: this.x(p.id) })) };
    const idx = gridIndices(times), G = idx.length;
    let best = null;
    for (const p of people) {
      const fr = this.frames.get(p.id), grid = new Float32Array(G * F);
      let present = 0;
      for (let g = 0; g < G; g++) if (fr[idx[g]]) { grid.set(fr[idx[g]], g * F); present++; }
      const share = present / G, eligible = share >= MIN_VISIBLE;
      const score = eligible ? activityScore(grid, G).score : null;
      const online = idx.reduce((n, i) => n + (this.online[i] === p.id ? 1 : 0), 0);
      info.people.push({ id: p.id, label: labelOf(p.n), frames: this.seen.get(p.id).count, share, eligible, score, x: this.x(p.id), online });
    }
    const ok = info.people.filter((p) => p.eligible);
    if (typeof mode === 'number') { info.chosen = ok.find((p) => p.id === mode)?.id ?? null; info.by = 'manual'; }
    else if (mode === 'online') {
      best = ok.reduce((b, p) => (!b || p.online > b.online ? p : b), null);
      info.chosen = best && best.online > 0 ? best.id : null;
      info.by = 'online';
    }
    if (info.chosen === null && typeof mode !== 'number') {       // 'online' falls back to the ranking when the tracker followed nobody eligible
      best = ok.reduce((b, p) => (!b || p.score > b.score ? p : b), null);
      info.chosen = best ? best.id : null;
      info.by = 'activity';
    }
    return info;
  }

  /** The id of the person with this label (Person A, ...) among those counted in this utterance, or undefined. */
  idOf(label) {
    return this.substantial().find((p) => p.n !== null && labelOf(p.n) === label)?.id;
  }

  /** One person's keypoints per captured frame (all-zero where they were not in view), as runtime.translate takes them. */
  keypointsFor(id) {
    const fr = this.frames.get(id) ?? [];
    return Array.from({ length: this.t }, (_, t) => fr[t] ?? new Float32Array(F));
  }

  /** Mean torso x as a fraction of the frame width (raw camera coordinates; the preview is mirrored), or null. */
  x(id) {
    const r = this.seen.get(id);
    return r && r.sw > 0 ? r.sx / r.count / (r.sw / r.count) : null;
  }
}
