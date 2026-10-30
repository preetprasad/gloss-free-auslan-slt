// G4 — JS DWPose (demo/static/engine/pose.js, onnxruntime-node + the cv.js kernels) vs the installed rtmlib as run by
// demo/service.py ModelService.extract, on browser-like JPEG frames of validation signer videos.
// Fixtures: demo/tests/fixtures/make_pose_fixtures.py (manifest committed; frames and intermediates under
// results/live_onnx/fixtures/pose, gitignored). Skips when those or the ONNX models are not on this machine.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import * as ort from 'onnxruntime-node';
import { resize, bgrFromRGBA, rgbFromRGBA, warpAffine, getAffineTransform, fma } from '../static/engine/cv.js';
import { createPose, DWPose, K } from '../static/engine/pose.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const M = JSON.parse(readFileSync(new URL('./fixtures/pose/manifest.json', import.meta.url)));
const LARGE = path.join(ROOT, M.large_dir);
const MODEL_FILES = { detector: 'pose/yolox_m.onnx', pose: 'pose/rtmw_256x192.onnx' };
const modelPath = (k) => [path.join(ROOT, 'results/live_onnx', MODEL_FILES[k]), path.join(homedir(), '.cache/rtmlib/hub/checkpoints', M.models[k].file)]
  .find(existsSync);
const ready = existsSync(LARGE) && modelPath('detector') && modelPath('pose');
const skip = ready ? false : `needs ${M.large_dir} and the rtmlib ONNX models (run make_pose_fixtures.py)`;

const bytes = (name) => { const b = readFileSync(path.join(LARGE, name)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
const f32 = (u8) => new Float32Array(u8.slice().buffer);
const b64 = (s, T) => new T(Uint8Array.from(Buffer.from(s, 'base64')).buffer);
const img = (e) => ({ data: bytes(`${e.name}.bgr.bin`), width: e.width, height: e.height, channels: 3 });
const sha = (u8) => createHash('sha256').update(u8).digest('hex');

function diff(a, b) {                     // count of unequal elements and max |a-b|
  assert.equal(a.length, b.length);
  let n = 0, mx = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { n++; mx = Math.max(mx, Math.abs(a[i] - b[i])); }
  return { n, mx };
}
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : 0; };

function mulberry32(seed) {             // same generator as make_pose_fixtures.py and engine_track.test.mjs
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let pose;
async function getPose() {
  for (const k of ['detector', 'pose']) assert.equal(sha(readFileSync(modelPath(k))), M.models[k].sha256, `${k} ONNX differs from the one the fixtures used`);
  return (pose ??= await createPose({ ort, detector: modelPath('detector'), pose: modelPath('pose') }));
}

test('solver parity: JS getAffineTransform (FMA LU port) == native cv2 on 300 random point sets; exact fma', async () => {
  const r = mulberry32(M.fma.seed), out = new Float64Array(M.fma.n);
  for (let i = 0; i < M.fma.n; i++) {
    const a = (r() - 0.5) * 2 ** (Math.trunc(r() * 60) - 30);
    const b = (r() - 0.5) * 2 ** (Math.trunc(r() * 60) - 30);
    const c = i % 2 ? -(a * b) * (1 + (r() - 0.5) * 2 ** -Math.trunc(r() * 60)) : (r() - 0.5) * 2 ** (Math.trunc(r() * 80) - 40);
    out[i] = fma(a, b, c);
  }
  assert.deepEqual([...out.subarray(0, 4)], M.fma.first);
  assert.equal(sha(new Uint8Array(out.buffer)), M.fma.sha256, 'fma emulation differs from math.fma');
  const src = b64(M.affine.src, Float32Array), dst = b64(M.affine.dst, Float32Array), py = b64(M.affine.M, Float64Array);
  const n = py.length / 6;
  let port = 0;
  for (let i = 0; i < n; i++) {
    const s = src.subarray(6 * i, 6 * i + 6), d = dst.subarray(6 * i, 6 * i + 6), m = py.subarray(6 * i, 6 * i + 6);
    port += diff(getAffineTransform(s, d), m).n > 0;
  }
  console.log(`getAffineTransform vs native cv2 ${M.versions.cv2}: FMA port ${n - port}/${n} identical`);
  assert.equal(port, 0);
});

// cv2.resize INTER_LINEAR on uint8 3-channel images: 2,549 (source, target) sizes run through native cv2 4.13 by
// fixtures/make_resize_fixtures.py — rtmlib letterboxes of frames up to 1280 x 960, get_crop boxes from 1 x 1 to
// 1280 x 960 -> 224 x 224, general sizes and corner cases (copies, 1 x N, exact 2x = INTER_AREA inside cv2, other
// integer factors). The images are cut from a SHA-256 byte pool the test rebuilds; each output is compared by hash.
test('resize fuzz: JS resize bit-identical to native cv2 on 2,549 size pairs', () => {
  const R = JSON.parse(readFileSync(new URL('./fixtures/resize/fuzz.json', import.meta.url)));
  const blocks = [];
  for (let i = 0; i < R.pool.bytes / 32; i++) {
    const n = Buffer.alloc(4); n.writeUInt32LE(i);
    blocks.push(createHash('sha256').update(Buffer.concat([Buffer.from(R.pool.tag), n])).digest());
  }
  const pool = new Uint8Array(Buffer.concat(blocks));
  assert.equal(sha(pool), R.pool.sha256, 'byte pool differs from the one the fixture used');
  const tally = {}, bad = [];
  for (const [kind, W, H, w, h, off, want] of R.cases) {
    const t = (tally[kind] ??= { cases: 0, unequal: 0 });
    t.cases++;
    const out = resize({ data: pool.subarray(off, off + W * H * 3), width: W, height: H, channels: 3 }, w, h);
    if (out.data.length !== w * h * 3 || sha(out.data).slice(0, 16) !== want) { t.unequal++; bad.push([kind, W, H, w, h]); }
  }
  console.table(tally);
  console.log(`resize vs native cv2 ${R.versions.cv2} (${R.versions.machine}): ${R.cases.length - bad.length}/${R.cases.length} identical`);
  assert.ok(tally.letterbox.cases + tally.crop.cases + tally.general.cases >= 2000);
  assert.deepEqual(bad, [], 'resize outputs differ from native cv2');
  assert.throws(() => resize({ data: pool.subarray(0, 12), width: 2, height: 2 }, 0, 5), /empty size/);
});

test('kernel parity: the cv.js resize / channel-order / warpAffine / getAffineTransform ports == native cv2 on 20 real frames', { skip }, async () => {
  const p = await getPose();
  const tally = {};
  const add = (op, d) => { const t = (tally[op] ??= { cases: 0, unequal: 0, maxAbs: 0 }); t.cases++; t.unequal += d.n; t.maxAbs = Math.max(t.maxAbs, d.mx); };
  const byName = Object.fromEntries(M.frames.map((e) => [e.name, e]));
  for (const k of M.kernel) {
    const e = byName[k.name], im = img(e);
    assert.equal(sha(im.data), e.bgr_sha256);
    add('resize 224x224 (downscale)', diff(resize(im, 224, 224).data, bytes(k.r224)));
    const [x0, y0, w, h] = k.up_region;
    const region = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) region.set(im.data.subarray(((y0 + y) * im.width + x0) * 3, ((y0 + y) * im.width + x0 + w) * 3), y * w * 3);
    add('resize 50x70 -> 224x224 (upscale)', diff(resize({ data: region, width: w, height: h }, 224, 224).data, bytes(k.up224)));
    add('resize letterbox (YOLOX)', diff(p.letterbox(im).padded, bytes(e.letterbox)));
    // The frame as the browser decodes it (RGBA); native cv2 gives its RGB (BGR2RGB) and back its BGR (RGBA2BGR).
    const rgba = new Uint8Array(im.width * im.height * 4);
    for (let i = 0; i < im.width * im.height; i++) { rgba[4 * i] = im.data[3 * i + 2]; rgba[4 * i + 1] = im.data[3 * i + 1]; rgba[4 * i + 2] = im.data[3 * i]; rgba[4 * i + 3] = 255; }
    const frame = { data: rgba, width: im.width, height: im.height };
    add('RGBA -> RGB (rgb_features.py BGR2RGB)', diff(rgbFromRGBA(frame).data, bytes(k.rgb)));
    add('RGBA -> BGR (cvtColor RGBA2BGR)', diff(bgrFromRGBA(frame).data, im.data));
    const boxes = [...e.persons.map((q, j) => ({ box: b64(e.boxes, Float32Array).subarray(4 * j, 4 * j + 4), src: q.src, M: q.M, crop: q.crop })),
      ...k.boxes.map((q) => ({ box: b64(q.box, Float32Array), src: q.src, M: q.M, crop: q.crop }))];
    for (const q of boxes) {
      const td = p.topDown(im, q.box);
      add('top-down src points (float32)', diff(td.src, b64(q.src, Float32Array)));
      add('getAffineTransform, FMA port (float64)', diff(td.M, b64(q.M, Float64Array)));
      add('warpAffine 256x192, port (python matrix)', diff(warpAffine(im, b64(q.M, Float64Array), 192, 256).data, bytes(q.crop)));
      add('warpAffine 256x192 (js pipeline)', diff(td.crop.data, bytes(q.crop)));
    }
  }
  console.table(tally);
  for (const [op, t] of Object.entries(tally)) assert.equal(t.unequal, 0, `${op}: ${t.unequal} elements differ (max |d| ${t.maxAbs})`);
});

// D1: 3,024 boxes on the 36 real frames + the 409-box degenerate set, each run through rtmlib's own preprocessing by
// make_pose_fixtures.py (warp_fuzz.json, pinned by sha256 in the manifest). JS recomputes src, M and the crop from the box.
test('warpAffine fuzz: JS top-down crops bit-identical to native cv2 on 3,433 boxes (real frames + degenerate set) + 10 extreme matrices', { skip }, async () => {
  const W = M.warp_fuzz, body = readFileSync(path.join(LARGE, W.file));
  assert.equal(sha(body), W.sha256, 'warp_fuzz.json differs from the one the manifest pins');
  const fuzz = JSON.parse(body), p = await getPose();
  const byName = Object.fromEntries(M.frames.map((e) => [e.name, e])), imgs = {};
  const rows = [];
  for (const [set, cases] of [['real frames', fuzz.real], ['degenerate', fuzz.degenerate]]) {
    const r = { set, boxes: cases.length, srcUnequal: 0, matrixUnequal: 0, cropUnequal: 0, msPerCrop: 0 };
    for (const c of cases) {
      const im = (imgs[c.frame] ??= img(byName[c.frame]));
      const t0 = performance.now();
      const td = p.topDown(im, Float32Array.from(c.box));
      r.msPerCrop += performance.now() - t0;
      r.srcUnequal += diff(td.src, Float32Array.from(c.src)).n > 0;
      r.matrixUnequal += diff(td.M, Float64Array.from(c.M)).n > 0;
      r.cropUnequal += sha(td.crop.data) !== c.crop_sha256;
    }
    r.msPerCrop = +(r.msPerCrop / cases.length).toFixed(2);
    rows.push(r);
  }
  // Matrices no box produces (NaN, Inf, subnormal determinant, offsets past int32 / int64): cvRound's wrap and saturation.
  const X = M.warp_extreme, xim = img(byName[X.frame]);
  const extremeUnequal = X.cases.filter((c) => sha(warpAffine(xim, b64(c.M, Float64Array), 192, 256).data) !== c.crop_sha256).length;
  console.table([...rows, { set: 'extreme matrices (warpAffine only)', boxes: X.cases.length, cropUnequal: extremeUnequal }]);
  assert.equal(fuzz.real.length, W.real); assert.equal(fuzz.degenerate.length, W.degenerate);
  assert.ok(fuzz.real.length >= 3000);
  for (const r of rows) {
    assert.equal(r.srcUnequal, 0, `${r.set}: top-down src points differ`);
    assert.equal(r.matrixUnequal, 0, `${r.set}: warp matrices differ`);
    assert.equal(r.cropUnequal, 0, `${r.set}: warpAffine crops differ`);
  }
  assert.equal(extremeUnequal, 0, 'extreme matrices: warpAffine crops differ');
});

test('extract: a run that threw is the empty frame plus `error`; no detection is the same frame without it', async () => {
  const img = { data: new Uint8Array(640 * 640 * 3).fill(90), width: 640, height: 640, channels: 3 };   // no letterbox resize
  const session = (outputNames, run) => ({ inputNames: ['x'], outputNames, run });
  const dets = (score) => ({ dets: new ort.Tensor('float32', Float32Array.of(100, 80, 300, 500, score), [1, 1, 5]) });
  const lost = new Error('WebGPU device lost'), throws = async () => { throw lost; };
  const track = { mid: [320, 200], wid: 90 };
  const cases = [
    ['detector threw', new DWPose(ort, session(['dets'], throws), null), lost],
    ['pose model threw', new DWPose(ort, session(['dets'], async () => dets(0.9)), session(['simcc_x', 'simcc_y'], throws)), lost],
    ['nobody detected', new DWPose(ort, session(['dets'], async () => dets(0.2)), null), undefined],
  ];
  for (const [name, p, error] of cases) {
    const r = await p.extract(img, track);
    assert.equal(r.count, 0, name);
    assert.deepEqual(r.keypoints, new Float32Array(K * 3), name);   // the frame service.py produces either way
    assert.equal(r.error, error, name);
    assert.deepEqual(Object.keys(r).sort(), error ? ['count', 'error', 'keypoints'] : ['count', 'keypoints'], name);
  }
  assert.deepEqual(track, { mid: [320, 200], wid: 90 });   // the tracker state is untouched
});

test('G4 stage parity: detector and pose model on identical inputs', { skip }, async () => {
  const p = await getPose();
  const r = { detInputUnequal: 0, detsMaxAbs: 0, detsShapeMismatch: 0, boxesFromPyDets: 0, boxesMaxAbs: 0,
    poseInputUnequal: 0, ortPoseMaxAbs: 0, simccMaxAbs: 0, decodeFromPySimcc: 0, kpMaxAbs: 0, scMaxAbs: 0, persons: 0 };
  for (const e of M.frames) {
    const im = img(e);
    r.detInputUnequal += diff(p.letterbox(im).padded, bytes(e.letterbox)).n;
    const { boxes, dets } = await p.detect(im);
    if (dets.dims.join() !== e.dets_shape.join()) { r.detsShapeMismatch++; continue; }
    r.detsMaxAbs = Math.max(r.detsMaxAbs, diff(dets.data, f32(bytes(e.dets))).mx);
    const pyBoxes = b64(e.boxes, Float32Array);
    const ratio = Math.min(640 / e.height, 640 / e.width);
    r.boxesFromPyDets += diff(p.decodeDets(f32(bytes(e.dets)), e.dets_shape, ratio), pyBoxes).n;   // pure port check
    r.boxesMaxAbs = Math.max(r.boxesMaxAbs, diff(boxes, pyBoxes).mx);
    if (!e.count) continue;
    const pyKp = f32(bytes(e.kp)), pySc = f32(bytes(e.sc));
    const kp2 = new Float32Array(pyKp.length), sc2 = new Float32Array(pySc.length);
    const est = await p.estimate(im, pyBoxes);                     // pose on Python's boxes: isolates the pose stage
    for (const [j, q] of est.persons.entries()) {
      const py = e.persons[j], pin = f32(bytes(py.input));
      r.poseInputUnequal += diff(q.input, pin).n;
      const sx = f32(bytes(py.simx)), sy = f32(bytes(py.simy));
      const o = await p.est.run({ [p.est.inputNames[0]]: new ort.Tensor('float32', pin, [1, 3, 256, 192]) });   // Python's exact tensor
      r.ortPoseMaxAbs = Math.max(r.ortPoseMaxAbs, diff(o[p.est.outputNames[0]].data, sx).mx, diff(o[p.est.outputNames[1]].data, sy).mx);
      r.simccMaxAbs = Math.max(r.simccMaxAbs, diff(q.simx.data, sx).mx, diff(q.simy.data, sy).mx);
      p.decodeSimcc(sx, sy, py.simx_shape[2], py.simy_shape[2], q.center, q.scale, kp2, sc2, j * K);  // pure port check
      r.persons++;
    }
    r.decodeFromPySimcc += diff(kp2, pyKp).n + diff(sc2, pySc).n;
    r.kpMaxAbs = Math.max(r.kpMaxAbs, diff(est.kp, pyKp).mx);
    r.scMaxAbs = Math.max(r.scMaxAbs, diff(est.sc, pySc).mx);
  }
  console.table(r);
  assert.equal(r.detsShapeMismatch, 0);
  assert.equal(r.detInputUnequal, 0);
  assert.equal(r.boxesFromPyDets, 0, 'YOLOX postprocess port differs on identical detector output');
  assert.equal(r.poseInputUnequal, 0, 'RTMPose input tensor differs on identical boxes');
  assert.equal(r.decodeFromPySimcc, 0, 'SimCC decode port differs on identical model output');
});

test('G4 end to end: extract() with the tracker, JS vs service.py, per clip in capture order', { skip }, async () => {
  const p = await getPose();
  const tracks = {}, dxy = [], dsc = [], rows = [];
  let countMismatch = 0, selMismatch = 0, trackMismatch = 0, ms = 0;
  for (const e of M.frames) {
    const track = (tracks[e.clip] ??= {});
    const t0 = performance.now();
    const { keypoints, count } = await p.extract(img(e), track);
    ms += performance.now() - t0;
    const py = b64(e.extract, Float32Array);
    if (count !== e.count) countMismatch++;
    let mxy = 0, msc = 0;
    for (let k = 0; k < K; k++) {
      mxy = Math.max(mxy, Math.abs(keypoints[3 * k] - py[3 * k]), Math.abs(keypoints[3 * k + 1] - py[3 * k + 1]));
      msc = Math.max(msc, Math.abs(keypoints[3 * k + 2] - py[3 * k + 2]));
    }
    if (e.count && diff(keypoints, py).n && mxy > 0) {
      // which Python detection is the JS output closest to? (tracker picked a different person?)
      const pk = f32(bytes(e.kp)), n = e.kp_shape[0];
      let best = -1, bd = Infinity;
      for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < K * 2; k++) s += Math.abs(pk[j * K * 2 + k] - keypoints[3 * (k >> 1) + (k & 1)]); if (s < bd) { bd = s; best = j; } }
      if (best !== e.selected) selMismatch++;
    }
    const tr = track.mid ? [track.mid[0], track.mid[1], track.wid] : null;
    if (JSON.stringify(tr) !== JSON.stringify(e.track_after)) trackMismatch++;
    dxy.push(mxy); dsc.push(msc);
    rows.push({ frame: e.name, count, py_count: e.count, max_dxy: mxy, max_dscore: msc });
  }
  console.table(rows);
  const summary = { frames: M.frames.length, multiPerson: M.frames.filter((e) => e.count >= 2).length, countMismatch, selMismatch, trackMismatch,
    maxDxy: Math.max(...dxy), medianDxy: median(dxy), maxDscore: Math.max(...dsc), medianDscore: median(dsc), msPerFrame: +(ms / M.frames.length).toFixed(1) };
  console.table(summary);
  assert.equal(countMismatch, 0, 'detection counts differ');
  assert.equal(selMismatch, 0, 'the tracker followed a different person');
  assert.equal(trackMismatch, 0, 'tracker state differs');
  assert.equal(summary.maxDxy, 0, 'keypoints differ');       // measured exact on this machine (ORT-node 1.30 vs ORT-python 1.28)
  assert.equal(summary.maxDscore, 0, 'scores differ');
});
