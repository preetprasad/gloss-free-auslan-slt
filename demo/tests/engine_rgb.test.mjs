// G5: JS RGB features (demo/static/engine/rgb.js) vs demo/rgb_features.py on five validation clips.
// Fixtures: python demo/tests/fixtures/make_rgb_fixtures.py (large arrays under results/live_onnx/fixtures/rgb/).
// Gates: sample indices and crop boxes equal; crops (cv.js resize) bit-identical to cv2; per-crop cosine >= 0.9999 against
// the transformers reference; empty slots in the same places and exactly zero.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { gunzipSync } from 'node:zlib';
import ort from 'onnxruntime-node';
import { cropBox, getCrop, rgbFeatures, sampleIndices } from '../static/engine/rgb.js';

const ROOT = new URL('../../', import.meta.url).pathname;
const FIX = join(ROOT, 'results/live_onnx/fixtures/rgb');
const index = JSON.parse(readFileSync(new URL('./fixtures/rgb/index.json', import.meta.url)));
const spec = JSON.parse(readFileSync(join(ROOT, 'results/live_onnx/manifest.json'))).dinov2;
const COS_MIN = 0.9999;

function load(clip, name) {
  const buf = readFileSync(join(FIX, clip.clip, name));
  assert.equal(createHash('sha256').update(buf).digest('hex'), clip.sha256[name], `${clip.clip}/${name} changed`);
  return name.endsWith('.gz') ? gunzipSync(buf) : buf;
}
const f32 = (b) => new Float32Array(b.buffer, b.byteOffset, b.length / 4);

function modelPath() {
  if (spec.file) return join(ROOT, 'results/live_onnx', spec.file);
  const hub = process.env.HF_HUB_CACHE ?? join(process.env.HF_HOME ?? join(homedir(), '.cache/huggingface'), 'hub');
  return join(hub, `models--${spec.hf_model.replace('/', '--')}`, 'snapshots', spec.revision, spec.path);
}

test('sampleIndices == np.linspace(0, n-1, min(24, n)).round()', () => {
  for (const [n, want] of Object.entries(index.sample_indices)) assert.deepEqual(sampleIndices(Number(n)), want, `n=${n}`);
  for (const c of index.clips) assert.deepEqual(sampleIndices(c.T), c.idx);
});

test('G5: crop boxes, crops and DINOv2 features match demo/rgb_features.py', async (t) => {
  const path = modelPath();
  assert.equal(createHash('sha256').update(readFileSync(path)).digest('hex'), spec.sha256, 'DINOv2 artefact hash');
  const session = await ort.InferenceSession.create(path, { executionProviders: ['cpu'] });
  let worstCos = 1, worstAbs = 0, crops = 0, badBytes = 0;
  for (const clip of index.clips) {
    const n = clip.idx.length, H = clip.height, W = clip.width;
    const rgb = load(clip, 'frames.rgb.gz'), kps = f32(load(clip, 'kps.f32')), ref = f32(load(clip, 'ref.f32'));
    const refCrops = load(clip, 'crops.u8.gz');
    const frames = Array.from({ length: n }, (_, f) => {
      const data = new Uint8ClampedArray(H * W * 4);
      for (let p = 0, o = f * H * W * 3; p < H * W; p++) {
        data[4 * p] = rgb[o + 3 * p]; data[4 * p + 1] = rgb[o + 3 * p + 1]; data[4 * p + 2] = rgb[o + 3 * p + 2]; data[4 * p + 3] = 255;
      }
      return { data, width: W, height: H };
    });
    // Boxes and bit-exact crops (cv.js resize vs native cv2).
    const slots = [[0, 133], [91, 112], [112, 133]];
    let clipBad = 0;
    for (let f = 0; f < n; f++) {
      const frame = { data: rgb.subarray(f * H * W * 3, (f + 1) * H * W * 3), width: W, height: H };
      slots.forEach(([a, b], s) => {
        const i = 3 * f + s, box = cropBox(kps.subarray(f * 399, (f + 1) * 399), a, b);
        assert.deepEqual(box, clip.boxes[i], `${clip.clip} frame ${f} slot ${s} box`);
        const c = getCrop(frame, box);
        assert.equal(c !== null, clip.slots[i], `${clip.clip} frame ${f} slot ${s} empty`);
        if (c) {
          const want = refCrops.subarray(i * 224 * 224 * 3, (i + 1) * 224 * 224 * 3);
          assert.equal(c.length, want.length);
          for (let p = 0; p < want.length; p++) if (c[p] !== want[p]) clipBad++;
        }
      });
    }
    badBytes += clipBad;
    // Features through the full JS path.
    const got = await rgbFeatures(frames, kps, { ort, session });
    assert.equal(got.length, ref.length);
    let cMin = 1, aMax = 0;
    for (let i = 0; i < 3 * n; i++) {
      const g = got.subarray(i * 384, (i + 1) * 384), r = ref.subarray(i * 384, (i + 1) * 384);
      if (!clip.slots[i]) {
        assert.ok(g.every((v) => v === 0) && r.every((v) => v === 0), `${clip.clip} slot ${i} should be zero`);
        continue;
      }
      let dot = 0, gg = 0, rr = 0;
      for (let d = 0; d < 384; d++) { dot += g[d] * r[d]; gg += g[d] * g[d]; rr += r[d] * r[d]; aMax = Math.max(aMax, Math.abs(g[d] - r[d])); }
      cMin = Math.min(cMin, dot / Math.sqrt(gg * rr));
      crops++;
    }
    t.diagnostic(`${clip.clip}: n=${n} empty=${clip.slots.filter((s) => !s).length} crop bytes differing=${clipBad} ` +
      `1-min cos=${(1 - cMin).toExponential(2)} max|diff|=${aMax.toExponential(3)}`);
    worstCos = Math.min(worstCos, cMin); worstAbs = Math.max(worstAbs, aMax);
  }
  t.diagnostic(`G5 overall: ${crops} crops, crop bytes differing=${badBytes}, 1-min cos=${(1 - worstCos).toExponential(2)}, ` +
    `max|diff|=${worstAbs.toExponential(3)} (onnxruntime-node ${ort.env.versions?.common ?? '?'} vs transformers torch CPU)`);
  assert.equal(badBytes, 0, 'crops must be bit-identical to cv2');
  assert.ok(worstCos >= COS_MIN, `min cosine ${worstCos} < ${COS_MIN}`);
});
