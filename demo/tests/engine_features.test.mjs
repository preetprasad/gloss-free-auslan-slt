// features.js parity: normalize_clip (live configuration) and the service's 25 fps preamble vs Python.
// Fixtures: python demo/tests/fixtures/make_feature_fixtures.py (binaries under results/live_onnx/fixtures/features/).
// Gate: bit-for-bit float32 equality of every feature (max |diff| is reported whatever it is), identical grid
// indices, identical error paths.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { assertLive, gridIndices, normalizeClip, prepareClip } from '../static/engine/features.js';

const ROOT = new URL('../../', import.meta.url).pathname;
const index = JSON.parse(readFileSync(new URL('./fixtures/features/index.json', import.meta.url)));
const manifest = JSON.parse(readFileSync(join(ROOT, 'results/live_onnx/manifest.json')));

function load({ file, sha256, floats }) {
  const buf = readFileSync(join(ROOT, index.bin_dir, file));
  assert.equal(createHash('sha256').update(buf).digest('hex'), sha256, `${file} changed; regenerate the fixtures`);
  const a = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  assert.equal(a.length, floats);
  return a;
}

function compare(got, want) {
  let maxAbs = 0, diff = 0;
  for (let i = 0; i < want.length; i++) {
    if (Object.is(got[i], want[i])) continue;
    diff++;
    maxAbs = Math.max(maxAbs, Math.abs(got[i] - want[i]));
  }
  return { maxAbs, diff };
}

test('manifest preprocess is the ported live configuration (and matches the fixtures)', () => {
  assertLive(manifest.preprocess);
  assert.deepEqual(index.preprocess, manifest.preprocess);
  assert.throws(() => assertLive({ ...manifest.preprocess, coord_norm: 'bbox' }), /coord_norm=frame/);
});

test('features.js output is exactly the translator fixtures\' src (G3 input) for the shared clips', () => {
  const tr = JSON.parse(readFileSync(new URL('./fixtures/translator/expected.json', import.meta.url)));
  const { path, sha256 } = tr.files.src;
  const buf = readFileSync(join(ROOT, path));
  assert.equal(createHash('sha256').update(buf).digest('hex'), sha256);
  const src = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  let shared = 0;
  for (const c of index.cases.filter((x) => x.kind === 'clip')) {
    const e = tr.clips.find((x) => c.name.endsWith(`_${x.clip}`));
    if (!e) continue;
    const { feats, T } = normalizeClip(load(c.input), c.T);
    assert.equal(T, e.T);
    assert.equal(compare(feats, src.subarray(e.src_offset, e.src_offset + e.T * e.F)).diff, 0, e.clip);
    shared++;
  }
  assert.ok(shared >= 6, `only ${shared} shared clips`);
});

test('normalize_clip + service preamble: bit-for-bit vs Python', (t) => {
  let worst = 0, values = 0, differing = 0;
  for (const c of index.cases) {
    const input = load(c.input);
    if (c.kind === 'error') {
      const frames = Array.from({ length: c.frames }, (_, i) => input.subarray(i * 399, (i + 1) * 399));
      const run = c.times ? () => prepareClip(frames, c.times) : () => normalizeClip(input, c.frames);
      assert.throws(run, (e) => e.message.startsWith(c.error), c.name);
      continue;
    }
    let out;
    if (c.kind === 'clip') {
      out = normalizeClip(input, c.T);
    } else {
      assert.deepEqual(gridIndices(c.times), c.idx, `${c.name}: grid indices`);
      const frames = Array.from({ length: c.frames }, (_, i) => input.subarray(i * 399, (i + 1) * 399));
      out = prepareClip(frames, c.times);
      assert.deepEqual(out.idx, c.idx);
      assert.equal(out.kps.length, c.T * 399);
    }
    assert.equal(out.T, c.T_out, `${c.name}: frames out`);
    const want = load(c.output);
    const { maxAbs, diff } = compare(out.feats, want);
    t.diagnostic(`${c.name}: T ${c.T} -> ${out.T}, differing ${diff}/${want.length}, max|diff| ${maxAbs}`);
    worst = Math.max(worst, maxAbs); values += want.length; differing += diff;
  }
  t.diagnostic(`all cases: ${values} features, ${differing} differ, max|diff| ${worst}`);
  assert.equal(differing, 0, `max|diff| ${worst}`);
});
