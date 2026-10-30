// G3: translate.js (onnxruntime-node) vs src/tools/onnx_reference_decode.py (ORT-python + numpy beam) on
// the 60 fixture validation clips, for both operating points. Gate: text identical on every clip and
// |conf_js - conf_python| < 1e-4. Also reported: tokenized/n-best agreement, max |raw diff|, agreement with the
// PyTorch hypothesis, and decode time per clip.
// Fixtures: demo/tests/fixtures/translator/expected.json (export stage) + results/live_onnx/fixtures/translator/*.bin;
// text.json from demo/tests/fixtures/make_feature_fixtures.py.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import ort from 'onnxruntime-node';
import { decodeIds, detokenize, loadTranslator } from '../static/engine/translate.js';

const ROOT = new URL('../../', import.meta.url).pathname;
const LIVE = join(ROOT, 'results/live_onnx');
const manifest = JSON.parse(readFileSync(join(LIVE, 'manifest.json')));
const expected = JSON.parse(readFileSync(new URL('./fixtures/translator/expected.json', import.meta.url)));
const text = JSON.parse(readFileSync(new URL('./fixtures/features/text.json', import.meta.url)));
const CONF_TOL = 1e-4;

function blob(name) {
  const { path, sha256, floats } = expected.files[name];
  const buf = readFileSync(join(ROOT, path));
  assert.equal(createHash('sha256').update(buf).digest('hex'), sha256, `${path} changed`);
  const a = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  assert.equal(a.length, floats);
  return a;
}

test('detokenize == slt/metrics.py detokenize', () => {
  for (const [s, want] of text.detokenize) assert.equal(detokenize(s), want, JSON.stringify(s));
});

test('max(1, words) ** lp == Python float pow, bit for bit', () => {
  for (const [lp, table] of Object.entries(text.pow)) {
    table.forEach((want, w) => assert.ok(Object.is(Math.max(1, w) ** Number(lp), want), `lp ${lp} w ${w}`));
  }
});

test('Vocab.decode: skips bos/pad, stops at eos, <unk> out of range', () => {
  const v = { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'hello', '.'], pad: 0, bos: 1, eos: 2 };
  assert.equal(decodeIds([1, 4, 0, 5, 9, 2, 4], v), 'hello . <unk>');
  assert.equal(decodeIds([1, 2], v), '');
});

for (const model of ['single', 'ensemble']) {
  test(`G3 ${model}: JS decode == Python reference on ${expected.clips.length} validation clips`, async (t) => {
    const src = blob('src'), rgb = blob('rgb');
    const tr = await loadTranslator({
      ort, manifest, model, fetchBytes: (p) => readFile(join(LIVE, p)),
      sessionOptions: { executionProviders: ['cpu'] },
    });
    let sameText = 0, sameTok = 0, sameNbest = 0, samePytorch = 0, maxConf = 0, maxRaw = 0, ms = 0;
    const bad = [];
    for (const c of expected.clips) {
      const want = c[model];
      const s = src.slice(c.src_offset, c.src_offset + c.T * c.F);
      const r = manifest.models[model].rgb ? rgb.slice(c.rgb_offset, c.rgb_offset + c.R * c.rgb_dim) : null;
      const t0 = performance.now();
      const got = await tr.translate(s, c.T, r, c.R);
      ms += performance.now() - t0;
      sameText += got.text === want.text;
      sameTok += got.tokenized === want.tokenized;
      samePytorch += got.tokenized === want.pytorch_hyp;
      const dConf = Math.abs(got.confidence - want.confidence);
      maxConf = Math.max(maxConf, dConf);
      if (got.nbest.length === want.nbest.length && got.nbest.every(([, s2], i) => s2 === want.nbest[i][1])) {
        sameNbest++;
        got.nbest.forEach(([raw], i) => { maxRaw = Math.max(maxRaw, Math.abs(raw - want.nbest[i][0])); });
      }
      if (got.text !== want.text || !(dConf < CONF_TOL)) bad.push(`${c.idx} ${c.clip}: "${got.text}" vs "${want.text}" dconf ${dConf}`);
    }
    const n = expected.clips.length;
    t.diagnostic(`${model}: text identical ${sameText}/${n}, tokenized ${sameTok}/${n}, n-best text ${sameNbest}/${n}, ` +
      `PyTorch hypothesis ${samePytorch}/${n}, max|dconf| ${maxConf.toExponential(3)}, n-best max|draw| ` +
      `${maxRaw.toExponential(3)}, ${(ms / n).toFixed(1)} ms/clip (onnxruntime-node ${ort.env.versions?.common ?? ''})`);
    assert.ok(n >= 50);
    assert.deepEqual(bad, []);
  });
}
