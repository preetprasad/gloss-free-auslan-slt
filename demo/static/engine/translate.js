// Utterance-final translation for the two research operating points in manifest.json: a port of
// src/tools/onnx_reference_decode.py, which mirrors tools/ensemble_mbr_wer.py::ensemble_nbest (n-best beam
// over the uniform mean of the members' log-probs) and demo/service.py::Translator.__call__ (pick the argmax of
// raw / max(1, words) ** lp, confidence -raw / (words + 1), text = detokenize(Vocab.decode(ids))).
// Float32 where the reference is float32: member log-probs summed left to right then divided by the member count,
// candidate = score + logp, each op rounded with Math.fround. Top-k: value descending, then index ascending
// (numpy stable argsort of -cand). Selection and confidence in float64, as in Python.
// ort (onnxruntime-web or onnxruntime-node) and fetchBytes(relPath) -> Uint8Array are injected. Parity gate G3:
// demo/tests/engine_translate.test.mjs.

const f = Math.fround;

async function sha256hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Load one operating point (manifest.models[model]). Every artefact is checked against its manifest sha256 before
 * use, as demo/service.py check_asset does for the checkpoints. sessionOptions go to InferenceSession.create
 * (e.g. { executionProviders: ['webgpu', 'wasm'] }).
 */
export async function loadTranslator({ ort, manifest, model, fetchBytes, sessionOptions = {} }) {
  const spec = manifest.models[model];
  if (!spec) throw new Error(`unknown model ${model}`);
  const members = [];
  let itos = null;
  for (const m of spec.members) {
    const bytes = {};
    for (const k of ['encoder', 'decoder', 'vocab']) {
      bytes[k] = await fetchBytes(m[k]);
      if ((await sha256hex(bytes[k])) !== m.sha256[k]) throw new Error(`checksum mismatch: ${m[k]}`);
    }
    const vocab = JSON.parse(new TextDecoder().decode(bytes.vocab));
    // Vocabulary order differs between recipes; a mismatch is a hard error, never a silent average.
    if (itos && (vocab.length !== itos.length || vocab.some((w, i) => w !== itos[i]))) {
      throw new Error('members must share one vocabulary');
    }
    itos = vocab;
    members.push({
      name: m.name, fusion: m.fusion,
      enc: await ort.InferenceSession.create(bytes.encoder, sessionOptions),
      dec: await ort.InferenceSession.create(bytes.decoder, sessionOptions),
    });
  }
  return new Translator(ort, spec, manifest.decode, members, itos);
}

class Translator {
  constructor(ort, spec, dec, members, itos) {
    Object.assign(this, { ort, members, itos, beam: spec.beam, lp: spec.lp, rgb: spec.rgb });
    Object.assign(this, { maxLen: dec.max_len, pad: dec.pad, bos: dec.bos, eos: dec.eos });
    this.nMembers = f(members.length);
  }

  /** Member.memory for every member. src: Float32Array(T * F); rgb: Float32Array(R * 1152) for fusion members. */
  async encode(src, T, rgb, R) {
    const { Tensor } = this.ort;
    const mems = [];
    for (const m of this.members) {
      const feed = { src: new Tensor('float32', src, [1, T, src.length / T]), src_pad: new Tensor('bool', new Uint8Array(T), [1, T]) };
      if (m.fusion) {
        // No zero-row fallback: a fusion member without RGB would decode a plausible, wrong sentence.
        if (!rgb || !R) throw new Error(`${m.name} needs RGB crop features`);
        feed.rgb = new Tensor('float32', rgb, [1, R, rgb.length / R]);
        feed.rgb_pad = new Tensor('bool', new Uint8Array(R), [1, R]);
      }
      const out = await m.enc.run(feed);
      mems.push(out);
    }
    return mems;
  }

  /** ensemble_nbest for one clip: [[raw, ids], ...] in the order the beams finished. */
  async nbest(mems) {
    const { Tensor } = this.ort, K = this.beam;
    const feeds = mems.map(({ memory, mem_pad }) => {        // broadcast (1, M, D) -> (K, M, D)
      const [, M, D] = memory.dims;
      const md = new Float32Array(K * M * D), pd = new Uint8Array(K * M);
      for (let k = 0; k < K; k++) { md.set(memory.data, k * M * D); pd.set(mem_pad.data, k * M); }
      return { memory: new Tensor('float32', md, [K, M, D]), mem_pad: new Tensor('bool', pd, [K, M]) };
    });
    let ys = Array.from({ length: K }, () => [this.bos]);
    const scores = new Float32Array(K).fill(-Infinity);
    scores[0] = 0;
    const fin = [];
    const topV = new Float64Array(K), topI = new Int32Array(K);
    for (let step = 0; step < this.maxLen - 1; step++) {
      const L = ys[0].length, ysData = new BigInt64Array(K * L);
      ys.forEach((y, k) => y.forEach((tok, j) => { ysData[k * L + j] = BigInt(tok); }));
      const ysT = new Tensor('int64', ysData, [K, L]);
      let logp = null, V = 0;
      for (let i = 0; i < this.members.length; i++) {
        const out = (await this.members[i].dec.run({ ys: ysT, ...feeds[i] })).logp;
        if (!logp) { logp = Float32Array.from(out.data); V = out.dims[1]; }   // weight 1.0: exact
        else for (let j = 0; j < logp.length; j++) logp[j] = f(logp[j] + out.data[j]);
      }
      for (let j = 0; j < logp.length; j++) logp[j] = f(logp[j] / this.nMembers);
      // top-K of cand = scores[b] + logp[b, t] over (b, t), value descending then flat index ascending
      let cnt = 0;
      for (let b = 0; b < K; b++) {
        const sb = scores[b];
        for (let t = 0, j = b * V; t < V; t++, j++) {
          const v = f(sb + logp[j]);
          if (cnt === K && !(v > topV[K - 1])) continue;
          let p = cnt < K ? cnt++ : K - 1;
          while (p > 0 && topV[p - 1] < v) { topV[p] = topV[p - 1]; topI[p] = topI[p - 1]; p--; }
          topV[p] = v; topI[p] = j;
        }
      }
      const next = [];
      for (let i = 0; i < K; i++) {
        const bi = Math.floor(topI[i] / V), ti = topI[i] % V;
        next.push([...ys[bi], ti]);
        scores[i] = topV[i];
        if (ti === this.eos) { fin.push([scores[i], next[i]]); scores[i] = -Infinity; }
      }
      ys = next;
      if (fin.length >= K) break;
    }
    if (!fin.length) for (let i = 0; i < K; i++) if (scores[i] > -Infinity) fin.push([scores[i], ys[i]]);
    return fin;
  }

  /** Translator.__call__. Returns { text, tokenized, confidence, raw, nbest: [[raw, tokenized], ...] }. */
  async translate(src, T, rgb = null, R = 0) {
    const cands = (await this.nbest(await this.encode(src, T, rgb, R))).map(([raw, ids]) => [raw, decodeIds(ids, this)]);
    let best = null, bestKey = -Infinity;
    for (const c of cands) {                                   // Python max(): the first maximal candidate wins
      const key = c[0] / Math.max(1, words(c[1])) ** this.lp;
      if (best === null || key > bestKey) { best = c; bestKey = key; }
    }
    const [raw, seq] = best;
    return { text: detokenize(seq), tokenized: seq, confidence: -raw / (words(seq) + 1), raw, nbest: cands };
  }
}

const words = (s) => s.split(/\s+/).filter(Boolean).length;   // len(str.split())

/** slt/data.py::Vocab.decode (regex tokenization: tokens joined by spaces). */
export function decodeIds(ids, { itos, bos, pad, eos }) {
  const toks = [];
  for (const i of ids) {
    if (i === bos || i === pad) continue;
    if (i === eos) break;
    toks.push(i >= 0 && i < itos.length ? itos[i] : '<unk>');
  }
  return toks.join(' ');
}

/** slt/metrics.py::detokenize. */
export function detokenize(text) {
  return text.replace(/\s+([.,!?;:%)\]}])/g, '$1').replace(/([(\[{])\s+/g, '$1').replace(/\s+'\s*/g, "'")
    .replace(/\s{2,}/g, ' ').trim();
}
