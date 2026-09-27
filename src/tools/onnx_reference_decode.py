"""Reference decoder for the in-browser engine: ONNX Runtime (Python) sessions plus a numpy beam search that is a
line-for-line mirror of what demo/static/engine/translate.js must do. No torch in the decode loop.

Mirrors, in order:
  tools/ensemble_mbr_wer.py::ensemble_nbest   n-best beam over the uniform mean of the members' log-probs
  demo/service.py::Translator.__call__        pick argmax raw / max(1, words) ** lp; confidence -raw / (words + 1);
                                              text = detokenize(Vocab.decode(ids))
Arithmetic is float32 in the torch order: sum_m (1.0 * logp_m) left to right, then / float32(n_members); candidate =
score + logp. topk order is value descending, then index ascending (torch leaves exact ties unspecified; the run
counts ties that straddle the top-K cut so their influence is visible). Selection and confidence are float64, as in
Python. The graphs and manifest come from tools/export_live_onnx.py (results/live_onnx/).

Gate G2 (default run): all 800 clips of results/_local_val_manifest.tsv for both operating points, compared with the
recorded PyTorch dumps (hypotheses, full n-best, BLEU-4) and with the MODELS self-CE thresholds.

  python src/tools/onnx_reference_decode.py                    # G2, both models
  python src/tools/onnx_reference_decode.py --fixtures 60      # + JS translator fixtures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
from config import Config                                                    # noqa: E402
from slt.data import load_pose, normalize_clip, read_manifest                # noqa: E402  (features, not decode)
from slt.metrics import corpus_bleu4, detokenize                             # noqa: E402

OUT = ROOT / 'results' / 'live_onnx'
DUMPS = {'single': ROOT / 'results/live_demo_2026-09-24/dump_w4A_s202_val',
         'ensemble': ROOT / 'results/final_ensemble_validation_decodes'}
FIXTURE_JSON = ROOT / 'demo/tests/fixtures/translator/expected.json'
FIXTURE_BIN = OUT / 'fixtures' / 'translator'
NEG_INF = np.float32(-np.inf)


def topk_desc(cand: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values: value descending, then index ascending (stable)."""
    return np.argsort(-cand, kind='stable')[:k]


class Translator:
    """One operating point from manifest.json: members' encoder/decoder sessions + the shared vocabulary."""

    def __init__(self, manifest: dict, model: str, providers=('CPUExecutionProvider',), threads: int = 0):
        spec = manifest['models'][model]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads                          # 0 = ORT default
        opts.add_session_config_entry('session.intra_op.allow_spinning', '0')   # polite on a shared machine
        dec = manifest['decode']
        self.beam, self.lp, self.rgb = spec['beam'], spec['lp'], spec['rgb']
        self.max_len, self.pad, self.bos, self.eos = dec['max_len'], dec['pad'], dec['bos'], dec['eos']
        self.members = []
        itos = None
        for m in spec['members']:
            vocab = json.loads((OUT / m['vocab']).read_text())
            assert itos is None or vocab == itos, 'members must share one vocabulary'
            itos = vocab
            self.members.append(dict(name=m['name'], fusion=m['fusion'],
                                     enc=ort.InferenceSession(str(OUT / m['encoder']), opts, providers=list(providers)),
                                     dec=ort.InferenceSession(str(OUT / m['decoder']), opts, providers=list(providers))))
        self.itos = itos
        self.weights = [np.float32(1.0)] * len(self.members)        # ensemble_nbest(weights=None): uniform
        self.wsum = np.float32(sum(1.0 for _ in self.members))
        self.boundary_ties = 0

    # -- encoder: Member.memory per member ------------------------------------------------------------------------
    def encode(self, src: np.ndarray, rgb: np.ndarray | None):
        mems = []
        for m in self.members:
            feed = {'src': src[None], 'src_pad': np.zeros((1, src.shape[0]), bool)}
            if m['fusion']:
                # As translate.js: no zero-row fallback (demo/service.py has one, unreachable because RGB models
                # always compute features); a fusion member without RGB would decode a plausible, wrong sentence.
                if rgb is None or not len(rgb):
                    raise ValueError(f"{m['name']} needs RGB crop features")
                feed |= {'rgb': rgb[None], 'rgb_pad': np.zeros((1, rgb.shape[0]), bool)}
            memory, mem_pad = m['enc'].run(None, feed)
            mems.append((memory, mem_pad))
        return mems

    # -- ensemble_nbest for one clip --------------------------------------------------------------------------------
    def nbest(self, mems):
        K = self.beam
        mem_b = [(np.ascontiguousarray(np.broadcast_to(mm, (K,) + mm.shape[1:])),
                  np.ascontiguousarray(np.broadcast_to(pp, (K, pp.shape[1])))) for mm, pp in mems]
        ys = np.full((K, 1), self.bos, np.int64)
        scores = np.full(K, NEG_INF, np.float32)
        scores[0] = 0.0
        fin = []
        for _ in range(self.max_len - 1):
            lp_sum = None
            for w, m, (mm, pp) in zip(self.weights, self.members, mem_b):
                (step,) = m['dec'].run(None, {'ys': ys, 'memory': mm, 'mem_pad': pp})
                l = w * step
                lp_sum = l if lp_sum is None else lp_sum + l
            logp = lp_sum / self.wsum
            V = logp.shape[-1]
            cand = (scores[:, None] + logp).reshape(-1)
            idx = topk_desc(cand, K)
            if cand.size > K:
                cut = np.partition(cand, cand.size - K - 1)[cand.size - K - 1]   # (K+1)-th largest
                self.boundary_ties += bool(np.isfinite(cut) and cut == cand[idx[-1]])
            bi, ti = idx // V, idx % V
            ys = np.concatenate([ys[bi], ti[:, None]], 1)
            scores = cand[idx].copy()
            for i in range(K):
                if ti[i] == self.eos:
                    fin.append((float(scores[i]), ys[i].tolist()))
                    scores[i] = NEG_INF
            if len(fin) >= K:
                break
        if not fin:                                     # nothing hit EOS -> take current beams
            for i in range(K):
                if scores[i] > NEG_INF:
                    fin.append((float(scores[i]), ys[i].tolist()))
        return fin

    # -- Vocab.decode ---------------------------------------------------------------------------------------------
    def decode_ids(self, ids) -> str:
        toks = []
        for i in ids:
            if i in (self.bos, self.pad):
                continue
            if i == self.eos:
                break
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else '<unk>')
        return ' '.join(toks)

    # -- Translator.__call__ --------------------------------------------------------------------------------------
    def __call__(self, src: np.ndarray, rgb: np.ndarray | None = None) -> dict:
        cands = [(raw, self.decode_ids(seq)) for raw, seq in self.nbest(self.encode(src, rgb))]
        raw, seq = max(cands, key=lambda c: c[0] / (max(1, len(c[1].split())) ** self.lp))
        return dict(text=detokenize(seq), tokenized=seq, confidence=-raw / (len(seq.split()) + 1), raw=raw,
                    nbest=[[r, t] for r, t in cands])


def preprocess_cfg(manifest: dict) -> Config:
    """normalize_clip's Config from the manifest's preprocess section alone (proves the section is sufficient)."""
    fields = Config().to_dict().keys()
    return Config(**{k: v for k, v in manifest['preprocess'].items() if k in fields})


def load_rgb(clip: str, rgb_dir: Path, cfg: Config) -> np.ndarray:
    """slt/rgb_fusion.py::RGBDataset._load_rgb (no face bank): FEATURES, capped at max_src_frames. Its one-zero-row
    fallback for a missing or invalid file is NOT mirrored: it raises instead (all 800 validation files exist, so
    G2 is unaffected; a silent zero row is how fusion members once ran blind, the project log 2026-09-10m)."""
    path = rgb_dir / f"rgb_{Path(clip).name.removesuffix('.npz')}.npz"
    with np.load(path) as z:
        f = np.asarray(z['FEATURES'], dtype=np.float32)
    if not (f.ndim == 2 and f.shape[1] == cfg.rgb_feature_dim and f.shape[0] >= 1 and np.isfinite(f).all()):
        raise ValueError(f'{path}: invalid RGB features {f.shape}')
    if f.shape[0] > cfg.max_src_frames:
        f = f[np.linspace(0, f.shape[0] - 1, cfg.max_src_frames).round().astype(int)]
    return f


def run_model(manifest, model, rows, rgb_dir, cfg, out_dir, threads=0):
    tr = Translator(manifest, model, threads=threads)
    dump = Path(DUMPS[model])
    lp = manifest['models'][model]['lp']
    ref_nbest = [json.loads(l) for l in open(dump / 'nbest.jsonl')]
    ref_hyps = [json.loads(l) for l in open(dump / f'hyps_LP={lp}.jsonl')]
    assert len(ref_nbest) == len(ref_hyps) >= len(rows)
    results, t0 = [], time.perf_counter()
    same_hyp = same_nbest_text = 0
    raw_max = 0.0
    for i, (clip, ref) in enumerate(rows):
        assert ref_hyps[i]['ref'] == ref == ref_nbest[i]['ref'], f'row {i}: dump does not follow the manifest'
        kps, bbox = load_pose(clip)
        src, _ = normalize_clip(kps, bbox, cfg)
        rgb = load_rgb(clip, rgb_dir, cfg) if tr.rgb else None
        r = tr(src, rgb)
        r |= dict(idx=i, clip=Path(clip).name.removesuffix('.npz'), ref=ref, pytorch_hyp=ref_hyps[i]['hyp_beam'])
        results.append(r)
        same_hyp += r['tokenized'] == ref_hyps[i]['hyp_beam']
        want = ref_nbest[i]['nbest']
        if [t for _, t in r['nbest']] == [t for _, t in want]:
            same_nbest_text += 1
            raw_max = max(raw_max, max(abs(a - b) for (a, _), (b, _) in zip(r['nbest'], want)))
        if (i + 1) % 100 == 0:
            print(f'  [{model}] {i + 1}/{len(rows)}  identical so far {same_hyp}', flush=True)
    secs = time.perf_counter() - t0
    bleu = corpus_bleu4([r['text'] for r in results], [detokenize(r['ref']) for r in results], tokenize='13a')
    selfce = np.array([r['confidence'] for r in results])
    tau = {q: float(np.quantile(selfce, float(q))) for q in manifest['models'][model]['tau']}
    tau_diff = {q: abs(tau[q] - v) for q, v in manifest['models'][model]['tau'].items()}
    summary = dict(model=model, clips=len(rows), bleu4=bleu, expected_bleu4=manifest['models'][model]['evidence']
                   ['validation_bleu4'], hyps_identical=same_hyp, nbest_text_identical=same_nbest_text,
                   nbest_raw_max_abs_diff=raw_max, boundary_ties=tr.boundary_ties, tau=tau,
                   tau_expected=manifest['models'][model]['tau'], tau_max_abs_diff=max(tau_diff.values()),
                   seconds=round(secs, 1), ms_per_clip=round(1000 * secs / len(rows), 1), threads=threads,
                   onnxruntime=ort.__version__, reference_dump=str(dump.relative_to(ROOT)))
    d = out_dir / model
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'nbest.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(dict(idx=r['idx'], ref=r['ref'], nbest=r['nbest'])) + '\n')
    with open(d / f'hyps_LP={lp}.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(dict(idx=r['idx'], ref=r['ref'], hyp_beam=r['tokenized'])) + '\n')
    (d / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"[G2] {model}: BLEU-4 {bleu:.3f} (expected {summary['expected_bleu4']}), hypotheses identical "
          f"{same_hyp}/{len(rows)}, n-best text identical {same_nbest_text}/{len(rows)}, n-best raw max|d| "
          f"{raw_max:.2e}, boundary ties {tr.boundary_ties}, tau {tau} max|d| {summary['tau_max_abs_diff']:.2e}, "
          f"{summary['ms_per_clip']} ms/clip", flush=True)
    return summary, results


def write_fixtures(n, rows, cfg, rgb_dir, per_model):
    """Small JSON (committed) + float32 little-endian binaries (gitignored) for the JS translator tests."""
    FIXTURE_BIN.mkdir(parents=True, exist_ok=True)
    FIXTURE_JSON.parent.mkdir(parents=True, exist_ok=True)
    src_parts, rgb_parts, clips = [], [], []
    so = ro = 0
    for i, (clip, ref) in enumerate(rows[:n]):
        kps, bbox = load_pose(clip)
        src, _ = normalize_clip(kps, bbox, cfg)
        rgb = load_rgb(clip, rgb_dir, cfg)
        src_parts.append(src.astype('<f4').ravel())
        rgb_parts.append(rgb.astype('<f4').ravel())
        entry = dict(idx=i, clip=Path(clip).name.removesuffix('.npz'), ref=ref,
                     src_offset=so, T=int(src.shape[0]), F=int(src.shape[1]),
                     rgb_offset=ro, R=int(rgb.shape[0]), rgb_dim=int(rgb.shape[1]))
        so += src.size
        ro += rgb.size
        for model, results in per_model.items():
            r = results[i]
            entry[model] = {k: r[k] for k in ('text', 'tokenized', 'confidence', 'raw', 'nbest', 'pytorch_hyp')}
        clips.append(entry)
    blobs = {'src': np.concatenate(src_parts), 'rgb': np.concatenate(rgb_parts)}
    files = {}
    for k, arr in blobs.items():
        p = FIXTURE_BIN / f'{k}.bin'   # *.bin is gitignored repo-wide
        p.write_bytes(arr.tobytes())
        files[k] = dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        floats=int(arr.size))
    FIXTURE_JSON.write_text(json.dumps(dict(
        note='Expected outputs of src/tools/onnx_reference_decode.py (ORT-python + numpy beam) on the '
             'first clips of results/_local_val_manifest.tsv. src = normalize_clip features (T x F float32, row-major), '
             'rgb = stored DINOv2 crop features (R x rgb_dim) as the ensemble reads them. Offsets count float32 '
             'elements into the binaries (little-endian, gitignored, regenerate with --fixtures).',
        manifest='results/live_onnx/manifest.json', files=files, clips=clips), indent=1) + '\n')
    print(f'[fixtures] {len(clips)} clips -> {FIXTURE_JSON.relative_to(ROOT)} '
          f'({FIXTURE_JSON.stat().st_size / 1e6:.2f} MB) + {FIXTURE_BIN.relative_to(ROOT)} '
          f"({sum(b.nbytes for b in blobs.values()) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='single,ensemble')
    ap.add_argument('--manifest_tsv', default=str(ROOT / 'results/_local_val_manifest.tsv'))
    ap.add_argument('--rgb_dir', default=str(ROOT / 'results/rgb_feats_val'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fixtures', type=int, default=0, help='also write JS translator fixtures for the first N clips')
    ap.add_argument('--out_dir', default=str(OUT / 'g2'))
    ap.add_argument('--threads', type=int, default=0, help='ORT intra-op threads (0 = ORT default)')
    a = ap.parse_args()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    cfg = preprocess_cfg(manifest)
    rows = read_manifest(a.manifest_tsv)
    rows = rows[:a.limit] if a.limit else rows
    out_dir = Path(a.out_dir)
    ok, per_model, summaries = True, {}, {}
    for model in a.models.split(','):
        s, res = run_model(manifest, model, rows, Path(a.rgb_dir), cfg, out_dir, a.threads)
        summaries[model], per_model[model] = s, res
        if not a.limit:
            ok &= (round(s['bleu4'], 3) == s['expected_bleu4'] and s['hyps_identical'] == len(rows)
                   and s['tau_max_abs_diff'] <= 1e-4)
    (out_dir / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if a.fixtures:
        write_fixtures(a.fixtures, rows, cfg, Path(a.rgb_dir), per_model)
    if not a.limit:
        print('[G2]', 'PASS' if ok else 'FAIL', flush=True)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
