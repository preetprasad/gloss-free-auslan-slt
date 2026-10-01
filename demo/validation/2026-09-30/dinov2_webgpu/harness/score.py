"""Scorer for PREREG_DINOV2_WEBGPU_2026-09-30: gates G-a / G-b / G-c, the primary count (rule 1), reported numbers.

Interpreter: ./.venv/bin/python (numpy 2.5.3; slt.metrics).
  score.py commands     print the decode commands (run them from the repo root; ~6 min each at 2 threads)
  score.py score [--py_live DIR]
      G-a  dumps/stored/hyps_LP=0.3.jsonl (stored Kaya features, same command) scores 12.470 on 800 rows
      G-b  F_wasm vs Python-live features, per image, G5 bar (demo/tests/engine_rgb.test.mjs): cosine >= 0.9999 on
           non-empty crops, empty crops in the same places and exactly zero. Python-live = --py_live DIR
           (rgb_recompute_val.py output, all 800) if given, else the 5 G5 fixture clips (ref.f32) -> PARTIAL
      G-c  every uploaded clip ran under a setup whose sessions were created with webgpu / wasm (out/setup_*.json)
      Rule 1: S6 hypotheses from F_gpu == F_wasm on 800/800 clips (rule 2, the 10/10 studio check, is separate)
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.environ.get('DGPU_DIR') or Path(__file__).resolve().parent)   # DGPU_DIR: self-test only
ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'src'))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

R = ROOT / 'results'
CLIPS = [Path(l.split('\t')[0]).stem for l in (R / '_local_val_manifest.tsv').read_text().splitlines() if l.strip()]
REC = {'stored_record': (R / 'final_ensemble_validation_decodes/hyps_LP=0.3.jsonl', 12.470),
       'python_live_record': (R / 'live_demo_2026-09-24/dump_S6_val_live_rgb/hyps_LP=0.3.jsonl', 12.291)}
DUMPS = {k: HERE / 'dumps' / k / 'hyps_LP=0.3.jsonl' for k in ('stored', 'F_wasm', 'F_gpu')}
FEAT = {'stored': R / 'rgb_feats_val', 'F_wasm': HERE / 'F_wasm', 'F_gpu': HERE / 'F_gpu'}
COS_MIN = 0.9999                                        # G5 (engine_rgb.test.mjs COS_MIN)
CKPTS = ','.join(f'results/ckpts/{c}.pt' for c in ('comm_dwpose_ctc_vlplong80', 'comm_dwpose_rgb_fusion', 'comm_dwpose_vlp_ctc',
                                                   'comm_dwpose_ctc_s202', 'comm_dwpose_fus_s101', 'comm_dwpose_fus_s202'))


def decode_cmd(feat_dir, dump_dir):
    """results/live_demo_2026-09-24/README.md's S6 command (only --rgb_feat_dir and --dump_dir change)."""
    return (f'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python src/tools/ensemble_mbr_wer.py --ckpts {CKPTS} '
            f'--num_keypoints 133 --keypoint_subset all --coord_norm frame --device cpu --beam 10 --rgb_feat_dir {feat_dir} '
            f'--rgb_feature_dim 1152 --eval_manifest results/_local_val_manifest.tsv --lps 0.3 --dump_dir {dump_dir}')


def load(path):
    rows = sorted((json.loads(l) for l in open(path, encoding='utf-8')), key=lambda r: r['idx'])
    assert [r['idx'] for r in rows] == list(range(len(rows))), path
    return [r['hyp_beam'] for r in rows], [r['ref'] for r in rows]


def bleu(hyps, refs):
    return corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs], '13a')


def same(a, b):
    return sum(x == y for x, y in zip(a, b))


def feats(d, cid):
    return np.load(Path(d) / f'rgb_{cid}.npz')['FEATURES'].astype(np.float64).reshape(-1, 384)


def per_image(a, b):
    """cosines of rows non-empty in both, and the count of rows empty in one but not the other / not exactly zero."""
    za, zb = ~a.any(1), ~b.any(1)
    ok = ~za & ~zb
    cos = (a[ok] * b[ok]).sum(1) / np.linalg.norm(a[ok], axis=1) / np.linalg.norm(b[ok], axis=1)
    return cos, int((za != zb).sum()), float(np.abs(a - b).max()) if a.size else 0.0


def gate_b(py_live):
    if py_live:
        ids = [c for c in CLIPS if (Path(py_live) / f'rgb_{c}.npz').exists()]
        get_ref = lambda c: feats(py_live, c)                                          # noqa: E731
        scope = f'--py_live {py_live}: {len(ids)}/800 clips'
    else:
        index = json.loads((ROOT / 'demo/tests/fixtures/rgb/index.json').read_text())
        ids = [c['clip'] for c in index['clips']]
        get_ref = lambda c: np.fromfile(R / 'live_onnx/fixtures/rgb' / c / 'ref.f32', '<f4').astype(np.float64).reshape(-1, 384)  # noqa: E731
        scope = f'G5 fixture refs only: {len(ids)} clips'
    cmin, zero_bad, shape_bad, n_img = 1.0, 0, 0, 0
    for c in ids:
        a, r = feats(FEAT['F_wasm'], c), get_ref(c)
        if a.shape != r.shape:
            shape_bad += 1
            continue
        cos, zb, _ = per_image(a, r)
        cmin, zero_bad, n_img = min(cmin, cos.min(initial=1.0)), zero_bad + zb, n_img + len(cos)
    ok = shape_bad == 0 and zero_bad == 0 and cmin >= COS_MIN and len(ids) > 0
    status = ('PASS' if len(ids) == 800 else 'PARTIAL') if ok else 'FAIL'
    print(f'G-b {status}: F_wasm vs Python-live ({scope}); {n_img} non-empty crops, min cos {cmin:.7f} (bar {COS_MIN}), '
          f'empty-slot mismatches {zero_bad}, shape mismatches {shape_bad}')
    return status


def gate_c():
    setups = {p.stem: json.loads(p.read_text()) for p in (HERE / 'out').glob('setup_*.json')}
    metas = [json.loads((HERE / 'out' / f'{c}.json').read_text()) for c in CLIPS if (HERE / 'out' / f'{c}.json').exists()]
    good = [m for m in metas if setups.get(m['setup'], {}).get('created', {}).get('webgpu', {}).get('provider') == 'webgpu'
            and setups[m['setup']]['created'].get('wasm', {}).get('provider') == 'wasm']
    status = 'PASS' if len(good) == len(metas) == 800 else 'FAIL'
    print(f'G-c {status}: {len(good)}/{len(metas)} uploaded clips ran with sessions created on webgpu and wasm '
          f'(setups: {[{k: v.get("provider") for k, v in s["created"].items()} for s in setups.values()]})')
    return status, metas, setups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['commands', 'score'])
    ap.add_argument('--py_live', default='')
    a = ap.parse_args()
    if a.mode == 'commands':
        print('# from .; each ~6 min on 2 threads (20-clip smoke: 8.5 s)')
        for k in ('stored', 'F_wasm', 'F_gpu'):
            print(decode_cmd(FEAT[k], HERE / 'dumps' / k))
        print(f'# optional, for a full G-b (Python-live features for all 800 clips; torch DINOv2 on CPU, long):\n'
              f'~/.pyenv/versions/3.14.3/bin/python results/live_demo_2026-09-24/scripts/rgb_recompute_val.py {HERE / "py_live"} cpu')
        return

    # the BLEU route must reproduce both recorded S6 validation scores before anything new is read
    rec = {}
    for k, (p, want) in REC.items():
        rec[k] = load(p)
        got = bleu(*rec[k])
        assert round(got, 3) == want, (k, got, want)
    print(f'BLEU route check: stored record {REC["stored_record"][1]:.3f} and Python-live record {REC["python_live_record"][1]:.3f} reproduced')

    for k in ('F_wasm', 'F_gpu'):
        n = sum((FEAT[k] / f'rgb_{c}.npz').exists() for c in CLIPS)
        assert n == 800, f'{FEAT[k]} has {n}/800 feature files: a decode over it silently used zero frames'
    d = {k: load(p) for k, p in DUMPS.items()}
    for k, (h, r) in d.items():
        assert len(h) == 800 and r == rec['stored_record'][1], f'{k}: {len(h)} rows or refs differ from the manifest order'

    b_stored = bleu(*d['stored'])
    ga = 'PASS' if round(b_stored, 3) == 12.470 else 'FAIL'
    print(f'G-a {ga}: stored features through the same command -> {b_stored:.3f} (want 12.470); '
          f'identical to the recorded dump {same(d["stored"][0], rec["stored_record"][0])}/800')
    gb = gate_b(a.py_live)
    gc, metas, setups = gate_c()

    hw, hg = d['F_wasm'][0], d['F_gpu'][0]
    n_same = same(hg, hw)
    print(f'\nPRIMARY (rule 1): S6 hypotheses F_gpu == F_wasm on {n_same}/800 clips')
    for k in ('F_wasm', 'F_gpu'):
        print(f'  {k}: BLEU-4 {bleu(*d[k]):.3f}; identical to Python-live dump {same(d[k][0], rec["python_live_record"][0])}/800, '
              f'to stored dump {same(d[k][0], rec["stored_record"][0])}/800')

    cmin, zmis, amax, nimg = 1.0, 0, 0.0, 0
    for c in CLIPS:
        cos, zb, mx = per_image(feats(FEAT['F_gpu'], c), feats(FEAT['F_wasm'], c))
        cmin, zmis, amax, nimg = min(cmin, cos.min(initial=1.0)), zmis + zb, max(amax, mx), nimg + len(cos)
    print(f'  features F_gpu vs F_wasm: {nimg} non-empty crops, min cos {cmin:.9f}, max |diff| {amax:.3e}, '
          f'empty-slot mismatches {zmis}')
    for ep in ('webgpu', 'wasm') if metas else ():
        run = [m['timing'][ep]['run_ms'] for m in metas]
        tot = [m['timing'][ep]['total_ms'] for m in metas]
        q = lambda x, p: float(np.percentile(x, p))                                    # noqa: E731
        print(f'  DINOv2 {ep}: session.run median {statistics.median(run):.0f} ms (p90 {q(run, 90):.0f}), '
              f'rgbFeatures median {statistics.median(tot):.0f} ms, over {len(run)} clips')

    print()
    failed = [g for g, s in (('G-a', ga), ('G-b', gb), ('G-c', gc)) if s == 'FAIL']
    if failed:
        print(f'VERDICT: gate(s) {failed} failed -> analysis stops; nothing is concluded (keep EP.dinov2 = ["wasm"]).')
    else:
        cond = '' if gb == 'PASS' else ' (CONDITIONAL: G-b checked on the 5 fixture clips only; pass --py_live for all 800)'
        met = n_same == 800
        print(f'VERDICT rule 1{cond}: {"MET" if met else "NOT MET"} ({n_same}/800 identical). '
              + ('Rule 2 (10/10 studio end-to-end with DINOv2 on WebGPU) must still pass before adopting '
                 'EP.dinov2 = ["webgpu", "wasm"].' if met else 'Keep EP.dinov2 = ["wasm"]; report the difference and speed.'))


if __name__ == '__main__':
    main()
