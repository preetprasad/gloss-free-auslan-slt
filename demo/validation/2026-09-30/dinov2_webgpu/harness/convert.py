"""Browser uploads (out/<clip>.{wasm,gpu}.f32 + out/<clip>.json) -> F_wasm/ and F_gpu/ feature directories in the format
tools/ensemble_mbr_wer.py --rgb_feat_dir reads: rgb_<clip>.npz, key FEATURES, float32 (n, 1152), per frame
[signer | left hand | right hand] x 384, empty slots zero (np.savez_compressed, as rgb_recompute_val.py writes).
Every written file is read back through the reader itself (slt.rgb_fusion.RGBDataset._load_feature) and compared.
Also checks G-c: every clip ran under a setup whose webgpu session was created with provider 'webgpu' (and wasm 'wasm').

Interpreter: ./.venv/bin/python (numpy 2.5.3, torch for the reader import).
  convert.py [--out OUT] [--dst DIR]      convert (default out/ -> ./F_wasm, ./F_gpu)
  convert.py --self_test K                 fake uploads from the stored features of the first K manifest clips ->
                                           convert -> reader -> byte-equal (writes under ./selftest/)
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'src'))
from slt.rgb_fusion import RGBDataset  # noqa: E402

R = ROOT / 'results'
D = 1152
CLIPS = [Path(l.split('\t')[0]).stem for l in (R / '_local_val_manifest.tsv').read_text().splitlines() if l.strip()]
READER = SimpleNamespace(cfg=SimpleNamespace(max_src_frames=256))   # Config default; n <= 24 never reaches it


def read_back(path):
    return RGBDataset._load_feature(READER, str(path), D)            # the ensemble tool's own loader


def convert(out, dst):
    setups = {p.stem: json.loads(p.read_text()) for p in out.glob('setup_*.json')}
    arms = {'wasm': dst / 'F_wasm', 'gpu': dst / 'F_gpu'}
    for d in arms.values():
        d.mkdir(parents=True, exist_ok=True)
    done, bad_provider, n_mismatch, zero_diff = [], [], [], []
    for cid in CLIPS:
        meta_p = out / f'{cid}.json'
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        su = setups.get(meta['setup'], {}).get('created', {})
        if su.get('webgpu', {}).get('provider') != 'webgpu' or su.get('wasm', {}).get('provider') != 'wasm':
            bad_provider.append(cid)
        stored = np.load(R / 'rgb_feats_val' / f'rgb_{cid}.npz')['FEATURES']
        for arm, d in arms.items():
            a = np.fromfile(out / f'{cid}.{arm}.f32', '<f4')
            assert a.size == meta['n'] * D, (cid, arm, a.size, meta['n'])
            a = a.reshape(meta['n'], D)
            assert np.isfinite(a).all(), (cid, arm)
            p = d / f'rgb_{cid}.npz'
            np.savez_compressed(p, FEATURES=a)
            back = read_back(p)
            assert back is not None and back.dtype == np.float32 and back.tobytes() == a.tobytes(), (cid, arm)
        if meta['n'] != len(stored):
            n_mismatch.append(cid)
        else:
            z = lambda x: (np.abs(x.reshape(-1, 384)).sum(1) == 0)                      # noqa: E731
            if (z(stored) != z(np.fromfile(out / f'{cid}.wasm.f32', '<f4'))).any():
                zero_diff.append(cid)
        done.append(cid)
    print(f'converted {len(done)}/{len(CLIPS)} clips -> {arms["wasm"]}, {arms["gpu"]}')
    print(f'G-c provider check: {len(done) - len(bad_provider)}/{len(done)} clips ran with webgpu=webgpu, wasm=wasm'
          + (f'  FAIL: {bad_provider[:5]}' if bad_provider else ''))
    print(f'vs stored Kaya features: frame count differs in {len(n_mismatch)} clips, empty-slot pattern differs in '
          f'{len(zero_diff)} (information only)')
    missing = [c for c in CLIPS if c not in set(done)]
    if missing:
        print(f'MISSING {len(missing)} clips (e.g. {missing[:3]}): do not decode until 800/800 -- a missing file makes '
              f'RGBDataset feed a zero frame silently')
    return done, bad_provider


def self_test(k):
    """Fake uploads (stored features stand in for both arms) -> convert -> byte-equal through the reader."""
    st = HERE / 'selftest'
    out = st / 'out'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'setup_0.json').write_text(json.dumps({'id': 'setup_0', 'created': {
        'webgpu': {'provider': 'webgpu', 'ok': True}, 'wasm': {'provider': 'wasm', 'ok': True}}}))
    for cid in CLIPS[:k]:
        f = np.load(R / 'rgb_feats_val' / f'rgb_{cid}.npz')['FEATURES'].astype('<f4')
        (out / f'{cid}.wasm.f32').write_bytes(f.tobytes())
        (out / f'{cid}.gpu.f32').write_bytes(f.tobytes())
        (out / f'{cid}.json').write_text(json.dumps({'clip': cid, 'setup': 'setup_0', 'n': len(f)}))
    done, bad = convert(out, st)
    assert len(done) == k and not bad
    for cid in CLIPS[:k]:
        want = np.load(R / 'rgb_feats_val' / f'rgb_{cid}.npz')['FEATURES']
        for arm in ('F_wasm', 'F_gpu'):
            assert read_back(st / arm / f'rgb_{cid}.npz').tobytes() == want.tobytes()
    print(f'convert self-test: {k} clips round-trip byte-equal through RGBDataset._load_feature -> {st}/F_wasm, F_gpu')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE / 'out'))
    ap.add_argument('--dst', default=str(HERE))
    ap.add_argument('--self_test', type=int, default=0)
    a = ap.parse_args()
    self_test(a.self_test) if a.self_test else convert(Path(a.out), Path(a.dst))
