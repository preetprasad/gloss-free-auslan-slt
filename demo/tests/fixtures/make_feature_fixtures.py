"""Fixtures for demo/static/engine/features.js (normalize_clip + the service's 25 fps preamble) and the two pieces of
translate.js text handling that do not need a model (detokenize, the length-penalty power).

Writes
  results/live_onnx/fixtures/features/<case>.in.bin   float32 LE keypoints: T x 133 x 3 ('clip'), or one row of
                                                      133 x 3 per captured frame ('capture'); *.bin is gitignored
  results/live_onnx/fixtures/features/<case>.out.bin  float32 LE normalize_clip features, T' x 399
  demo/tests/fixtures/features/index.json             cases, capture times, expected grid indices, errors, hashes
  demo/tests/fixtures/features/text.json              detokenize cases + Python's float(w) ** lp for both LPs

References are the Python code itself, never a retyping of it:
  'clip'    slt/data.py::normalize_clip with the Config demo/service.py::Translator builds (asserted equal to
            manifest.json "preprocess"), called as Translator.__call__ calls it (float32 KPS, all-zero BBOX)
  'capture' demo/service.py::ModelService.translate run unbound on a stub whose translator records the keypoints it
            is handed (grid_indices, stack, visibility check), then the same normalize_clip line
Cases: seven validation clips (first six + the longest, T=250); a synthetic clip probing the float32 1e-6
zero-frame threshold and zero-variance frames; a 12 s irregular capture with dropped detections whose resampled
clip exceeds max_src_frames (exercises the linspace subsample); a short capture with exact grid ties; and the two
error paths (every frame empty; signer visible in < 80% of the grid).

Run: ~/.pyenv/versions/3.14.3/bin/python demo/tests/fixtures/make_feature_fixtures.py
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from config import Config                                            # noqa: E402
from slt.data import load_pose, normalize_clip, read_manifest        # noqa: E402
from slt.metrics import detokenize                                   # noqa: E402
from demo.service import MODELS, ModelService, grid_indices         # noqa: E402

assert np.__version__ == '1.26.3', np.__version__
OUT = ROOT / 'results' / 'live_onnx' / 'fixtures' / 'features'
HERE = Path(__file__).resolve().parent / 'features'
MANIFEST = json.loads((ROOT / 'results' / 'live_onnx' / 'manifest.json').read_text())
# demo/service.py::Translator.__init__ (device only affects torch placement)
CFG = Config(num_keypoints=133, keypoint_subset='all', coord_norm='frame', rgb_feature_dim=1152)


def check_cfg():
    pre, fields = MANIFEST['preprocess'], CFG.to_dict()
    for k, v in pre.items():
        if k == 'feature_dim':
            assert CFG.feature_dim() == v, (k, CFG.feature_dim(), v)
        elif k in fields:
            assert fields[k] == v, (k, fields[k], v)
    assert not CFG.hierarchical_pose_norm and CFG.temporal_pool == 1 and not CFG.use_velocity


def live_normalize(kps):
    """demo/service.py::Translator.__call__, first line."""
    return normalize_clip(kps.astype(np.float32), np.zeros((len(kps), 4), np.float32), CFG)


def service_preamble(keypoints, times):
    """ModelService.translate up to the translator call; returns (grid idx, stacked kps) or raises its ValueError."""
    seen = {}

    def record(kps, rgb):
        seen['kps'] = kps
        return SimpleNamespace()

    stub = SimpleNamespace(translators={'single': record})
    assert not MODELS['single']['rgb']
    ModelService.translate(stub, 'single', keypoints, times)
    return grid_indices(times), seen['kps']


def write(name, arr):
    OUT.mkdir(parents=True, exist_ok=True)
    b = np.ascontiguousarray(arr, dtype='<f4').tobytes()
    (OUT / name).write_bytes(b)
    return dict(file=name, sha256=hashlib.sha256(b).hexdigest(), floats=len(b) // 4)


def clip_case(name, kps, note):
    feats, mask = live_normalize(kps)
    assert mask.all() and feats.dtype == np.float32
    return dict(name=name, kind='clip', note=note, T=int(len(kps)), T_out=int(len(feats)),
                input=write(f'{name}.in.bin', kps), output=write(f'{name}.out.bin', feats))


def capture_case(name, keypoints, times, note):
    idx, kps = service_preamble(keypoints, [float(t) for t in times])
    feats, _ = live_normalize(kps)
    return dict(name=name, kind='capture', note=note, frames=len(keypoints), times=[float(t) for t in times],
                idx=[int(i) for i in idx], T=int(len(kps)), T_out=int(len(feats)),
                input=write(f'{name}.in.bin', np.stack(keypoints)), output=write(f'{name}.out.bin', feats))


def error_case(name, keypoints, times, note):
    try:
        if times is None:
            live_normalize(np.stack(keypoints))
        else:
            service_preamble(keypoints, [float(t) for t in times])
    except ValueError as e:
        msg = str(e)
    else:
        raise AssertionError(f'{name}: expected ValueError')
    return dict(name=name, kind='error', note=note, frames=len(keypoints),
                times=None if times is None else [float(t) for t in times],
                input=write(f'{name}.in.bin', np.stack(keypoints)), error=msg.split('.')[0] + '.')


def main():
    check_cfg()
    rows = read_manifest(str(ROOT / 'results' / '_local_val_manifest.tsv'))
    clips = [load_pose(p)[0].astype(np.float32) for p, _ in rows]
    longest = int(np.argmax([len(k) for k in clips]))
    cases = []
    for i in list(range(6)) + [longest]:
        cases.append(clip_case(f'val{i:03d}_{Path(rows[i][0]).stem}', clips[i], f'validation clip {i}'))

    # Zero-frame threshold (numpy: float32 sum > float32(1e-6)) and zero-variance frames.
    base = clips[0][:12].copy()
    base[1] = 0                                               # all zero -> dropped
    base[2] = 0; base[2, 7, 2] = np.float32(1e-6)             # sum == float32(1e-6) -> not > -> dropped
    base[3] = 0; base[3, 7, 2] = np.nextafter(np.float32(1e-6), np.float32(1))   # just above -> kept, xy std 0
    base[4] = 0; base[4, 50, 0] = -3e-7; base[4, 60, 1] = 4e-7   # |.| sum 7e-7 -> dropped
    base[5, :, :2] = 321.5                                    # constant xy -> std 0 -> (x - m) / 1e-6 = 0
    base[6, :, 2] = 0                                         # zero confidences, coordinates remain
    base[7] = 0; base[7, :, 2] = 0.5                          # confidence only -> kept, xy all 0
    cases.append(clip_case('edge_frames', base, 'zero-frame threshold at float32(1e-6); zero-variance frames'))

    # A 12 s capture at an irregular ~20 fps with 8% failed detections (zero rows): the 25 fps grid gives 300
    # frames, ~92% visible, so > 256 valid frames reach the linspace subsample.
    rng = np.random.default_rng(20260925)
    src = clips[longest]
    n = 240
    times = np.cumsum(np.r_[0.0, rng.uniform(35.0, 65.0, n - 1)])
    times = 1_727_000_000_000.0 + times * (12_000.0 / times[-1])      # epoch-ms scale, as captured_at arrives
    frames = [src[min(len(src) - 1, round(c * (len(src) - 1) / (n - 1)))].copy() for c in range(n)]
    for c in rng.choice(n, int(0.08 * n), replace=False):
        frames[c] = np.zeros((133, 3), np.float32)
    frames = [f + np.float32(rng.normal(0, 0.25)) * (f != 0) for f in frames]
    cases.append(capture_case('capture_12s', [f.astype(np.float32) for f in frames], times,
                              '12 s irregular capture, 8% dropped detections, resampled clip > 256 frames'))

    # Exact ties on the 40 ms grid (nearest frame: first index wins) and a repeated timestamp.
    t = [1000.0, 1020.0, 1060.0, 1060.0, 1100.0, 1140.0, 1180.0, 1200.0, 1240.0, 1300.0]
    cases.append(capture_case('capture_ties', [clips[longest][j] for j in range(len(t))], t,
                              'grid ties and a duplicate timestamp'))

    cases.append(error_case('all_empty', [np.zeros((133, 3), np.float32)] * 5, None,
                            'normalize_clip: every frame invalid'))
    few = [clips[longest][j] if j % 3 == 0 else np.zeros((133, 3), np.float32) for j in range(20)]
    cases.append(error_case('mostly_empty', few, [2000.0 + 40.0 * j for j in range(20)],
                            'service: signer visible in < 80% of the grid'))

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'index.json').write_text(json.dumps(dict(
        purpose='features.js parity fixtures; regenerate with make_feature_fixtures.py',
        versions=dict(python=sys.version.split()[0], numpy=np.__version__),
        preprocess=MANIFEST['preprocess'], bin_dir='results/live_onnx/fixtures/features', cases=cases), indent=1) + '\n')

    # translate.js text handling: detokenize and the selection denominator max(1, words) ** lp.
    texts = ['hello .', 'that is great .', 'what ?', "i do n't know ' s it", 'wow ! really ? yes ; no : ok , fine .',
             '( a ) [ b ] { c } 50 %', '  spaced   out  .  ', "it ' s", '', '.', 'a  ,  b']
    lps = sorted({MODELS[m]['lp'] for m in MODELS})
    (HERE / 'text.json').write_text(json.dumps(dict(
        detokenize=[[s, detokenize(s)] for s in texts],
        pow={str(lp): [max(1, w) ** lp for w in range(0, 65)] for lp in lps},
        pow_note='entry w = max(1, w) ** lp exactly as Translator.__call__ computes it (Python float pow)'),
        indent=1) + '\n')
    for c in cases:
        print(c['name'], c['kind'], c.get('T'), c.get('T_out'), c.get('error', ''))


if __name__ == '__main__':
    main()
