"""G5 fixtures: five validation clips through demo/rgb_features.py (the reference) + the DINOv2 artefact decision.

Writes
  results/live_onnx/fixtures/rgb/<clip>/  frames.rgb.gz (sampled frames, RGB uint8 as rgb_features.py crops them),
      kps.f32 (sampled keypoints), crops.u8.gz (the 224x224 crops, zeros for empty slots), ref.f32 (reference features)
  demo/tests/fixtures/rgb/index.json      clip list, sample indices, crop boxes, file hashes, np sample_indices table
  results/live_onnx/manifest.json         ONLY the "dinov2" key (the export agent owns the rest)

Clips are chosen in results/_local_val_manifest.tsv order by the zero-slot pattern of the stored validation
features (a selector only; the reference is recomputed): no empty slot; an empty signer slot; an empty left-hand
slot; an empty right-hand slot; fewer than 24 sampled frames.

Run: ~/.pyenv/versions/3.14.3/bin/python demo/tests/fixtures/make_rgb_fixtures.py
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from demo.rgb_features import DINO, DINO_REVISION, IMN, RGBFeatures, crop_box, get_crop  # noqa: E402

R = ROOT / 'results'
OUT = R / 'live_onnx' / 'fixtures' / 'rgb'
IDX = Path(__file__).resolve().parent / 'rgb' / 'index.json'
MANIFEST = R / 'live_onnx' / 'manifest.json'
ONNX_REPO, ONNX_FILE = 'onnx-community/dinov2-small', 'onnx/model.onnx'
ONNX_REVISION = '8b1f705a3a7f6f062f6bdd21986c1583d3ef105d'  # pinned; exported from facebook/dinov2-small
TOL = 1e-4  # "reproduces pooler_output" = max |diff| <= 1e-4 on real crops; otherwise export our own
SLOTS = ((0, 133), (91, 112), (112, 133))  # rgb_features.py: signer, left hand, right hand


def sha256(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pick_clips():
    clips = [Path(l.split('\t')[0]).stem for l in (R / '_local_val_manifest.tsv').read_text().splitlines() if l]
    want = [lambda z, n: n == 24 and not z.any(), lambda z, n: z[:, 0].any(),
            lambda z, n: z[:, 1].any() and not z[:, 0].any(), lambda z, n: z[:, 2].any() and not z[:, 0].any(),
            lambda z, n: n < 24]
    got = []
    for rule in want:
        for cid in clips:
            f = np.load(R / 'rgb_feats_val' / f'rgb_{cid}.npz')['FEATURES']
            if cid not in got and rule(np.abs(f.reshape(len(f), 3, 384)).sum(-1) == 0, len(f)):
                got.append(cid)
                break
    return got


def main():
    ex = RGBFeatures('cpu')
    onnx_rev = ONNX_REVISION
    onnx_path = hf_hub_download(ONNX_REPO, ONNX_FILE, revision=onnx_rev)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    index = dict(clips=[], sample_indices={n: ex.sample_indices(n).tolist() for n in range(1, 301)})
    worst, cos_min, n_crops = 0.0, 1.0, 0
    for cid in pick_clips():
        kps = np.load(R / 'val_clips_dwpose' / f'{cid}.npz')['KPS'].astype(np.float32)
        cap, frames = cv2.VideoCapture(str(R / 'dwpose_experiment/rgb_clips/Signer' / f'{cid}_signer.mp4')), []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        T = min(len(frames), len(kps))
        idx = ex.sample_indices(T)
        bgr, k = [frames[i] for i in idx], kps[idx]
        ref = ex(bgr, k)  # THE reference: demo/rgb_features.py exactly as the live service calls it
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr])
        boxes, crops, slots = [], [], []
        for fr, kk in zip(rgb, k):
            for a, b in SLOTS:
                box = crop_box(kk[a:b, :2], kk[a:b, 2])
                c = get_crop(fr, box)
                boxes.append(None if box is None else [int(v) for v in box])
                slots.append(c is not None)
                crops.append(c if c is not None else np.zeros((224, 224, 3), np.uint8))
        slots = np.array(slots)
        assert ((np.abs(ref.reshape(-1, 384)).sum(1) == 0) == ~slots).all()
        # Artefact check: ORT-python on the same normalised batch vs transformers pooler_output (valid slots).
        x = ((torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2) - ex.mean) / ex.std)
        y = sess.run(None, {'pixel_values': x.numpy()})[0][:, 0][slots]
        r = ref.reshape(-1, 384)[slots]
        worst = max(worst, float(np.abs(y - r).max()))
        cos_min = min(cos_min, float(((y * r).sum(1) / np.linalg.norm(y, axis=1) / np.linalg.norm(r, axis=1)).min()))
        n_crops += int(slots.sum())
        d = OUT / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / 'frames.rgb.gz').write_bytes(gzip.compress(rgb.tobytes(), 1))
        (d / 'crops.u8.gz').write_bytes(gzip.compress(np.stack(crops).tobytes(), 1))
        (d / 'kps.f32').write_bytes(k.astype('<f4').tobytes())
        (d / 'ref.f32').write_bytes(ref.astype('<f4').tobytes())
        index['clips'].append(dict(clip=cid, T=T, idx=idx.tolist(), height=rgb.shape[1], width=rgb.shape[2],
                                   boxes=boxes, slots=slots.tolist(),
                                   sha256={f.name: sha256(f) for f in sorted(d.iterdir())}))
        print(cid, 'T', T, 'n', len(idx), 'empty slots', int((~slots).sum()), flush=True)
    print(f'{ONNX_REPO}@{onnx_rev} vs transformers {DINO}@{DINO_REVISION} pooler_output: '
          f'{n_crops} crops, max|diff| {worst:.3e}, min cos {cos_min:.7f}')
    if worst > TOL:
        sys.exit(f'onnx-community does NOT reproduce pooler_output within {TOL}: export our own (results/live_onnx/dinov2)')
    IDX.parent.mkdir(parents=True, exist_ok=True)
    IDX.write_text(json.dumps(index, separators=(',', ':')) + '\n')
    lfs = {s.rfilename: s.lfs for s in HfApi().model_info(ONNX_REPO, revision=onnx_rev, files_metadata=True).siblings}
    digest = sha256(onnx_path)
    assert digest == lfs[ONNX_FILE].sha256, 'downloaded file differs from the Hub LFS hash'
    dinov2 = dict(hf_model=ONNX_REPO, revision=onnx_rev, path=ONNX_FILE, sha256=digest, bytes=Path(onnx_path).stat().st_size,
                  input='pixel_values', output='last_hidden_state', pool='token 0 (= transformers pooler_output)',
                  source=dict(hf_model=DINO, revision=DINO_REVISION), frames=ex.frames, crop=224,
                  mean=IMN[0], std=IMN[1],
                  parity=dict(reference='transformers AutoModel pooler_output, torch CPU (demo/rgb_features.py)',
                              runtime=f'onnxruntime {ort.__version__} CPU', crops=n_crops, max_abs=worst, min_cos=cos_min))
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m['dinov2'] = dinov2
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2) + '\n')
    print('wrote', IDX, 'and manifest.dinov2')


if __name__ == '__main__':
    main()
