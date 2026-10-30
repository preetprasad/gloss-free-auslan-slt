# Which browser capture setting keeps live DWPose closest to the stored training-pipeline pose?
# Per condition: median keypoint offset from stored KPS (confident body+hand points, in shoulder widths,
# after rescaling to native pixels) and whether the single-best decode matches the stored-pose decode.
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from demo.app import Settings
from demo.service import ModelService
R = './results'
svc = ModelService(Settings(mode='model', models=('single',)))
tr = svc.translators['single']
clips = [l.split('\t')[0].split('/')[-1][:-4] for l in open(f'{R}/_local_val_manifest.tsv')][:int(sys.argv[1])]
CONDS = {'native': (None, None), 'native_q90': (None, 90), 'max960_q75 (current)': (960, 75), 'max960_q90': (960, 90), 'max1280_q90': (1280, 90)}
stats = {c: dict(off=[], same=0) for c in CONDS}
for cid in clips:
    cap = cv2.VideoCapture(f'{R}/dwpose_experiment/rgb_clips/Signer/{cid}_signer.mp4'); raw = []
    while True:
        ok, f = cap.read()
        if not ok: break
        raw.append(f)
    stored = np.load(f'{R}/val_clips_dwpose/{cid}.npz')['KPS'].astype(np.float32)
    ref = tr(stored).text
    for c, (mx, q) in CONDS.items():
        track, kps = {}, []
        for f in raw:
            h, w = f.shape[:2]; s = 1.0
            if mx: s = min(1, mx / w, (mx * 3 / 4) / h)
            g = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else f
            if q: g = cv2.imdecode(cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
            k = svc.extract(g, track)[0].copy(); k[:, :2] /= s; kps.append(k)
        live = np.stack(kps)
        T = min(len(live), len(stored))
        a, b = live[:T], stored[:T]
        sw = np.linalg.norm(b[:, 5, :2] - b[:, 6, :2], axis=1)[:, None] + 1e-6
        m = (a[:, :, 2] > .3) & (b[:, :, 2] > .3)
        m[:, 23:91] = False  # body + hands only
        d = np.linalg.norm(a[:, :, :2] - b[:, :, :2], axis=2) / sw
        stats[c]['off'].append(float(np.median(d[m])))
        stats[c]['same'] += tr(live).text == ref
    print(cid, {c: round(stats[c]['off'][-1], 4) for c in CONDS}, flush=True)
for c, s in stats.items():
    print(f'{c:22s} median offset {np.median(s["off"]):.4f} sw | single decode identical {s["same"]}/{len(clips)}')
