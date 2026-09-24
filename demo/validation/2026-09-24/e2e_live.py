# End-to-end: real ModelService (DWPose + tracker + 25 fps grid + translators + live DINOv2) on validation
# signer videos, frames sent as browser-like JPEG data URLs. Compares with the same models decoded on the
# stored Kaya DWPose (and stored RGB features for S6). Correctness smoke, not a benchmark.
import base64, glob, json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, '.')
from demo.app import Settings
from demo.service import ModelService, MODELS
R = './results'
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
svc = ModelService(Settings(mode='model'))
clips = [l.split('\t')[0].split('/')[-1][:-4] for l in open(f'{R}/_local_val_manifest.tsv')][:n]
refs = {l.split('\t')[0].split('/')[-1][:-4]: l.rstrip('\n').split('\t')[1] for l in open(f'{R}/_local_val_manifest.tsv')}
agree = {m: 0 for m in svc.translators}
for cid in clips:
    cap = cv2.VideoCapture(f'{R}/dwpose_experiment/rgb_clips/Signer/{cid}_signer.mp4'); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; s = min(1, 960 / w, 720 / h)
        f = cv2.resize(f, (round(w * s), round(h * s))) if s < 1 else f
        frames.append('data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]).decode())
    from demo.service import decode_frame
    track, kps, t0 = {}, [], time.perf_counter()
    for d in frames: kps.append(svc.extract(decode_frame(d), track)[0])
    pose_ms = (time.perf_counter() - t0) * 1000 / len(frames)
    times = [i * 40 for i in range(len(frames))]
    stored = np.load(f'{R}/val_clips_dwpose/{cid}.npz')['KPS'].astype(np.float32)
    print(f'{cid}: {len(frames)} frames, pose {pose_ms:.0f} ms/frame | ref: {refs[cid]}')
    for m, tr in svc.translators.items():
        live = svc.translate(m, kps, times, frames)
        feats = np.load(f'{R}/rgb_feats_val/rgb_{cid}.npz')['FEATURES'] if MODELS[m]['rgb'] else None
        off = tr(stored, feats)
        agree[m] += live.text == off.text
        print(f'  {m:8s} live: {live.text!r} ce {live.confidence:.3f} (translate {live.translate_ms:.0f} ms, rgb {live.rgb_ms:.0f} ms)'
              f' | stored-pose: {off.text!r} ce {off.confidence:.3f}')
print('identical to stored-pose decode:', agree, 'of', len(clips))
