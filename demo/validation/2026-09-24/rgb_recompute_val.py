# Recompute validation RGB features through demo/rgb_features.py for the S6 decode gate.
import sys, glob, os, numpy as np, cv2, time
sys.path.insert(0, '.')
from demo.rgb_features import RGBFeatures
R = './results'; OUT = sys.argv[1]; dev = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ex = RGBFeatures(dev); t0 = time.time(); n = 0
for p in sorted(glob.glob(f'{R}/rgb_feats_val/rgb_*.npz')):
    cid = os.path.basename(p)[4:-4]; out = f'{OUT}/rgb_{cid}.npz'
    if os.path.exists(out): continue
    kps = np.load(f'{R}/val_clips_dwpose/{cid}.npz')['KPS'].astype(np.float32)
    cap = cv2.VideoCapture(f'{R}/dwpose_experiment/rgb_clips/Signer/{cid}_signer.mp4'); fr = []
    while True:
        r, f = cap.read()
        if not r: break
        fr.append(f)
    idx = ex.sample_indices(min(len(fr), len(kps)))
    np.savez_compressed(out, FEATURES=ex([fr[i] for i in idx], kps[idx])); n += 1
    if n % 100 == 0: print(n, f'{(time.time()-t0)/n:.2f}s/clip', flush=True)
print('DONE', n, f'{(time.time()-t0)/max(n,1):.2f}s/clip')
