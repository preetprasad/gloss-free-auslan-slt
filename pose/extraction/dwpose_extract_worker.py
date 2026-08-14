"""One DWPose extraction worker (shardable, resumable, thread-capped).
Forces onnxruntime to ORT_THREADS intra-op threads so N workers * ORT_THREADS ~
core count (no oversubscription). Writes (T,133,3) COCO-WholeBody pose + dummy BBOX
to dwpose_clips/<clip>.npz. Skips clips already done.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys, glob, argparse, numpy as np, cv2
cv2.setNumThreads(1)

import onnxruntime as ort
_ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))
_orig_session = ort.InferenceSession
def _capped_session(*a, **k):
    so = k.get("sess_options") or ort.SessionOptions()
    so.intra_op_num_threads = _ORT_THREADS
    so.inter_op_num_threads = 1
    k["sess_options"] = so
    return _orig_session(*a, **k)
ort.InferenceSession = _capped_session      # cap every session rtmlib creates

from rtmlib import Wholebody

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--mode", default="balanced")
ap.add_argument("--device", default=os.environ.get("DWPOSE_DEVICE", "cpu"))  # cpu | cuda (Kaya GPU)
a = ap.parse_args()

RGB = os.environ.get("RGB_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/rgb_clips"))
OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/dwpose_clips"))
os.makedirs(OUT, exist_ok=True)
mp4s = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
mine = [v for i, v in enumerate(mp4s) if i % a.nshards == a.shard]
model = Wholebody(mode=a.mode, backend="onnxruntime", device=a.device)

# Clip-id derivation: Auslan-Daily = "<clip>_signer.mp4"; MM-WLAuslan = "<id>_kf_rgb.mp4".
# CLIP_SUFFIX lets the same worker serve both without a fork (labels key on bare id).
_SUFFIX = os.environ.get("CLIP_SUFFIX", "_signer.mp4")
done = skip = 0
for v in mine:
    cid = os.path.basename(v)
    cid = cid[:-len(_SUFFIX)] if cid.endswith(_SUFFIX) else os.path.splitext(cid)[0]
    outp = os.path.join(OUT, cid + ".npz")
    if os.path.exists(outp):
        skip += 1
        continue
    cap = cv2.VideoCapture(v)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        try:
            kp, sc = model(fr)
        except Exception:
            frames.append(np.zeros((133, 3), np.float32)); continue
        if sc is None or len(sc) == 0:
            frames.append(np.zeros((133, 3), np.float32)); continue
        k = np.asarray(kp[0], np.float32)
        s = np.asarray(sc[0], np.float32)[:, None]
        frames.append(np.concatenate([k, s], axis=1))
    cap.release()
    if not frames:
        frames = [np.zeros((133, 3), np.float32)]
    arr = np.stack(frames).astype(np.float32)
    bbox = np.zeros((arr.shape[0], 4), np.float32)
    tmp = os.path.join(OUT, cid + ".writing")
    np.savez_compressed(tmp, KPS=arr, BBOX=bbox)
    os.replace(tmp + ".npz", outp)
    done += 1
    if done % 25 == 0:
        print(f"[shard {a.shard}] done={done} skip={skip} / {len(mine)}", flush=True)
print(f"[shard {a.shard}] FINISHED done={done} skip={skip}", flush=True)
