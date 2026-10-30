"""Cross-estimator hand-keypoint agreement (RTMPose/RTMW family via rtmlib).
Runs several whole-body models on a sample of clips and measures how far apart they
place the 42 hand keypoints, normalised by hand span (scale-invariant). Low disagreement
= the keypoints are consistent across independent models -> accurate, not just confident.
No ground truth needed; cross-model agreement is the proxy for accuracy.
"""
import os, glob, json, random, itertools, argparse
import numpy as np, cv2
os.environ.setdefault("OMP_NUM_THREADS", "2")
import onnxruntime as ort
_o = ort.InferenceSession
def _cap(*a, **k):
    so = k.get("sess_options") or ort.SessionOptions(); so.intra_op_num_threads = 4
    k["sess_options"] = so; return _o(*a, **k)
ort.InferenceSession = _cap
from rtmlib import Wholebody

ap = argparse.ArgumentParser()
ap.add_argument("--n_clips", type=int, default=120)
ap.add_argument("--frames", type=int, default=6)
ap.add_argument("--device", default=os.environ.get("DWPOSE_DEVICE", "cpu"))
a = ap.parse_args()
RGB = os.environ.get("RGB_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/rgb_clips"))
OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment"))

clips = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
random.Random(0).shuffle(clips); clips = clips[:a.n_clips]
print(f"device={a.device}  clips={len(clips)}", flush=True)
models = {
    "RTMW-x(perf)":    Wholebody(mode="performance", backend="onnxruntime", device=a.device),
    "DWPose(bal)":     Wholebody(mode="balanced",    backend="onnxruntime", device=a.device),
    "RTMPose-m(light)":Wholebody(mode="lightweight", backend="onnxruntime", device=a.device),
}
HAND = slice(91, 133)   # 42 whole-body hand keypoints
conf = {m: [] for m in models}
pair = {f"{x}|{y}": [] for x, y in itertools.combinations(models, 2)}

for ci, c in enumerate(clips):
    cap = cv2.VideoCapture(c); frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    if len(frames) < 3: continue
    for i in np.linspace(0, len(frames) - 1, min(a.frames, len(frames))).astype(int):
        fr = frames[i]; kp = {}
        for name, m in models.items():
            k, s = m(fr)
            if k is None or len(k) == 0:
                kp[name] = None; continue
            kp[name] = np.asarray(k[0])[HAND]
            conf[name].append(float(np.asarray(s[0])[HAND].mean()))
        for x, y in itertools.combinations(models, 2):
            if kp[x] is None or kp[y] is None: continue
            span = np.linalg.norm(kp[x].max(0) - kp[x].min(0)) + 1e-6
            pair[f"{x}|{y}"].append(float(np.linalg.norm(kp[x] - kp[y], axis=1).mean() / span))
    if (ci + 1) % 20 == 0:
        print(f"  {ci+1}/{len(clips)}", flush=True)

res = {"n_clips": len(clips),
       "mean_hand_conf": {m: round(float(np.mean(conf[m])), 3) for m in models},
       "pair_disagreement_normed": {k: {"mean": round(float(np.mean(v)), 3),
                                        "median": round(float(np.median(v)), 3), "n": len(v)}
                                    for k, v in pair.items()}}
json.dump(res, open(os.path.join(OUT, "agreement_rtmlib.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
