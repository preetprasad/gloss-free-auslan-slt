"""Multi-clip DWPose validation: run RTMW-DW-x on a real sample of extracted signer
clips and compare hand-keypoint confidence to the shipped AlphaPose tensors."""
import os, glob, json, numpy as np, cv2, random
from rtmlib import Wholebody

RGB = os.path.expanduser("~/Honours/results/dwpose_experiment/rgb_clips")
CLIPS = os.path.expanduser("~/Honours/full_run/clips")
N = 25

mp4s = glob.glob(RGB + "/**/*.mp4", recursive=True)
pairs = []
for v in mp4s:
    cid = os.path.basename(v).replace("_signer.mp4", "")
    npz = os.path.join(CLIPS, cid + ".npz")
    if os.path.exists(npz):
        pairs.append((cid, v, npz))
random.Random(0).shuffle(pairs)
pairs = pairs[:N]
print(f"validating {len(pairs)} clips that have both RGB and shipped pose", flush=True)

model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
rows = []
for cid, v, npz in pairs:
    k = np.load(npz)["KPS"].astype(np.float32)
    ship = float(k[:, 94:136, 2].mean())
    cap = cv2.VideoCapture(v); hc = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        kp, sc = model(fr)
        if sc is not None and len(sc):
            hc.append(float(np.mean(np.asarray(sc[0])[91:133])))
    cap.release()
    dw = float(np.mean(hc)) if hc else 0.0
    rows.append({"clip": cid, "shipped": round(ship, 3), "dwpose": round(dw, 3)})
    print(f"  {cid}: shipped {ship:.3f} -> dwpose {dw:.3f}", flush=True)

ship = np.array([r["shipped"] for r in rows]); dw = np.array([r["dwpose"] for r in rows])
S = {"n": len(rows),
     "shipped_hand_conf_mean": round(float(ship.mean()), 3),
     "dwpose_hand_conf_mean": round(float(dw.mean()), 3),
     "mean_per_clip_ratio": round(float((dw / np.maximum(ship, 1e-3)).mean()), 2),
     "dwpose_higher_in_pct_of_clips": round(float((dw > ship).mean()) * 100, 1)}
json.dump({"summary": S, "per_clip": rows},
          open(os.path.expanduser("~/Honours/results/dwpose_experiment/validate_multi.json"), "w"),
          indent=2)
print(json.dumps(S, indent=2))
