"""One-off: derive the data files of this package from local saved outputs (not run by make).
Per-clip hand speed uses tools/hand_visibility_cv.py::hand_stats unchanged."""
import shutil, sys, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = HERE / "data"
sys.path.insert(0, str(ROOT / "src/tools"))
from hand_visibility_cv import hand_stats

R = ROOT / "results"
FIRST = R / "dwpose_experiment/dwpose_clips"      # first-detection keypoints
TRACK = R / "nmm_probe/tracked_train"             # tracked keypoints (same clips)

# 1. hand speed per clip, both versions, clips present in both
rows = []
for f in sorted(os.listdir(TRACK)):
    if not (FIRST / f).exists():
        continue
    a = hand_stats(np.load(FIRST / f)["KPS"].astype(np.float32))
    b = hand_stats(np.load(TRACK / f)["KPS"].astype(np.float32))
    if a and b:
        rows.append((f[:-4], np.load(FIRST / f)["KPS"].shape[0], a["hand_speed"], b["hand_speed"]))
pd.DataFrame(rows, columns=["clip", "frames", "speed_first", "speed_tracked"]).to_csv(D / "hand_speed_591.csv", index=False)

# 2. byte copies
shutil.copy(R / "dwpose_experiment/dwpose_conf.csv", D / "dwpose_conf.csv")
shutil.copy(R / "checks/dataset_statistics/data/dwpose_train_clip_ids.txt", D / "dwpose_train_clip_ids.txt")
shutil.copy(R / "probe_signerfix_2026-10-25/pred_dwpose.csv", D / "pred_dwpose.csv")
shutil.copy(R / "probe_signerfix_2026-10-25/spans45.csv", D / "spans45.csv")

# 3. hand confidence per annotated span (DWPose keypoints of the 361 probe clips)
sp = pd.read_csv(D / "spans45.csv")
sp["cid"] = sp.Video_Name_Clip.str.replace("_signer.mp4", "", regex=False)
out = []
for r in sp.itertuples():
    k = np.load(R / f"probe_signerfix_2026-10-25/dwpose_clips/{r.cid}.npz")["KPS"].astype(np.float32)
    b, e = max(0, int(r.Begin_Frame)), min(len(k), max(int(r.Begin_Frame) + 3, int(r.End_Frame)))
    s = hand_stats(k[b:e])
    if s:
        out.append((f"{r.cid}:{b}:{e}", s["hand_conf"]))
pd.DataFrame(out, columns=["span", "hand_conf"]).to_csv(D / "span_hand_conf.csv", index=False)

# 4. per-frame hand confidence of the figure clip (confidences only), and the keypoint tensor shape
import cv2
clip = "video_11_6"
ap = np.load(ROOT / "full_run/clips" / f"{clip}.npz")["KPS"]
dw = np.load(R / "dwpose_experiment/dwpose_clips" / f"{clip}.npz")["KPS"]
cap = cv2.VideoCapture(str(R / f"dwpose_experiment/rgb_clips/Signer/{clip}_signer.mp4")); nfr = 0
while cap.read()[0]:
    nfr += 1
T = min(nfr, ap.shape[0])
pd.DataFrame({"frame": range(T), "alphapose_hand_conf": ap[:T, 94:136, 2].mean(1), "dwpose_hand_conf": dw[:T, 91:133, 2].mean(1)}) \
    .to_csv(D / "figure_clip_frame_hand_conf.csv", index=False)
(D / "kps_shape.txt").write_text(f"{dw.shape[1]} {dw.shape[2]}\n")
print("done", len(rows), len(out), T, dw.shape)
