"""Compute per-clip DWPose hand/body confidence from the re-extracted npz (runs on Kaya).
COCO-WholeBody-133: body 0-16, left hand 91-111, right hand 112-132."""
import glob, os, csv, numpy as np
CL = "/scratch/pmc099/pprasad/dwpose/dwpose_clips"
OUT = "/scratch/pmc099/pprasad/dwpose/dwpose_conf.csv"
fs = sorted(glob.glob(CL + "/*.npz"))
n = 0
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip", "dwpose_hand_conf", "dwpose_body_conf", "T"])
    for p in fs:
        try:
            k = np.load(p)["KPS"].astype("float32")
            w.writerow([os.path.basename(p)[:-4],
                        round(float(k[:, 91:133, 2].mean()), 4),
                        round(float(k[:, 0:17, 2].mean()), 4), k.shape[0]])
            n += 1
        except Exception:
            pass
print(f"wrote {n} clips -> {OUT}")
