"""Run once (not by make): reduce saved raw outputs to per-clip frame counts (numbers only, no text).
Reads: full_run/clips_signer/*.npz (shipped per-clip pose tensors), the alignment and sign-spotting tables, the pose file of the
isolated-sign dictionary. Writes clip_lengths.csv, alignment_frames.csv, spotting_frames.csv, mmwl_lengths.csv."""
import csv, gzip, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T = ROOT/"data/raw/Auslan-Daily/Dataset Split Table"
a = pd.read_excel(T/"Sign Language Alignment/Auslan_Daily_Communication.xlsx")
# independent length per clip: join alignment to translation on (video, subtitle) over pairs unique in both tables
# (a join by row order is wrong). Only numbers are written.
tr = pd.read_excel(T/"Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx")
tr["v"] = tr.Video_Name.str.lower(); a["v"] = a.Video_Name.str.lower()
ut = tr[~tr.duplicated(["v", "Subtitle"], keep=False)]
ua = a[~a.duplicated(["v", "Subtitle"], keep=False)]
j = ut.merge(ua, on=["v", "Subtitle"])
al = dict(zip(j.Video_Clip_Name, (j.End_Frame - j.Begin_Frame)))
sp = dict(zip(tr.Video_Clip_Name, tr.Split))
with open(HERE/"clip_lengths.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip", "split", "frames", "align_len"])
    for p in sorted((ROOT/"full_run/clips_signer").glob("*.npz")):
        w.writerow([p.stem, sp.get(p.stem, ""), np.load(p)["KPS"].shape[0], al.get(p.stem, "")])
def sec(s):
    m, r = str(s).split(":"); return int(m)*60 + float(r)
a["bs"] = a.Begin_Time.map(sec); a["es"] = a.End_Time.map(sec)
a[["Begin_Frame", "End_Frame", "bs", "es"]].to_csv(HERE/"alignment_frames.csv", index=False)
s = pd.read_excel(T/"Sign Spotting/Sign_Spotting.xlsx")
s[["Begin_Frame", "End_Frame"]].to_csv(HERE/"spotting_frames.csv", index=False)
pk = ROOT/"mmwl_pose/pose_cam1_trainval.pkl"
try:
    d = pickle.load(gzip.open(pk, "rb"))
except OSError:
    d = pickle.load(open(pk, "rb"))
with open(HERE/"mmwl_lengths.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "frames"])
    for k, v in d.items(): w.writerow([k, len(v)])
# frame counts of the same dictionary clips as extracted from the VIDEOS (sidecar of the adopted stitched corpus, 8,581 recordings)
import json
src = {}
for l in open(ROOT/"results/stitched_corpus_audit/audit_A.jsonl"):
    for x in json.loads(l)["segments"]: src[x["source_id"]] = x["source_frames_original"]
with open(HERE/"mmwl_video_frames.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "video_frames"])
    for k in sorted(src): w.writerow([k, src[k]])
print("done", len(d), len(src), len(j))
