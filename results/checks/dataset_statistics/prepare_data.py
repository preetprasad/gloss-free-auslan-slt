"""Prep (run once in the private tree, not by make): reduce the dataset's annotation tables to counts-only files.
No subtitle text is written. Usage: python prepare_data.py   (run from the repository root or anywhere)"""
import glob, json, os, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "data"; OUT.mkdir(exist_ok=True)
DS = ROOT / "data/raw/Auslan-Daily/Dataset Split Table"
TR = ROOT / "data/auslan-daily/translation"

# News translation table: counts only
n = pd.read_excel(TR / "Auslan-Daily_News.xlsx")
news = {"rows": len(n), "videos": int(n.Video_Name.nunique())}
for s in ("train", "dev", "test"):
    x = n[n.Split == s]
    news[s] = {"clips": len(x), "signers": int(x.Signer_ID.astype(str).str.strip().nunique())}
tr = n[n.Split == "train"].Subtitle.astype(str)
news["train_vocab_lower_whitespace"] = len({w for t in tr for w in t.lower().split()})
json.dump(news, open(OUT / "news_counts.json", "w"), indent=1)

# annotation tables: ids / glosses only
sp = pd.read_excel(DS / "Sign Spotting/Sign_Spotting.xlsx")
sp[["Video_Name_Clip", "Query_Gloss", "Split"]].to_csv(OUT / "spotting.csv", index=False)
fs = pd.read_excel(DS / "Fingerspelling Detection/Fingerspelling_Detection.xlsx")
fs[["Video_Clip_Name", "split"]].to_csv(OUT / "fingerspelling.csv", index=False)
iso = pd.read_excel(DS / "Isolated Sign Language Recognition/Isolated_Sign_Language_Recognition.xlsx")
iso[["Gloss", "Split"]].to_csv(OUT / "isolated.csv", index=False)

# MM-WLAuslan labels: clips per gloss
B = ROOT / "data/raw/MM-WLAuslan/Annotation/Labels & Split"
rows = {}
for sp_name in ("Train", "Valid"):
    for _id, g in json.load(open(B / f"{sp_name}.json")).items():
        rows.setdefault(g, {"Train": 0, "Valid": 0})[sp_name] += 1
pd.DataFrame([(g, c["Train"], c["Valid"]) for g, c in sorted(rows.items())],
             columns=["gloss", "train_clips", "valid_clips"]).to_csv(OUT / "mmwl_gloss_counts.csv", index=False)

# clips of the DWPose training list (clip ids only)
ids = [os.path.basename(l.split("\t")[0])[:-4] for l in open(ROOT / "results/real_data_learning_curve/manifests/R100_s42.tsv")]
(OUT / "dwpose_train_clip_ids.txt").write_text("\n".join(ids) + "\n")

# alignment table (no text): row order kept; clip name and signer where the (video, subtitle) pair is unique in both tables
a = pd.read_excel(DS / "Sign Language Alignment/Auslan_Daily_Communication.xlsx")
t = pd.read_excel(TR / "Auslan-Daily_Communication.xlsx")
ka = a.groupby(["Video_Name", "Subtitle"]).size(); kt = t.groupby(["Video_Name", "Subtitle"]).size()
u = ka[ka == 1].index.intersection(kt[kt == 1].index)
tj = t.set_index(["Video_Name", "Subtitle"]).loc[u][["Video_Clip_Name", "Signer_ID"]].reset_index()
am = a.merge(tj, on=["Video_Name", "Subtitle"], how="left")
am[["Video_Name", "Begin_Frame", "End_Frame", "Video_Clip_Name", "Signer_ID"]].to_csv(OUT / "alignment_rows.csv", index=False)

# frames of the locally held DWPose clips
L = {os.path.basename(f)[:-4]: int(np.load(f)["KPS"].shape[0])
     for f in glob.glob(str(ROOT / "pose/extraction/dwpose_clips/*.npz"))}
pd.DataFrame(sorted(L.items()), columns=["clip", "frames"]).to_csv(OUT / "dwpose_clip_frames.csv", index=False)
print("ok", news)
