"""Run once in the private tree (needs the raw datasets); writes the small csv files check.py reads. Not run by make."""
import gzip, json, pickle, zipfile
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "data/raw/Auslan-Daily/Dataset Split Table"
tr = pd.read_excel(ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
split = dict(zip(tr["Video_Clip_Name"], tr["Split"]))

sp = pd.read_excel(RAW / "Sign Spotting/Sign_Spotting.xlsx")
sp["clip"] = sp["Video_Name_Clip"].str.replace("_signer.mp4", "", regex=False)
sp["official_split"] = sp["clip"].map(split)
sp[["clip", "Query_Gloss", "Begin_Frame", "End_Frame", "Split", "official_split"]].to_csv(HERE / "spotting.csv", index=False)

al = pd.read_excel(RAW / "Sign Language Alignment/Auslan_Daily_Communication.xlsx")
def secs(s):  # "00:1.12" = mm:ss.xx
    m, r = str(s).split(":"); return int(m) * 60 + float(r)
al["bt"] = al["Begin_Time"].map(secs); al["et"] = al["End_Time"].map(secs)
al[["Begin_Frame", "End_Frame", "bt", "et"]].to_csv(HERE / "alignment_frames_seconds.csv", index=False)

rows = []
for c in sorted((ROOT / "full_run/clips_signer").glob("*.npz")):
    with zipfile.ZipFile(c) as z:
        with z.open("KPS.npy") as f:
            v = np.lib.format.read_magic(f); shp = np.lib.format._read_array_header(f, v)[0]
    rows.append((c.stem, shp[0], split.get(c.stem)))
pd.DataFrame(rows, columns=["clip", "frames", "split"]).to_csv(HERE / "auslan_daily_clip_frames.csv", index=False)

lab = {**json.load(open(ROOT / "mmwl_pose/Train.json")), **json.load(open(ROOT / "mmwl_pose/Valid.json"))}
d = pickle.load(gzip.open(ROOT / "mmwl_pose/pose_cam1_trainval.pkl", "rb"))
pd.DataFrame([(k, lab.get(str(k)), v.shape[0]) for k, v in d.items()], columns=["id", "gloss", "frames"]).to_csv(HERE / "mmwl_clip_frames.csv", index=False)
