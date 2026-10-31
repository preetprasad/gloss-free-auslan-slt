"""Dataset statistics of Auslan-Daily and MM-WLAuslan (dissertation Review, Methodology, Results, Appendix B).

Recomputes from the dataset spreadsheet (Communication translation table, restored by `make data`) and from counts-only
files saved from the release's annotation tables (prepare_data.py): split sizes, signers, vocabulary, video count,
annotation inventory, the DWPose training list, the alignment audit (extraction exactness, overlapping subtitles) and
the MM-WLAuslan lexicon size.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = HERE / "data"
xl = ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx"
if not xl.exists():
    sys.exit("needs the dataset spreadsheet: run `make data` first")
c = pd.read_excel(xl)

# GATE: the release has 14,041 Communication rows and the official 12,441 / 800 / 800 split (recorded in the ReadMe tables and the manifests)
sizes = {s: int((c.Split == s).sum()) for s in ("train", "dev", "test")}
print(f"GATE Communication rows {len(c)} (expected 14041), split {sizes['train']}/{sizes['dev']}/{sizes['test']} (expected 12441/800/800)")
if len(c) != 14041 or (sizes["train"], sizes["dev"], sizes["test"]) != (12441, 800, 800):
    sys.exit("GATE FAILED")

# --- Communication ---
print("\nAUSLAN-DAILY COMMUNICATION (translation spreadsheet)")
print(f"Videos: {c.Video_Name.nunique()}  [c03-016]")
print(f"Clips train/dev/test: {sizes['train']:,} / {sizes['dev']:,} / {sizes['test']:,}  [c04-010 c04-022]")
sg = {s: int(c[c.Split == s].Signer_ID.astype(str).str.strip().nunique()) for s in sizes}
print(f"Signers train/dev/test: {sg['train']}/{sg['dev']}/{sg['test']}  [c04-010 c04-022]")
voc = len({w for t in c[c.Split == 'train'].Subtitle.astype(str) for w in t.lower().split()})
print(f"Training vocabulary (lower-cased whitespace tokens): {voc:,}  [c04-010 c04-022]")
print(f"Total clips (Communication): {len(c):,}  [c15-013]")
# the official manifests written by `make data` agree
for s, k in (("train", "train"), ("val", "dev"), ("test", "test")):
    mp = ROOT / "comm_run/manifests" / f"{s}.tsv"
    if mp.exists():
        nl = sum(1 for _ in open(mp))
        assert nl == sizes[k], (s, nl)
print(f"Training list used for DWPose input and CTC target: ", end="")
ids = [l.strip() for l in open(D / "dwpose_train_clip_ids.txt") if l.strip()]
tr_ids = set(c[c.Split == "train"].Video_Clip_Name)
assert set(ids) <= tr_ids
print(f"{len(ids):,} of {sizes['train']:,} training clips ({len(tr_ids - set(ids))} of the official list absent: {sorted(tr_ids - set(ids))})  [c08-001]")
print(f"Test clips {sizes['test']}  [c08-001]")

# --- News (counts-only file) ---
nw = json.load(open(D / "news_counts.json"))
print("\nAUSLAN-DAILY NEWS (counts saved from the News translation spreadsheet; read from the saved file news_counts.json)")
print(f"Clips train/dev/test: {nw['train']['clips']:,} / {nw['dev']['clips']:,} / {nw['test']['clips']:,}; rows {nw['rows']:,}  [c04-011 c04-023]")
print(f"Signers train/dev/test (identifiers stripped of spaces): {nw['train']['signers']}/{nw['dev']['signers']}/{nw['test']['signers']}  [c04-011 c04-023]")
print(f"Training vocabulary (lower-cased whitespace tokens): {nw['train_vocab_lower_whitespace']:,}  [c04-011 c04-023]")
tot = len(c) + nw["rows"]; trn = sizes["train"] + nw["train"]["clips"] + 0
print(f"Total clips, both subsets: {tot:,}; training clips both subsets: {sizes['train'] + nw['train']['clips']:,}  [c04-025 c04-037 c05-028]")
print("published (Auslan-Daily paper, Table 2): News 9,665 / 700 / 700 clips, 18/17/17 signers, vocabulary 12,346; 25,106 clips in all")
print(f"DISAGREES with the published News figures: signers {nw['train']['signers']}/{nw['dev']['signers']}/{nw['test']['signers']} vs published 18/17/17, vocabulary {nw['train_vocab_lower_whitespace']:,} vs published 12,346  [c04-011 c04-023]")
print(f"DISAGREES with the published News figures: News train {nw['train']['clips']:,} vs 9,665 (+{nw['train']['clips']-9665}); total {tot:,} vs 25,106 (+{tot-25106}); "
      f"training clips {sizes['train'] + nw['train']['clips']:,} vs 22,106 (+{sizes['train'] + nw['train']['clips']-22106})  [c04-011 c04-023 c04-025 c04-037 c05-028]")

# --- annotation inventory ---
sp = pd.read_csv(D / "spotting.csv"); fs = pd.read_csv(D / "fingerspelling.csv"); iso = pd.read_csv(D / "isolated.csv")
cl = set(c.Video_Clip_Name)
fs_comm = int(fs.Video_Clip_Name.str.replace("_signer.mp4", "", regex=False).isin(cl).sum())
al = pd.read_csv(D / "alignment_rows.csv")
print("\nANNOTATION INVENTORY (release tables, counts saved by prepare_data.py)")
print(f"Sign-spotting instances: {len(sp):,} occurrences of {sp.Query_Gloss.nunique()} glosses  [c06-014 c15-009]")
print(f"Isolated gloss segments: {len(iso):,} ({iso.Gloss.nunique()} glosses)  [c06-015 c15-010]")
print(f"Fingerspelling spans: {len(fs):,}, of which {fs_comm:,} on Communication clips  [c06-015 c15-011]")
print(f"Subtitle alignment file (Communication): {len(al):,} rows  [c15-012]")
print(f"English sentence per Communication clip: {c.Subtitle.notna().sum():,} clips  [c15-013]")

# --- MM-WLAuslan ---
g = pd.read_csv(D / "mmwl_gloss_counts.csv")
ncl = int(g.train_clips.sum() + g.valid_clips.sum())
print("\nMM-WLAuslan (Train.json + Valid.json labels, per-gloss clip counts)")
print(f"Front-view clips {ncl:,}, glosses {len(g):,}, exemplars per gloss {ncl/len(g):.1f}  [c06-027]")

# --- alignment audit ---
cf = pd.read_csv(D / "dwpose_clip_frames.csv").set_index("clip").frames
mm = al.dropna(subset=["Video_Clip_Name"]); mm = mm[mm.Video_Clip_Name.isin(cf.index)].copy()
mm["n"] = mm.Video_Clip_Name.map(cf); mm["span"] = mm.End_Frame - mm.Begin_Frame
d = (mm.n - mm.span).abs()
print(f"\nALIGNMENT AUDIT\nExtraction vs alignment span on {len(mm)} locally held clips: identical {100*(d==0).mean():.1f}%, within 2 frames {100*(d<=2).mean():.1f}%, "
      f"largest difference {int(d.max())} frames, correlation {np.corrcoef(mm.span, mm.n)[0,1]:.4f}  [c10-049]")
# consecutive subtitles by begin frame within a video
a = al.copy(); a["_i"] = range(len(a))
a = a.sort_values(["Video_Name", "Begin_Frame"], kind="stable")
pairs = []
for v, gr in a.groupby("Video_Name", sort=False):
    gr = gr.reset_index(drop=True)
    for i in range(1, len(gr)):
        p, q = gr.loc[i - 1], gr.loc[i]
        pairs.append((p.End_Frame - q.Begin_Frame, q.End_Frame - q.Begin_Frame, p.Signer_ID, q.Signer_ID, q.Video_Clip_Name))
P = pd.DataFrame(pairs, columns="ov dq sp sq cl".split())
frac = (P.ov > 0).mean()
print(f"Consecutive subtitle pairs {len(P):,}; overlapping {100*frac:.2f}% (recorded by the audit: 15.98%)")
if round(100 * frac, 2) != 15.98:
    sys.exit("GATE FAILED: overlap share")
K = P[(P.ov > 0) & P.sp.notna() & P.sq.notna()]
same = K.sp == K.sq
big = K[same & (K.ov / K.dq > 0.25)]
n_clips = big.cl.nunique()
print(f"Overlapping pairs with both signers known: {len(K):,}; different signer {100*(~same).mean():.1f}%, same signer {100*same.mean():.1f}%  [c10-050]")
print(f"Same signer and more than 25% of the later clip overlapped: {n_clips} clips, {100*n_clips/len(al.dropna(subset=['Video_Clip_Name'])):.2f}% of the {al.Video_Clip_Name.notna().sum():,} matched clips "
      f"(about 1%; the audit's own join gave 131 clips, 1.05%)  [c10-050]")
