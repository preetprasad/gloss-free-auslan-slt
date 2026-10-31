"""Citation vs conversational signs: clip lengths, frame rates, Sign_Spotting durations, GOAL durations.

Recomputes from small tables written by prepare_data.py: Auslan-Daily Sign_Spotting spans, the alignment
table (frames and seconds), the frame counts of the 14,041 shipped Communication tensors (signer entry),
and the frame counts and gloss labels of the 45,010 MM-WLAuslan pose clips (Train + Valid).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sp = pd.read_csv(HERE / "spotting.csv")
al = pd.read_csv(HERE / "alignment_frames_seconds.csv")
ad = pd.read_csv(HERE / "auslan_daily_clip_frames.csv")
mm = pd.read_csv(HERE / "mmwl_clip_frames.csv")

# GATE: the recorded per-split mean clip lengths of the shipped tensors (numpy audit of 26 Sep 2026: train 75.2)
tr_mean = ad.frames[ad.split == "train"].mean()
print(f"GATE train-split mean clip length {tr_mean:.1f} frames (recorded 75.2)")
if round(tr_mean, 1) != 75.2:
    sys.exit("GATE FAILED")

print("\nAuslan-Daily Communication clip length (shipped tensors, signer entry)")
m_all = ad.frames.mean()
print(f"clips {len(ad)}; mean {m_all:.1f} frames (train {tr_mean:.1f}, dev {ad.frames[ad.split=='dev'].mean():.1f}, "
      f"test {ad.frames[ad.split=='test'].mean():.1f})  [c09-037 c09-045 c07-031]")
print(f"alignment table End - Begin: mean {(al.End_Frame-al.Begin_Frame).mean():.1f} frames")

fps = (al.End_Frame - al.Begin_Frame).sum() / (al.et - al.bt).sum()
print(f"Auslan-Daily frames per second from the alignment table: {fps:.2f} (25 expected)  [c10-044]")

mm_mean = mm.frames.mean()
print(f"\nMM-WLAuslan pose clips (Train+Valid): {len(mm)}; mean {mm_mean:.2f} frames; dissertation says 79.4  [c07-031]")
print(f"DISAGREES (c07-031): shipped pose gives {mm_mean:.1f} frames, not 79.4 (79.4 may come from the DWPose re-extraction, not saved locally); "
      f"the 73.1 half agrees. Ratio citation/sentence {mm_mean/m_all:.2f}  [c07-031]")

sp["L"] = sp.End_Frame - sp.Begin_Frame
vt = sp.official_split.isin(["dev", "test"]).sum()
print(f"\nSign_Spotting: {len(sp)} instances, {vt} on official validation and test clips, "
      f"mean {sp.L.mean():.2f} and median {sp.L.median():.0f} frames at 25 fps (End - Begin)  [c07-034]")

# GOAL
g_c = mm[mm.gloss == "GOAL"].frames
g_s = sp[sp.Query_Gloss.str.lower() == "goal"].L
mc, ms = g_c.median() / 30, g_s.median() / 25
print(f"\nGOAL citation form: n={len(g_c)}, median {g_c.median():.1f} frames = {mc:.2f} s at 30 fps  [c07-038]")
print(f"GOAL conversation:  n={len(g_s)}, median {g_s.median():.0f} frames = {ms:.2f} s at 25 fps  [c07-038]")
print(f"ratio {mc/ms:.1f} times (dissertation: 6.4, medians 3.08 s and 0.48 s, n=12 and n=6)  [c07-038]")
print(f"DISAGREES (c07-038) on n only: saved data give n={len(g_c)} and n={len(g_s)}, dissertation says 12 and 6; medians and ratio agree  [c07-038]")

# 45 shared glosses
gl = mm.dropna(subset=["gloss"]).copy(); gl["g"] = gl.gloss.str.upper()
sp["g"] = sp.Query_Gloss.str.upper()
sh = sorted(set(gl.g) & set(sp.g))
c = gl[gl.g.isin(sh)].frames; s = sp[sp.g.isin(sh)].L
sec_c, sec_s = c.mean() / 30, s.mean() / 25
print(f"\nGlosses present in both datasets (exact name match): {len(sh)}  [c07-032]")
print(f"citation {c.mean():.1f} frames = {sec_c:.2f} s (n={len(c)}); conversation {s.mean():.1f} frames = {sec_s:.2f} s (n={len(s)})  [c07-032]")
print(f"conversation is {100*(1-sec_s/sec_c):.0f}% shorter in seconds ({sec_c/sec_s:.2f} times); dissertation says 72% (3.60 times)  [c07-032]")
print(f"DISAGREES (c07-032): pooled by sign the saved spans give {100*(1-sec_s/sec_c):.0f}% shorter, not 72% (the original paired per-gloss selection is not saved)  [c07-032]")
