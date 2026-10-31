"""DWPose front end: extraction and training-list counts, input vector size, example-frame hand confidence,
hand speed on first-detection versus tracked keypoints, and hand confidence versus identification.

Recomputes from the saved per-clip files in data/ (see SOURCES.md)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = HERE / "data"

# ---- extraction and training-list counts
conf = pd.read_csv(D / "dwpose_conf.csv")
print(f"Communication clips with DWPose keypoints: {len(conf):,}  [c06-029 c08-019]")
ids = [l.strip() for l in open(D / "dwpose_train_clip_ids.txt") if l.strip()]
man = ROOT / "comm_run" / "manifests" / "train.tsv"
if not man.exists():
    sys.exit("needs the training manifest: run `make data` first")
train = [Path(l.split("\t")[0]).name[:-4] for l in open(man) if "\t" in l]
assert set(ids) <= set(train) and set(ids) <= set(conf["clip"])
print(f"Training clips in the official list: {len(train):,}; in the DWPose training list: {len(ids):,}  [c06-017 c08-002]")
print(f"Official training clips absent from the DWPose list: {len(set(train) - set(ids))} ({', '.join(sorted(set(train) - set(ids)))})")

# ---- input vector
nk, nc = map(int, open(D / "kps_shape.txt").read().split())
print(f"Keypoints per frame {nk}, values per keypoint {nc} (x, y, confidence): frame vector of {nk * nc} dimensions  [c06-028]")

# ---- hand confidence on the four example frames of the figure clip
f = pd.read_csv(D / "figure_clip_frame_hand_conf.csv")
mv = [t for t in range(len(f)) if f.dwpose_hand_conf[t] > 0.5]
picks = [mv[int(len(mv) * q)] for q in (0.15, 0.4, 0.62, 0.85)]
ap, dw = f.alphapose_hand_conf[picks].to_numpy(), f.dwpose_hand_conf[picks].to_numpy()
print(f"Figure clip, frames {picks} (15/40/62/85% of the frames with DWPose hand confidence above 0.5)")
print(f"Distributed hand keypoints, mean confidence per frame: {', '.join(f'{x:.2f}' for x in ap)}  range {ap.min():.2f} to {ap.max():.2f}  [c08-023]")
print(f"DWPose re-extraction, mean confidence per frame: {', '.join(f'{x:.2f}' for x in dw)}  range {dw.min():.2f} to {dw.max():.2f}  [c08-023]")

# ---- hand speed per frame, shoulder-width units, 591 training clips with both keypoint versions
h = pd.read_csv(D / "hand_speed_591.csv")
m1, m2 = h.speed_first.mean(), h.speed_tracked.mean()
print(f"\nGATE hand speed per frame on first-detection keypoints, {len(h)} clips: mean {m1:.3f} (recorded 0.391)")
if round(m1, 3) != 0.391 or len(h) != 591:
    sys.exit("GATE FAILED")
print(f"Tracked keypoints, same {len(h)} clips: mean {m2:.3f}; medians {h.speed_first.median():.3f} (first-detection) and {h.speed_tracked.median():.3f} (tracked)")
print(f"First-detection speed is {m1 / m2:.1f} times the tracked speed on the same clips (inflation of the first-detection figure)")

# ---- hand confidence against identification (hands descriptor, DWPose, 45 glosses)
sp = pd.read_csv(D / "span_hand_conf.csv"); p = pd.read_csv(D / "pred_dwpose.csv")
for v in ("hands",):
    q = p[p.variant == v].merge(sp, on="span")
    g = q.groupby("correct").hand_conf.mean()
    print(f"\nProbe variant '{v}': {len(q)} scored spans; mean hand confidence of correctly identified spans {g[1]:.3f}, of wrongly identified {g[0]:.3f}, correlation {q[['correct', 'hand_conf']].corr().iloc[0, 1]:+.3f}")
print(f"The two means differ by {abs(g[1] - g[0]):.3f} (hand confidence does not separate correct from wrong spans); the dissertation's particular pair of values is not reproduced by the saved probe output")
