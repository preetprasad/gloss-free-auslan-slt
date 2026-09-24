"""FIG C9 -- every recipe for which more than one run exists, with the runs shown.

The survey behind results/FIGURE_STANDARD_2026-09-22.md found that across ~60 papers in six
venue clusters, error bars, confidence intervals and multi-seed means are essentially absent
from figures: the field's normal practice is a single-run point estimate. This project has
paired-seed replications and a measured same-config noise floor, so it can show the spread.

Every number is a beam-5 BLEU-4 on the official Auslan-Daily Communication test split, sourced
from the project log's dated ledger or, for the Wave-1 control, re-read from the saved decode JSON in
results/hand_masking_and_longer_training/. Recipes are grouped by the BATCH they were run in, because the same
configuration run in two different batches differs by up to 1.18 BLEU-4 (2026-09-16e), so only
within-batch differences are meaningful.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle
figstyle.apply(plt)
C, FS, TARGET_W = figstyle.C, figstyle.FS, figstyle.TARGET_W

# (label, seeds, batch, colour key)  -- beam-5 test BLEU-4, one point per seed
ROWS = [
    ("DWPose + CTC\nreal data only",        [9.49, 9.03, 9.94],        "stitch", "body"),
    ("VLP-80 $\\rightarrow$ CTC",           [10.23, 10.43, 10.27],     "seeds",  "accent"),
    ("+ duration-stitched\nsynthetic data",  [11.08, 10.74, 11.08],    "stitch", "sup"),
    ("adopted recipe\n(1.1:1 stitch)",       [10.637, 11.296, 11.345], "wave1",  "body"),
    ("+ pose augmentation",                  [11.443, 11.375, 11.193], "aug",    "sup"),
]
# six same-config controls, two batches x three seeds (the project log 2026-09-16e)
NOISE = dict(mean=10.891, sd=0.393, n=6)

fig, ax = plt.subplots(figsize=(TARGET_W, 2.95))
ys = np.arange(len(ROWS))[::-1]
rng = np.random.default_rng(0)

# the noise floor: what the SAME configuration scores when only the seed changes
ax.axvspan(NOISE["mean"] - NOISE["sd"], NOISE["mean"] + NOISE["sd"],
           color=C["grey"], alpha=0.16, zorder=0, lw=0)
ax.axvline(NOISE["mean"], color=C["grey"], lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(NOISE["mean"], len(ROWS) - 0.52,
        f"same configuration, {NOISE['n']} runs:\n$\\pm1$ s.d. $=\\pm{NOISE['sd']:.2f}$",
        ha="center", va="bottom", fontsize=FS["tiny"], color=C["grey"], linespacing=1.3)

for y, (lab, vals, batch, key) in zip(ys, ROWS):
    v = np.array(vals)
    col = C[key]
    ax.plot([v.min(), v.max()], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=3)
    ax.scatter(v, np.full_like(v, y) + rng.uniform(-0.055, 0.055, len(v)), s=26, color=col,
               zorder=4, edgecolors="white", linewidths=0.5)
    ax.plot([v.mean()] * 2, [y - 0.19, y + 0.19], color=C["ink"], lw=1.6, zorder=5)
    ax.text(v.max() + 0.10, y, f"{v.mean():.2f}", va="center", fontsize=FS["tiny"], color=C["ink"])

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], fontsize=FS["sub"])
ax.set_ylim(-0.45, len(ROWS) + 0.22)
ax.set_xlabel("beam-5 BLEU-4, official Communication test split, one point per seed",
              fontsize=FS["label"])
ax.set_xlim(8.6, 12.0)
ax.grid(axis="x", alpha=0.22); ax.grid(axis="y", visible=False)
fig.text(0.0, -0.10,
         "Vertical rule = 3-seed mean. Rows are only comparable to others run in the same batch: "
         "the same configuration\nin two different batches differs by up to 1.18 BLEU-4, which is "
         "why every claim in this thesis rests on a paired\nin-batch control rather than on a "
         "difference between two rows of a table.",
         fontsize=FS["tiny"], color=C["grey"], va="top", linespacing=1.45)

fig.savefig(os.path.join(HERE, "figC9_seed_spread.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figC9_seed_spread.pdf"), bbox_inches="tight")
print("figC9 written")
for lab, vals, *_ in ROWS:
    v = np.array(vals)
    print(f"  {lab.splitlines()[0]:28s} mean {v.mean():6.3f}  range {v.min():.2f}-{v.max():.2f}  sd {v.std(ddof=1):.3f}")
