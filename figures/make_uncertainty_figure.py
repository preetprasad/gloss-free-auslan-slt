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
    ("real data only\n(control of the stitching runs)",        [9.49, 9.03, 9.94],        "stitch", "body"),
    ("40-epoch pre-trained\n(three seeds, August)",           [10.23, 10.43, 10.27],     "seeds",  "accent"),
    ("+ duration-stitched\nsynthetic data",  [11.08, 10.74, 11.08],    "stitch", "sup"),
    ("1.1:1 stitch + augmentation\n(control of a later job)",       [10.637, 11.296, 11.345], "wave1",  "body"),
    ("1.1:1 stitch + augmentation\n(first job)",                  [11.443, 11.375, 11.193], "aug",    "sup"),
]
# six same-config controls, two batches x three seeds (the project log 2026-09-16e)
NOISE = dict(mean=10.891, sd=0.393, n=6)

# built at printed width; constrained layout keeps the long row labels inside 6.30 in,
# so the file is saved WITHOUT bbox_inches="tight" (which would change the printed width)
fig, ax = plt.subplots(figsize=(TARGET_W, 2.6), layout="constrained")
rng = np.random.default_rng(0)

# rows grouped by batch (only within-batch differences are meaningful):
#   stitching batch: control, treatment | the 1.1:1 recipe in two batches | seed-only batch
# read top to bottom: seed-only reference | the 1.1:1 recipe in two batches (control first) |
# stitching batch (control, then the proposed duration-stitched recipe last, at the bottom)
ORDER = [1, 3, 4, 0, 2]
YPOS = [5.0, 3.6, 2.8, 1.4, 0.6]
_ypos = dict(zip(ORDER, YPOS))
ys = np.array([_ypos[i] for i in range(len(ROWS))])
for sep in (4.3, 2.1):                                   # thin rules between batch groups
    ax.axhline(sep, color="#DDDDDD", lw=0.4, zorder=0.5)

# role styles: the frozen anatomical colours are NOT reused here (figstyle rule).
# two roles only: reference/control runs = grey, dashed, open marks (circles for controls,
# diamonds for the seed-only pre-trained reference); stitched-synthetic-data runs (the
# contribution) = saturated accent, solid, filled circles. Readable in greyscale by mark alone.
STYLE = {
    "body":   dict(col=C["grey"],   ls=(0, (3, 2)), lw=0.75, mk="o", s=20, fill=False),
    "sup":    dict(col=C["accent"], ls="-",         lw=1.4,  mk="o", s=20, fill=True),
    "accent": dict(col=C["grey"],   ls=(0, (3, 2)), lw=0.75, mk="D", s=16, fill=False),
}

# the noise floor: what the SAME configuration scores when only the seed changes
ax.axvspan(NOISE["mean"] - NOISE["sd"], NOISE["mean"] + NOISE["sd"],
           color=C["grey"], alpha=0.2, zorder=0, lw=0)
ax.axvline(NOISE["mean"], color=C["grey"], lw=0.75, ls=(0, (4, 3)), zorder=1)
# (the explanation of the shaded band is in the caption, not inside the plot)

for y, (lab, vals, batch, key) in zip(ys, ROWS):
    v = np.array(vals)
    st = STYLE[key]
    col = st["col"]
    ax.plot([v.min(), v.max()], [y, y], color=col, lw=st["lw"], ls=st["ls"],
            solid_capstyle="round", zorder=3)
    jit = rng.uniform(-0.055, 0.055, len(v))
    if st["fill"]:
        ax.scatter(v, np.full_like(v, y) + jit, s=st["s"], marker=st["mk"], color=col,
                   zorder=4, edgecolors="white", linewidths=0.5)
    else:
        ax.scatter(v, np.full_like(v, y) + jit, s=st["s"], marker=st["mk"],
                   facecolors="white", edgecolors=col, linewidths=0.9, zorder=4)
    ax.plot([v.mean()] * 2, [y - 0.22, y + 0.22], color=C["ink"], lw=1.5, zorder=5)
    # (exact means are given in the caption; no per-row value labels)

_srt = np.argsort(ys)
ax.set_yticks(ys[_srt]); ax.set_yticklabels([ROWS[i][0] for i in _srt], fontsize=8)
for _tl, _i in zip(ax.get_yticklabels(), _srt):          # contribution rows named in accent
    _tl.set_color(C["accent"] if ROWS[_i][3] == "sup" else C["ink"])
ax.set_ylim(0.15, 5.45)
ax.set_xlabel("BLEU-4 $\\uparrow$ (beam 5), test split, one point per seed", fontsize=9)
ax.set_xlim(8.6, 12.0)
ax.set_xticks([9, 10, 11, 12])
ax.grid(axis="x", color="#DDDDDD", lw=0.4, alpha=1.0); ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_linewidth(0.6)
ax.tick_params(axis="x", direction="out", width=0.6, length=3, labelsize=8)
ax.tick_params(axis="y", length=0, labelsize=8)
# (the explanatory note that used to be printed under the axes is in the caption)

fig.savefig(os.path.join(HERE, "figC9_seed_spread.png"), dpi=300)
fig.savefig(os.path.join(HERE, "figC9_seed_spread.pdf"))
print("figC9 written")
for lab, vals, *_ in ROWS:
    v = np.array(vals)
    print(f"  {lab.splitlines()[0]:28s} mean {v.mean():6.3f}  range {v.min():.2f}-{v.max():.2f}  sd {v.std(ddof=1):.3f}")
