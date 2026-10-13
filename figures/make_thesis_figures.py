"""Thesis Results-chapter figures.

Every number here is traceable to the project log §0 (the dated experiment ledger),
results/PHASE2_RESEARCH_JOURNEY_LOGBOOK.md §8 (the results ladder), or
results/FAILURE_ANALYSIS_2026-09-08.md. Sources are given per figure.

Run:  python make_thesis_figures.py     (writes PNG + PDF beside this file)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# One house style for the whole thesis, shared with make_concept_figures.py: the document's
# font, publication text sizes, and one colour per concept. Before this these figures shipped
# DejaVu Sans into a Computer Modern thesis and were built 7.0-11.0in wide, so LaTeX shrank them
# and the smallest labels printed at 5.7pt.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle
figstyle.apply(plt)
TARGET_W, FS, LADDER = figstyle.TARGET_W, figstyle.FS, figstyle.LADDER
# Standard 1.2-1.4 on top of the house style (fonts and colours unchanged): one tick size (8 pt),
# one label/annotation size (9 pt), 0.6 pt spines and outward 3 pt ticks, light value-axis grid.
FS = dict(FS, label=9.0, sub=8.0, tiny=8.0)
TICK, PANEL = FS["sub"], 10.0
REF = figstyle.C["grey"]            # baselines, prior work and reference lines: grey, 0.75 pt
plt.rcParams.update({
    "font.size": FS["label"], "axes.labelsize": FS["label"],
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#DDDDDD", "grid.alpha": 1.0, "grid.linewidth": 0.4,
    "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "errorbar.capsize": 2.0, "hatch.linewidth": 0.6, "legend.frameon": False,
})


def panel(ax, s, dx=-30):
    """Panel label (a), (b): 10 pt bold, top-left, outside the axes."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 6),
                textcoords="offset points", fontsize=PANEL, weight="bold",
                ha="left", va="bottom")

# Local aliases so the existing figure code reads unchanged; the values now come from the shared
# palette, so a colour cannot mean one thing here and something else in the concept figures.
C = dict(grey=figstyle.C["grey"], ink=figstyle.C["ink"], pale=figstyle.C["pale"],
         pose=figstyle.C["body"],        # the pose front end, matching the anatomical body colour
         pre=figstyle.C["accent"],       # pretraining / the contribution accent
         fusion=figstyle.C["good"],      # appearance fusion
         ens=LADDER[6],                  # ensembling
         base=figstyle.C["warn"],        # published baselines and harmful effects
         synth=figstyle.C["sup"])        # synthetic training data (a supervision artefact)

# =====================================================================
# FIG 1 — RESULTS LADDER
# The rows ARE result.tex's tab:res_ladder. Its beam-5 column is parsed below and
# asserted against this list, so the figure cannot silently fall behind the table
# again (it had already missed the 1.1:1 and pose-augmentation rows).
# VLP-80 / SWA entries are hatched: the project log 2026-09-06 records that recipe's
# control at 9.27 and 9.55 in two later batches, so 10.71/10.89 are unreplicated.
# =====================================================================
models = [
    ("AlphaPose keypoints (control)",       2.87,   "grey",   False),
    ("+ pretraining, tuned decoding",       3.87,  "grey",   False),
    ("DWPose re-extraction",                 8.78,  "pose",   False),
    ("+ CTC loss (3 seeds)",                9.63,  "pose",   False),
    ("+ pretraining, 40 epochs",           10.27,  "pre",    False),
    ("+ RGB fusion",                       10.33,  "fusion", False),
    ("pretraining, 80 epochs*",            10.71,  "pre",    True),
    ("+ checkpoint averaging*",            10.89,  "pre",    True),
    ("stitched synthetic data, 0.55:1 (3 seeds)", 10.97,  "synth",  False),
    ("    at 1.1:1 synthetic:real (3 seeds)", 11.14, "synth",  False),
    ("    + pose augmentation (3 seeds)",   11.34,  "synth",  False),
    ("6-member ensemble, beam width 5",    11.50,  "ens",    False),
    ("final ensemble: beam 10, length penalty 0.3", 11.80,  "ens",    False),
]

# Parse tab:res_ladder's beam-5 column and refuse to draw a stale ladder.
import os, re
_tex = open(os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "..", "SignLanguage_Preet", "src", "result.tex")).read()
_tab = _tex.split("\\label{tab:res_ladder}")[1].split("\\end{tabular}")[0]
_rows = []
for _line in _tab.splitlines():
    _f = _line.strip().removesuffix("\\\\").split("&")
    if len(_f) == 4:
        _m = re.fullmatch(r"(?:\\textbf\{)?([\d.]+)\}?", _f[2].strip())
        if _m:
            _rows.append(float(_m.group(1)))
assert len(_rows) == len(models), f"tab:res_ladder has {len(_rows)} beam-5 rows, figure has {len(models)}"
for _t, (_lab, _v, _, _) in zip(_rows, models):
    assert abs(_t - _v) < 1e-9, f"tab:res_ladder says {_t} where the figure plots {_v} ({_lab})"
labels = [m[0] for m in models]
vals   = [m[1] for m in models]
cols   = [C[m[2]] for m in models]
hatch  = ["///" if m[3] else None for m in models]
# a gap separates consecutive stages, so the grouping reads without colour
_gap = np.cumsum([0] + [0.55 if models[i][2] != models[i - 1][2] else 0
                        for i in range(1, len(models))])
y = np.arange(len(models)) + _gap

fig, ax = plt.subplots(figsize=(TARGET_W, 4.4))
bars = ax.barh(y, vals, color=cols, edgecolor="white", height=0.68, zorder=3)
for b, h in zip(bars, hatch):
    if h:
        b.set_hatch(h)
        b.set_edgecolor("white")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.tick_params(axis="both", labelsize=TICK)
ax.grid(False); ax.grid(True, axis="x")      # value axis only
ax.invert_yaxis()
# error bar on the replicated duration-stitch mean (10.74 / 11.08 / 11.08), found by label so
# a row inserted above it cannot silently move it onto a different bar
_dur = y[labels.index("stitched synthetic data, 0.55:1 (3 seeds)")]
# min-max range of the three seeds (10.74 / 11.08 / 11.08): 0.75 pt line, 2 pt caps
for xv in (10.74, 11.08):
    ax.plot([xv, xv], [_dur - 0.07, _dur + 0.07], color=C["ink"], lw=0.75, zorder=9,
            solid_capstyle="butt")
ax.plot([10.74, 11.08], [_dur, _dur], color=C["ink"], lw=0.75, zorder=9, solid_capstyle="butt")
# published baselines (prior work): grey dashed 0.75 pt, behind the bars, each labelled directly
# above the axis. Rows are staggered so each leader runs to the left of the labels below it.
for _v, _n, _h in ((9.20, "SL-Transformer 9.20", 0.16), (9.45, "SL-Luong 9.45", 0.10),
                   (9.95, "GASLT 9.95", 0.04)):
    ax.axvline(_v, ls="--", lw=0.75, color=REF, zorder=1)
    ax.plot([_v, _v], [1.0, 1.0 + _h], ls="-", lw=0.75, color=REF, clip_on=False,
            transform=ax.get_xaxis_transform())
    ax.text(_v + 0.08, 1.0 + _h, _n, fontsize=TICK, color=C["ink"], ha="left", va="center",
            transform=ax.get_xaxis_transform(), clip_on=False)
ax.set_ylim(y[-1] + 0.6, y[0] - 0.6)
ax.set_xlabel("Test BLEU-4, official Auslan-Daily Communication split")
ax.set_xlim(0, 12.5)
ax.set_xticks([0, 3, 6, 9, 12])
from matplotlib.lines import Line2D
leg = [Patch(fc=C["grey"], label="AlphaPose keypoints"),
       Patch(fc=C["pose"], label="DWPose keypoints"),
       Patch(fc=C["pre"],  label="pretraining"),
       Patch(fc=C["fusion"], label="RGB fusion"),
       Patch(fc=C["synth"], label="synthetic data / augmentation"),
       Patch(fc=C["ens"],  label="ensemble"),
       Patch(fc="white", ec=C["ink"], hatch="///", label="* unreplicated"),
       Line2D([], [], ls="-", lw=0.75, c=C["ink"], marker="|", ms=4, mew=0.75,
              label="range of 3 seeds")]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
           fontsize=TICK, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig("fig1_results_ladder.png", dpi=300, bbox_inches="tight")
fig.savefig("fig1_results_ladder.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 2 — VLP PRETRAINING SCALING (inverted-U)
# Source: the project log 2026-08-14, job 1089144 (single batch, matched recipe).
# =====================================================================
ep = [40, 80, 128, 256]
bl = [10.27, 10.71, 9.69, 9.66]
fig, ax = plt.subplots(figsize=(TARGET_W, 2.4))
ax.plot(ep, bl, "-o", color=C["pre"], lw=1.5, ms=4.5, zorder=3)
# label only the peak; the other values are read off the axis (no value label on every point)
ax.annotate(f"{bl[1]:.2f}", (ep[1], bl[1]), textcoords="offset points",
            xytext=(0, 5), ha="center", va="bottom", fontsize=TICK, color=C["ink"])
ax.set_xticks(ep)
ax.set_xlim(20, 270)
ax.set_xlabel("VLP pretraining epochs")
ax.set_ylabel("Test BLEU-4 (beam-5)")
ax.set_ylim(9.3, 11.0)
ax.set_yticks([9.5, 10.0, 10.5, 11.0])
fig.tight_layout()
fig.savefig("fig2_vlp_scaling.png", dpi=300, bbox_inches="tight")
fig.savefig("fig2_vlp_scaling.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 3 — SEEN vs UNSEEN TARGETS
# Source: the project log 2026-08-14 (metric suite, job 1081984); the video-disjoint
# reference line is LOGBOOK §8 row 29 (beam-5 6.39, 1,254 clips, 8 held-out videos).
# =====================================================================
mods = ["DWPose+CTC", "VLP (40 ep)", "RGB fusion"]  # the saved VLP decodes are the 40-epoch run (10.27)
official = [9.52, 10.27, 10.33]
seen     = [29.7, 29.9, 30.2]
unseen   = [4.88, 5.31, 5.81]
x = np.arange(len(mods)); w = 0.28
fig, ax = plt.subplots(figsize=(TARGET_W, 2.7))
# reference line (a separate retrain): grey dashed 0.75 pt, behind the bars
ax.axhline(6.39, ls="--", lw=0.75, color=REF, zorder=0.5)
# colour = method (the global mapping of fig1); split = fill: solid official, pale seen,
# hatched unseen, so the split survives greyscale and never borrows a method colour
_mc = [C["pose"], C["pre"], C["fusion"]]
b1 = ax.bar(x - w, official, w, color=_mc, edgecolor=_mc, linewidth=0.6, zorder=3)
b2 = ax.bar(x,      seen,     w, color=[matplotlib.colors.to_rgba(c, 0.30) for c in _mc],
            edgecolor=_mc, linewidth=0.6, zorder=3)
b3 = ax.bar(x + w,  unseen,   w, color="white", edgecolor=_mc, linewidth=0.6,
            hatch="////", zorder=3)
ax.text(1.008, 6.39, "video-disjoint\nretrain 6.39", fontsize=TICK, color=C["ink"],
        ha="left", va="center", transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_xlim(-0.48, 2.48)
# direct labels: each bar is named on the axis, the model on a second line below
_pos = np.concatenate([x - w, x, x + w])
ax.set_xticks(_pos)
ax.set_xticklabels(["official"] * len(mods) + ["seen"] * len(mods) + ["unseen"] * len(mods))
for _xi, _m in zip(x, mods):
    ax.annotate(_m, xy=(_xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -17),
                textcoords="offset points", ha="center", va="top", fontsize=FS["label"],
                color=C["ink"])
ax.set_ylabel("Test BLEU-4 (beam-5)")
ax.set_ylim(0, 32)
ax.set_yticks([0, 10, 20, 30])
fig.tight_layout()
fig.savefig("fig3_seen_vs_unseen.png", dpi=300, bbox_inches="tight")
fig.savefig("fig3_seen_vs_unseen.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 4 — ESTIMATOR 2x2 (the cleanest ablation)
# Source: the project log 2026-08-22, jobs 1101249 (stage) -> 1101250 (ablation).
# Matched articulator set (body+hands) and matched normalisation; beam-5.
# =====================================================================
fig, ax = plt.subplots(figsize=(TARGET_W, 2.7))
groups = ["bbox normalisation", "frame normalisation"]
alpha_v = [4.60, 5.58]
dw_v    = [1.45, 10.03]
x = np.arange(2); w = 0.32
b1 = ax.bar(x - w/2, alpha_v, w, color=C["grey"], edgecolor="white", linewidth=0.6,
            hatch="...", zorder=3)
b2 = ax.bar(x + w/2, dw_v, w, color=C["pose"], edgecolor=C["ink"], linewidth=0.6, zorder=3)
b2[0].set_hatch("///")          # the degenerate DWPose / bbox cell
ax.text(b2[0].get_x() + w/2, b2[0].get_height() + 0.3, "degenerate\n(1 unique output)",
        ha="center", va="bottom", fontsize=TICK, color=C["ink"])
# the isolated estimator effect, drawn in the gap between the two groups
xa = 1 - w - 0.06
ax.annotate("", xy=(xa, 10.03), xytext=(xa, 5.58),
            arrowprops=dict(arrowstyle="<->", lw=0.75, color=C["ink"]), zorder=5)
ax.plot([xa, 1 - w/2], [5.58, 5.58], ls=":", lw=0.75, color=REF, zorder=4)
ax.plot([xa, 1 + w/2], [10.03, 10.03], ls=":", lw=0.75, color=REF, zorder=4)
ax.text(xa - 0.04, 7.8, "+4.45", fontsize=FS["label"], color=C["ink"], ha="right",
        va="center")
# direct labels: estimator under each bar, normalisation on a second line
ax.set_xticks([x[0] - w/2, x[0] + w/2, x[1] - w/2, x[1] + w/2])
ax.set_xticklabels(["AlphaPose", "DWPose", "AlphaPose", "DWPose"])
for _xi, _g in zip(x, groups):
    ax.annotate(_g, xy=(_xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -17),
                textcoords="offset points", ha="center", va="top", fontsize=FS["label"],
                color=C["ink"])
ax.set_ylabel("Test BLEU-4 (beam-5)")
ax.set_xlim(-0.45, 1.45)
ax.set_ylim(0, 12)
ax.set_yticks([0, 4, 8, 12])
fig.tight_layout()
fig.savefig("fig4_estimator_2x2.png", dpi=300, bbox_inches="tight")
fig.savefig("fig4_estimator_2x2.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 5 — TEMPORAL-FIDELITY DOSE-RESPONSE
# Source: the project log 2026-09-06, job 1175390. One sentence pool, k=1, identical
# 6,880-clip budget, seed 42; only the temporal treatment moves. Real
# Auslan-Daily Communication clips average 73.1 frames over all 14,041 shipped tensors
# (alignment table End-Begin mean 72.7; the earlier 91.2 could not be reproduced).
# =====================================================================
frames = [246.1, 205.9, 108.4]
valb   = [9.987, 10.363, 10.625]
testb  = [10.80, 10.80, 11.08]
ticklab = ["246\nno\nresampling", "206\n+ frame-rate\nfix (30$\\rightarrow$25)",
           "108\n+ per-sign\nduration model"]
ctrl_val, real_frames = 9.908, 73.1

# two panels, because validation (greedy) and test (beam-5) are not on a comparable level;
# both y-ranges span the same 1.5 BLEU-4 so their slopes can be compared honestly
fig, (axv, axt) = plt.subplots(2, 1, figsize=(TARGET_W, 3.5), sharex=True)
for a in (axv, axt):
    a.axvline(real_frames, ls="--", lw=0.75, color=REF, zorder=1)
# the reference is labelled once, above panel (a), outside the data region
axv.plot([real_frames, real_frames], [1.0, 1.10], ls="--", lw=0.75, color=REF, clip_on=False,
         transform=axv.get_xaxis_transform())
axv.text(real_frames + 2, 1.10, "real clips, 73.1 frames", fontsize=TICK, color=C["ink"],
         ha="right", va="center", transform=axv.get_xaxis_transform(), clip_on=False)

axv.plot(frames, valb, "-o", color=C["synth"], lw=1.5, ms=4.5, zorder=5)
axv.axhline(ctrl_val, ls=":", lw=0.75, color=REF, zorder=1)
axv.text(150, ctrl_val - 0.04, "real-only control 9.91", fontsize=TICK,
         color=C["ink"], ha="center", va="top")
axv.annotate(f"{valb[-1]:.2f}", (frames[-1], valb[-1]), textcoords="offset points",
             xytext=(0, 5), ha="center", va="bottom", fontsize=TICK, color=C["ink"])
axv.set_ylabel("Validation BLEU-4\n(greedy)")
axv.set_ylim(9.6, 11.1)
axv.set_yticks([10.0, 10.5, 11.0])

axt.plot(frames, testb, "-s", color=C["synth"], lw=1.5, ms=4.5, zorder=4)
axt.annotate(f"{testb[-1]:.2f}", (frames[-1], testb[-1]), textcoords="offset points",
             xytext=(0, 5), ha="center", va="bottom", fontsize=TICK, color=C["ink"])
axt.set_ylabel("Test BLEU-4\n(beam-5)")
axt.set_ylim(10.2, 11.7)
axt.set_yticks([10.5, 11.0, 11.5])
panel(axv, "(a)", dx=-46); panel(axt, "(b)", dx=-46)

axt.set_xticks(frames)
axt.set_xticklabels(ticklab, fontsize=TICK)
axt.set_xlabel("Mean synthetic clip length (frames), decreasing $\\longrightarrow$", labelpad=6)
axt.set_xlim(262, 70)
fig.align_ylabels((axv, axt))
fig.tight_layout()
fig.savefig("fig5_temporal_fidelity.png", dpi=300, bbox_inches="tight")
fig.savefig("fig5_temporal_fidelity.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 6 — WER AGAINST ITS NULL-MODEL FLOOR
# Source: the project log 2026-09-05 (WER investigation: null models selected on
# validation, scored once on the 800-clip test split) and 2026-08-23 (LP sweep,
# MBR decodes). BLEU-4 for the same systems is printed alongside.
# =====================================================================
rows = [
    ("empty output (all deletions)",              1.0000, 0.00,   "null"),
    ('constant "hello ."',                        0.9988, 0.00,   "null"),
    ("6-member ensemble, beam 5, LP 1.0",              0.976,  11.50,  "model"),
    ("6-member ensemble, MBR decoding",               0.956,  10.69,  "model"),
    ('constant "it is sally ." (video ignored)',  0.9504, 0.38,   "null"),
    ("RGB-fusion model, beam 5",    0.9497, 10.33,  "model"),
    ("final ensemble (beam 10, LP 0.3)",             0.9069, 11.802, "best"),
    ("6-member ensemble, beam 10, LP 0.2",             0.9064, 11.616, "model"),
    ("RGB-fusion output cut to 3 words",        0.8999, 4.87,   "degenerate"),
    ("per-sentence oracle over 6 outputs", 0.807,  14.33,  "oracle"),
]
# marker areas in pt^2: diameters of 4-5 pt (the star slightly larger, it is the reported point)
# shape = kind of row; fill colour = the method (global mapping of fig1: ensemble yellow,
# RGB fusion teal); null models in the warn colour; the oracle hollow (it is not a method).
# Every marker has a dark edge so the yellow stays visible on white and in greyscale.
style = {"null":   dict(c=C["base"],   m="X", s=28),
         "model":  dict(c=C["grey"],   m="o", s=24),
         "best":   dict(c=C["ens"],    m="*", s=110),
         "oracle": dict(c="white",     m="D", s=20),
         "degenerate": dict(c=C["grey"], m="v", s=26)}


def _row_colour(lab, kind):
    if kind in ("null", "oracle"):
        return style[kind]["c"]
    if "ensemble" in lab:
        return C["ens"]
    if lab.startswith("RGB-fusion"):
        return C["fusion"]
    return style[kind]["c"]

y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(TARGET_W, 3.8))
ax.axvline(0.9504, ls="--", lw=0.75, color=REF, zorder=1)
ax.axvline(1.0000, ls=":",  lw=0.75, color=REF, zorder=1)

for yi, (lab, wer, bleu, kind) in zip(y, rows):
    st = style[kind]
    ax.plot([1.02, wer], [yi, yi], lw=0.75, color="#DDDDDD", zorder=1)
    ax.scatter(wer, yi, c=[_row_colour(lab, kind)], marker=st["m"], s=st["s"], zorder=5,
               edgecolors=C["ink"], linewidths=0.8 if kind == "best" else 0.5)
    ax.text(1.11, yi, f"{bleu:5.2f}", va="center", ha="right",
            fontsize=TICK, color=C["ink"], transform=ax.get_yaxis_transform(),
            clip_on=False, weight="bold" if kind == "best" else "normal")

# column header for the BLEU-4 values printed beside each row
ax.text(1.11, -0.95, "BLEU-4", fontsize=FS["label"], color=C["ink"], ha="right",
        va="bottom", transform=ax.get_yaxis_transform(), clip_on=False)

ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=TICK)
ax.invert_yaxis()
ax.set_xlim(1.021, 0.795)
ax.set_ylim(len(rows) - 0.4, -1.25)
ax.set_xlabel("WER (axis reversed: lower, i.e. better, to the right)")
ax.grid(False); ax.grid(True, axis="x")      # value axis only
_mk = lambda kind, lab, fc: Line2D([], [], ls="none", marker=style[kind]["m"],
                                   ms=np.sqrt(style[kind]["s"]), mfc=fc,
                                   mec=C["ink"], mew=0.5, label=lab)
leg = [_mk("null", "null model (ignores the video)", C["base"]),
       _mk("model", "trained system", "white"),
       _mk("degenerate", "trained system, output cut short", "white"),
       _mk("best", "reported operating point", C["ens"]),
       _mk("oracle", "oracle upper bound", "white"),
       Patch(fc=C["ens"], ec=C["ink"], lw=0.5, label="6-member ensemble"),
       Patch(fc=C["fusion"], ec=C["ink"], lw=0.5, label="RGB fusion"),
       Line2D([], [], ls="--", lw=0.75, c=REF, label="lowest null-model WER, 0.950"),
       Line2D([], [], ls=":", lw=0.75, c=REF, label="empty output, WER 1.000")]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
           fontsize=TICK, frameon=False)
fig.tight_layout(rect=(0, 0.19, 1, 1))
fig.savefig("fig6_bleu_wer_tradeoff.png", dpi=300, bbox_inches="tight")
fig.savefig("fig6_bleu_wer_tradeoff.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 7 — SELECTIVE PREDICTION (risk-coverage), reportable range only
# Source: results/SELECTIVE_PREDICTION_2026-09-09.md §3 (test, DWPose+CTC; ranking = z-sum of
# self-CE and self-reliance), §4 (unseen targets) and §5 (validation replication). Coverage below
# 50% is omitted: §5 shows the retained sets there are too small to replicate.
# =====================================================================
cov = [100, 90, 75, 50]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TARGET_W, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
# baselines first, grey, 0.75 pt; the confidence ranking (the method) last, saturated
ax1.plot(cov, [9.52, 9.90, 10.39, 10.48], "--s", color=REF, lw=0.75, ms=3.5, zorder=3,
         label="keep the longest outputs, test")
ax1.plot(cov, [9.52, 9.50, 9.37, 9.34], "-.x", color=REF, lw=0.75, ms=4, zorder=3,
         label="random abstention, test")
ax1.plot(cov, [9.52, 10.21, 11.54, 16.16], ":D", color=C["fusion"], lw=1.25, ms=3.5, zorder=4,
         label="oracle: true per-clip chrF, test")
ax1.plot(cov, [9.64, 10.37, 12.44, 18.42], "--o", color=C["pre"], lw=1.25, ms=4, mfc="white",
         zorder=5, label="confidence ranking, validation")
ax1.plot(cov, [9.52, 10.37, 12.05, 17.42], "-o", color=C["pre"], lw=1.5, ms=4.5, zorder=6,
         label="confidence ranking, test")
ax1.annotate("17.42", (50, 17.42), textcoords="offset points", xytext=(6, -1), ha="left",
             va="center", fontsize=TICK, color=C["ink"])
ax1.set_xlim(104, 43); ax1.set_xticks(cov); ax1.set_xticklabels([f"{c}%" for c in cov])
ax1.set_xlabel("Coverage (% of clips answered)")
ax1.set_ylabel("BLEU-4, answered clips\n(all clips)")
ax1.set_ylim(8.5, 19.5)
ax1.set_yticks([10, 14, 18])
covu = [100, 75, 50]
ax2.plot(covu, [4.88, 4.88, 4.68], "-.x", color=REF, lw=0.75, ms=4, zorder=3,
         label="random abstention, test")
ax2.plot(covu, [4.27, 4.34, 4.20], ":x", color=REF, lw=0.75, ms=4, zorder=3,
         label="random abstention, validation")
ax2.plot(covu, [4.27, 5.33, 7.24], "--o", color=C["pre"], lw=1.25, ms=4, mfc="white", zorder=5,
         label="confidence ranking, validation")
ax2.plot(covu, [4.88, 5.83, 7.92], "-o", color=C["pre"], lw=1.5, ms=4.5, zorder=6,
         label="confidence ranking, test")
ax2.invert_xaxis(); ax2.set_xticks(covu); ax2.set_xticklabels([f"{c}%" for c in covu])
ax2.set_xlabel("Coverage (% of clips answered)")
ax2.set_ylabel("BLEU-4, answered clips\n(unseen-target clips)")
ax2.set_ylim(3.5, 9.0)
ax2.set_yticks([4, 6, 8])
panel(ax1, "(a)", dx=-40); panel(ax2, "(b)", dx=-34)
# one shared legend below both panels (identical series appear once)
_h1, _l1 = ax1.get_legend_handles_labels()
_h2, _l2 = ax2.get_legend_handles_labels()
_h = _h1 + [h for h, l in zip(_h2, _l2) if l not in _l1]
_l = _l1 + [l for l in _l2 if l not in _l1]
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
           fontsize=TICK, frameon=False)
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig("fig7_selective_prediction.png", dpi=300, bbox_inches="tight")
fig.savefig("fig7_selective_prediction.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 8 — CONTINUOUS IDENTIFICATION PROBE
# Source: results/CONTINUOUS_ID_PROBE_2026-09-09.md (45 glosses present in both datasets,
# 8 exemplars per class, clip-disjoint folds, nearest centroid, chance 2.2%).
# =====================================================================
arts = ["face", "body", "hands", "body + hands"]
iso = [13.6, 32.5, 42.8, 41.9]   # all 630 citation clips (id-parsing fix, 2026-09-26; was 564)
cont = [3.3, 15.4, 18.1, 18.7]
w = 0.30
# per-group positions: groups hold 2, 2, 3, 2 bars and every gap between groups is the same
_nb = [2, 2, 3, 2]; _ggap = 0.45
_gstart = np.concatenate([[0.0], np.cumsum([n * w + _ggap for n in _nb[:-1]])])
x = _gstart + w                 # x - w/2 and x + w/2 are the first two bars of each group
fig, ax = plt.subplots(figsize=(TARGET_W, 2.8))
_ch = ax.axhline(2.2, ls=":", lw=0.75, color=REF, zorder=1)
# conditions, not methods: neutral fills only (solid grey isolated, hatched continuous,
# cross-hatched hand-local), so no method colour is reused for a condition
b1 = ax.bar(x - w/2, iso, w, color=C["grey"], edgecolor=C["ink"], linewidth=0.6, zorder=3)
b2 = ax.bar(x + w/2, cont, w, color="white", edgecolor=C["ink"], linewidth=0.6, hatch="///",
            zorder=3)
# the hand-local-normalisation result is a third bar in the hands group, not a floating marker
b3 = ax.bar([x[2] + 1.5 * w], [24.9], w, color="white", edgecolor=C["ink"], linewidth=0.6,
            hatch="xxx", zorder=3)
# horizontal direct labels on the first bar of each series (replaces the legend)
ax.text(b1[0].get_x() + w / 2, b1[0].get_height() + 1.0, "isolated", color=C["ink"],
        fontsize=TICK, ha="center", va="bottom")
ax.text(b2[0].get_x(), b2[0].get_height() + 1.0, "continuous", color=C["ink"],
        fontsize=TICK, ha="left", va="bottom")
ax.text(b3[0].get_x() + w / 2, b3[0].get_height() + 1.0, "hand-local", color=C["ink"],
        fontsize=TICK, ha="center", va="bottom")
_tx = x.astype(float); _tx[2] += w / 2          # centre the hands tick under its three bars
ax.set_xticks(_tx); ax.set_xticklabels(arts)
ax.set_xlim(-0.2, 4.25)
ax.text(1.008, 2.2, "chance 2.2%", fontsize=TICK, color=C["ink"], ha="left", va="center",
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_ylabel("Top-1 identification of 45 glosses (%)")
ax.set_ylim(0, 50)
fig.tight_layout()
fig.savefig("fig8_identification_probe.png", dpi=300, bbox_inches="tight")
fig.savefig("fig8_identification_probe.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 9 — SYNTHETIC : REAL RATIO (duration-modelled stitching)
# Source: the project log 2026-09-10x (job 1196567: seed 42, including the compute-matched arm) and
# 2026-09-15 (job 1199393: seeds 101, 202). Every point has its own in-batch real-only control.
# =====================================================================
ratios = [0, 0.55, 1.11, 1.66]
test_b = {42: [9.89, 10.90, 11.52, 10.67], 101: [10.10, 10.80, 11.03, 10.46], 202: [9.11, 10.70, 10.87, 10.41]}
# The real-only controls' validation scores are not in the project record, so they are missing here.
val_b = {42: [np.nan, 10.268, 10.271, 10.109], 101: [np.nan, 9.643, 10.912, 10.346],
         202: [np.nan, 11.091, 9.969, 10.399]}
_seed_mk = {42: "o", 101: "s", 202: "^"}        # one marker shape per seed, named in the caption
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TARGET_W, 2.6), sharey=False)
for ax, data, name in ((ax1, test_b, "Test BLEU-4 (beam-5)"),
                       (ax2, val_b, "Validation BLEU-4 (greedy)")):
    for s, v in data.items():
        ax.plot(ratios, v, "-", marker=_seed_mk[s], color=C["grey"], mfc="white", mec=C["ink"],
                mew=0.6, lw=0.75, ms=4, zorder=3)
    ax.set_xlim(-0.15, 2.05)
    arr = np.array(list(data.values()), dtype=float)
    mean = [np.nan if np.isnan(col).all() else float(np.nanmean(col)) for col in arr.T]
    ax.plot(ratios, mean, "-D", color=C["synth"], lw=1.5, ms=4, zorder=5)
    ax.text(ratios[-1] + 0.09, mean[-1], "mean", color=C["synth"], fontsize=TICK,
            ha="left", va="center")
    ax.set_xticks(ratios); ax.set_xticklabels(["real only", "0.55:1", "1.1:1", "1.66:1"])
    ax.set_xlabel("Synthetic : real training clips")
    ax.set_ylabel(name)
# (b) has no real-only validation scores, so its empty category is dropped; test and validation
# are not on a comparable level, so (b) gets its own labelled y-axis
ax2.set_xlim(0.40, 2.05)
ax2.set_xticks(ratios[1:]); ax2.set_xticklabels(["0.55:1", "1.1:1", "1.66:1"])
ax2.set_ylim(9.4, 11.3)
ax2.set_yticks([9.5, 10.0, 10.5, 11.0])
ax1.plot([1.66], [8.54], "X", ms=5, color=C["base"], ls="none", zorder=6)
ax1.text(1.56, 8.54, "compute-matched\ncontrol", color=C["base"], fontsize=TICK,
         ha="right", va="center")
ax1.set_ylim(8.2, 11.9)
ax1.set_yticks([9, 10, 11])
panel(ax1, "(a)", dx=-34); panel(ax2, "(b)", dx=-40)
fig.tight_layout()
fig.savefig("fig9_synthetic_ratio.png", dpi=300, bbox_inches="tight")
fig.savefig("fig9_synthetic_ratio.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 10 — WHAT AN 800-SENTENCE TEST SPLIT CAN RESOLVE (paired bootstrap)
# Source: results/bootstrap_all_2026-09-15.txt, written by src/tools/paired_bootstrap.py
# (2,000 resamples, seed 0; the tool first re-scores S6 11.802, F 12.114 and G1 11.562 exactly).
# Parsed at run time so the figure cannot drift from the tool's output.
# =====================================================================
import os, re
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bootstrap_all_2026-09-15.txt")
pat = re.compile(r"\[all n=800\] (\w+) vs (\w+): observed ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), "
                 r"([+-][\d.]+)\]\s+P\(\w+>\w+\) ([\d.]+)%")
boot = {(m.group(1), m.group(2)): tuple(float(m.group(i)) for i in (3, 4, 5, 6))
        for m in pat.finditer(open(src).read())}
order = [("VLP", "CTC", "40-epoch pre-trained $-$ baseline"),
         ("RGBF", "CTC", "RGB-fusion $-$ baseline"),
         ("S6", "RGBF", "final ensemble $-$ its RGB-fusion member"),
         ("F", "S6", "stitched-data ensemble (0.55:1) $-$ final ensemble"),
         ("G1", "S6", "8-member stitched-data ensemble $-$ final ensemble"),
         ("Fprime", "S6", "stitched-data ensemble (1.1:1) $-$ final ensemble"),
         ("Fprime", "F", "stitched-data ensemble: 1.1:1 $-$ 0.55:1")]
missing = [f"{a}:{b}" for a, b, _ in order if (a, b) not in boot]
assert not missing, f"bootstrap output lacks {missing}"
fig, ax = plt.subplots(figsize=(TARGET_W, 2.6))
n_sig, sig_pairs = 0, []
for yi, (a, b, lab) in enumerate(order):
    obs, lo, hi, p = boot[(a, b)]
    sig = lo > 0 or hi < 0
    n_sig += sig
    if sig:
        sig_pairs.append((a, b))
    # excludes zero: accent, diamond, bold numbers; otherwise grey circle (colour + shape + weight)
    col = C["pre"] if sig else C["grey"]
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.5, solid_capstyle="butt", zorder=3)
    ax.plot(obs, yi, "D" if sig else "o", ms=5 if sig else 4.5, color=col, mec="white",
            mew=0.6, zorder=4)
    ax.text(1.03, yi, f"{obs:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", ha="left",
            fontsize=TICK, color=C["ink"], transform=ax.get_yaxis_transform(),
            clip_on=False, weight="bold" if sig else "normal")
ax.text(1.03, 1.0, "$\\Delta$  [95% CI]", va="bottom", ha="left", fontsize=TICK,
        color=C["ink"], transform=ax.transAxes, clip_on=False)
ax.axvline(0, color=C["ink"], lw=0.75, zorder=2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order], fontsize=TICK)
ax.invert_yaxis()
ax.set_xlim(-2.0, 3.0)
ax.set_xlabel("Difference in test BLEU-4 (95% paired-bootstrap interval)")
ax.grid(False); ax.grid(True, axis="x")      # value axis only
fig.tight_layout()
fig.savefig("fig10_bootstrap_resolution.png", dpi=300, bbox_inches="tight")
fig.savefig("fig10_bootstrap_resolution.pdf", bbox_inches="tight")
plt.close(fig)

print("OK — wrote fig1..fig10 (png + pdf)")
import os
for f in sorted(os.listdir(".")):
    if f.startswith("fig") and f.endswith((".png", ".pdf")):
        print(f"  {f:34s} {os.path.getsize(f):>8d} bytes")
