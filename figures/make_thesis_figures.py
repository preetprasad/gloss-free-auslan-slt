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
    ("AlphaPose control (HALPE-136)",       2.87,   "grey",   False),
    ("VLP + tuned decode (pre-DWPose best)", 3.87,  "grey",   False),
    ("DWPose re-extraction",                 8.78,  "pose",   False),
    ("DWPose + CTC (3-seed)",                9.63,  "pose",   False),
    ("VLP-on-DWPose",                       10.27,  "pre",    False),
    ("RGB fusion",                          10.34,  "fusion", False),
    ("VLP-80 $\\rightarrow$ CTC*",          10.71,  "pre",    True),
    ("SWA-VLP80*",                          10.89,  "pre",    True),
    ("Duration-stitch (3-seed mean)",       10.97,  "synth",  False),
    ("    at 1.1:1 synthetic:real (3-seed)", 11.14, "synth",  False),
    ("    + pose augmentation (3 seeds)",   11.34,  "synth",  False),
    ("6-member ensemble, beam-5",           11.50,  "ens",    False),
    ("6-member ensemble, beam-10/LP0.3",    11.80,  "ens",    False),
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
y = np.arange(len(models))

fig, ax = plt.subplots(figsize=(TARGET_W, 4.75))
bars = ax.barh(y, vals, color=cols, edgecolor="white", height=0.68, zorder=3)
for b, h in zip(bars, hatch):
    if h:
        b.set_hatch(h)
        b.set_edgecolor("white")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.invert_yaxis()
# error bar on the replicated duration-stitch mean (10.74 / 11.08 / 11.08), found by label so
# a row inserted above it cannot silently move it onto a different bar
_dur = labels.index("Duration-stitch (3-seed mean)")
for yi, v in zip(y, vals):
    # that one row's value label is pushed clear of the whisker at 11.08
    ax.text(v + (0.32 if yi == _dur else 0.14), yi, f"{v:.2f}", va="center", ha="left",
            fontsize=FS["label"], color=C["ink"],
            zorder=8, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
# white casing so the range reads over the purple bar as well as over the background: without it
# the left whisker vanished inside the bar and the figure did not show the range its caption claims
import matplotlib.patheffects as _pe
_case = [_pe.withStroke(linewidth=3.4, foreground="white")]
for xv in (10.74, 11.08):                        # whiskers first: the connector is drawn over
    ax.plot([xv, xv], [_dur - 0.18, _dur + 0.18], color=C["ink"], lw=1.4, zorder=9,
            path_effects=_case)                  # their white casing, not erased by it
ax.plot([10.74, 11.08], [_dur, _dur], color=C["ink"], lw=1.4, zorder=10, path_effects=_case)
# published Auslan-Daily Communication/Pose baselines
ax.axvline(9.20, ls="--", lw=1.2, color=C["base"], zorder=2)
ax.axvline(9.45, ls="-.", lw=1.2, color=C["base"], zorder=2)
ax.axvline(9.95, ls=":",  lw=1.5, color=C["base"], zorder=2)
ax.set_ylim(len(models) - 0.3, -0.75)
ax.set_xlabel("Test BLEU-4, official Auslan-Daily Communication split")
ax.set_xlim(0, 13.0)
ax.set_title("Fixing the pose representation broke the $\\approx$3 BLEU-4 plateau",
             fontsize=FS["head"], weight="bold", pad=10)
from matplotlib.lines import Line2D
leg = [Patch(fc=C["grey"], label="pre-DWPose (AlphaPose input)"),
       Patch(fc=C["pose"], label="DWPose pose fix"),
       Patch(fc=C["pre"],  label="pretraining (VLP / SWA)"),
       Patch(fc=C["fusion"], label="RGB fusion"),
       Patch(fc=C["synth"], label="synthetic data / augmentation"),
       Patch(fc=C["ens"],  label="ensemble + decode tuning"),
       Patch(fc="white", ec=C["ink"], hatch="///", label="* unreplicated (see text)"),
       Line2D([], [], ls="--", lw=1.2, c=C["base"], label="SL-Transformer 9.20"),
       Line2D([], [], ls="-.", lw=1.2, c=C["base"], label="SL-Luong 9.45"),
       Line2D([], [], ls=":",  lw=1.5, c=C["base"], label="GASLT 9.95")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=FS["sub"], framealpha=0.95)
fig.tight_layout()
fig.savefig("fig1_results_ladder.png", dpi=300, bbox_inches="tight")
fig.savefig("fig1_results_ladder.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 2 — VLP PRETRAINING SCALING (inverted-U)
# Source: the project log 2026-08-14, job 1089144 (single batch, matched recipe).
# =====================================================================
ep = [40, 80, 128, 256]
bl = [10.27, 10.71, 9.69, 9.66]
fig, ax = plt.subplots(figsize=(TARGET_W, 3.96))
ax.axvspan(80, 256, color=C["base"], alpha=0.06, zorder=0)
ax.plot(ep, bl, "-o", color=C["pre"], lw=2.2, ms=8, zorder=3)
for x, v in zip(ep, bl):
    ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=FS["label"])
ax.plot(80, 10.71, "o", ms=15, mfc="none", mec=C["ens"], mew=2.4, zorder=4)
ax.annotate("peak (80 ep)", (80, 10.71), textcoords="offset points", xytext=(14, -16),
            fontsize=FS["label"], color=C["ink"], weight="bold")
ax.text(168, 9.55, "over-pretraining\n(encoder over-specialises)",
        fontsize=FS["sub"], color=C["base"], ha="center")
ax.set_xticks(ep)
ax.set_xlabel("VLP pretraining epochs")
ax.set_ylabel("Test BLEU-4 (beam-5)")
ax.set_ylim(9.3, 11.0)
ax.set_title("VLP pretraining follows an inverted-U; the optimum is $\\approx$80 epochs",
             fontsize=FS["head"], weight="bold", pad=10)
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
x = np.arange(len(mods)); w = 0.26
fig, ax = plt.subplots(figsize=(TARGET_W, 3.88))
b1 = ax.bar(x - w, official, w, label="Official split (blended)", color=C["grey"], zorder=3)
b2 = ax.bar(x,      seen,     w, label="Seen target (230 / 800)", color=C["fusion"], zorder=3)
b3 = ax.bar(x + w,  unseen,   w, label="Unseen target (570 / 800)", color=C["base"], zorder=3)
for bars_ in (b1, b2, b3):
    for r in bars_:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.4, f"{r.get_height():.1f}",
                ha="center", va="bottom", fontsize=FS["sub"])
ax.axhline(6.39, ls="--", lw=1.4, color=C["ink"], zorder=4)
ax.text(-1.02, 6.39, "video-disjoint\nretrain 6.39", fontsize=FS["sub"], color=C["ink"],
        ha="left", va="center")
ax.set_xlim(-1.08, 2.52)
ax.set_xticks(x); ax.set_xticklabels(mods)
ax.set_ylabel("BLEU-4 (beam-5)")
ax.set_ylim(0, 34)
ax.set_title("The official score is a blend: true generalisation is $\\approx$5–6 BLEU-4",
             fontsize=FS["head"], weight="bold", pad=10)
ax.legend(loc="upper center", fontsize=FS["sub"], ncol=3, framealpha=0.9)
ax.text(1.5, 15.8, "the order is preserved on unseen targets:\nRGB fusion > VLP > CTC (gaps not significant)",
        fontsize=FS["sub"], color=C["ink"], ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9), zorder=5)
fig.tight_layout()
fig.savefig("fig3_seen_vs_unseen.png", dpi=300, bbox_inches="tight")
fig.savefig("fig3_seen_vs_unseen.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 4 — ESTIMATOR 2x2 (the cleanest ablation)
# Source: the project log 2026-08-22, jobs 1101249 (stage) -> 1101250 (ablation).
# Matched articulator set (body+hands) and matched normalisation; beam-5.
# =====================================================================
fig, ax = plt.subplots(figsize=(TARGET_W, 4.09))
groups = ["bbox normalisation", "frame normalisation"]
alpha_v = [4.60, 5.58]
dw_v    = [1.45, 10.03]
x = np.arange(2); w = 0.32
b1 = ax.bar(x - w/2, alpha_v, w, label="AlphaPose HALPE-136 (shipped with the dataset)",
            color=C["grey"], zorder=3)
b2 = ax.bar(x + w/2, dw_v, w, label="DWPose RTMW-DW-x (re-extracted here)",
            color=C["pose"], zorder=3)
for bars_ in (b1, b2):
    for r in bars_:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.18, f"{r.get_height():.2f}",
                ha="center", va="bottom", fontsize=FS["label"])
# the isolated estimator effect, drawn clear of the bars
xa = 1 + w/2 + 0.20
ax.annotate("", xy=(xa, 10.03), xytext=(xa, 5.58),
            arrowprops=dict(arrowstyle="<->", lw=1.8, color=C["base"]), zorder=5)
ax.plot([1 - w/2, xa], [5.58, 5.58], ls=":", lw=1.1, color=C["base"], zorder=4)
ax.plot([1 + w/2, xa], [10.03, 10.03], ls=":", lw=1.1, color=C["base"], zorder=4)
ax.text(xa + 0.04, 7.8, "+4.45\npurely from the\npose estimator", fontsize=FS["head"],
        color=C["base"], ha="left", va="center", weight="bold")
ax.text(0 + w/2, 2.35, "degenerate\n(1 unique output)", fontsize=FS["sub"], color=C["ink"],
        ha="center", va="bottom", style="italic")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Test BLEU-4 (beam-5)")
ax.set_xlim(-0.55, 1.95)
ax.set_ylim(0, 13.6)
ax.set_title("Matched 2$\\times$2: the pose estimator, not normalisation, broke the plateau",
             fontsize=FS["head"], weight="bold", pad=10)
ax.legend(loc="upper left", fontsize=FS["sub"], framealpha=0.95)
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

fig, ax = plt.subplots(figsize=(TARGET_W, 4.04))
ax.axvline(real_frames, ls="--", lw=1.6, color=C["fusion"], zorder=2)
ax.text(real_frames + 4, 10.30, "real Auslan-Daily\nclips: 73.1 frames",
        fontsize=FS["sub"], color="#0B7D7D", ha="right", va="top")

ax.plot(frames, testb, "-s", color=C["synth"], lw=1.8, ms=8, zorder=4,
        label="test BLEU-4 (beam-5)")
ax.plot(frames, valb, "-o", color=C["pre"], lw=2.4, ms=9, zorder=5,
        label="validation BLEU-4 (the selection metric)")
ax.axhline(ctrl_val, ls=":", lw=1.6, color=C["grey"], zorder=2)
ax.text(150, ctrl_val - 0.04, "real-only control, validation 9.908", fontsize=FS["sub"],
        color="#5f6368", ha="center", va="top")

for f, v in zip(frames, valb):
    ax.annotate(f"{v:.3f}", (f, v), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=FS["label"], color=C["pre"], weight="bold")
for f, t in zip(frames, testb):
    ax.annotate(f"{t:.2f}", (f, t), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=FS["label"], color=C["synth"])

ax.invert_xaxis()
ax.set_xticks(frames)
ax.set_xticklabels(ticklab, fontsize=FS["sub"])
ax.set_xlabel("Mean synthetic clip length (frames) $\\longrightarrow$ closer to real signing",
              labelpad=6)
ax.set_ylabel("BLEU-4")
ax.set_xlim(262, 70)
ax.set_ylim(9.65, 11.62)
ax.set_title("Validation orders the arms monotonically with temporal fidelity",
             fontsize=FS["head"], weight="bold", pad=10)
ax.legend(loc="upper left", fontsize=FS["sub"], framealpha=0.95)
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
    ("S6 ensemble, beam-5 / LP 1.0",              0.976,  11.50,  "model"),
    ("S6 ensemble, MBR-WER decode",               0.956,  10.69,  "model"),
    ('constant "it is sally ." (video ignored)',  0.9504, 0.38,   "null"),
    ("RGB fusion, beam-5 (best single model)",    0.9497, 10.33,  "model"),
    ("S6 ensemble, beam-10 / LP 0.3",             0.9069, 11.802, "best"),
    ("S6 ensemble, beam-10 / LP 0.2",             0.9064, 11.616, "model"),
    ("RGB fusion, blindly cut to 3 words",        0.8999, 4.87,   "degenerate"),
    ("per-sentence oracle over 6 configurations", 0.807,  14.33,  "oracle"),
]
style = {"null":   dict(c=C["base"],   m="X", s=105),
         "model":  dict(c=C["pre"],    m="o", s=85),
         "best":   dict(c=C["ens"],    m="*", s=340),
         "oracle": dict(c=C["fusion"], m="D", s=75),
         "degenerate": dict(c=C["synth"], m="v", s=105)}

y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(TARGET_W, 3.62))
ax.axvspan(0.9504, 1.02, color=C["base"], alpha=0.08, zorder=0)
ax.axvline(0.9504, ls="--", lw=1.6, color=C["base"], zorder=2)
ax.axvline(1.0000, ls="-",  lw=1.3, color=C["grey"], zorder=2)

for yi, (lab, wer, bleu, kind) in zip(y, rows):
    st = style[kind]
    ax.plot([1.02, wer], [yi, yi], lw=0.9, color="#D6D8DA", zorder=1)
    ax.scatter(wer, yi, c=st["c"], marker=st["m"], s=st["s"], zorder=5,
               edgecolors="white", linewidths=0.9)
    ax.text(0.7860, yi, f"{bleu:5.2f}", va="center", ha="right",
            fontsize=FS["label"], color=C["ink"],
            weight="bold" if kind == "best" else "normal")

# the 0.0007 gap that makes the point
ax.annotate("", xy=(0.9497, 5), xytext=(0.9504, 4),
            arrowprops=dict(arrowstyle="-", lw=1.4, color=C["ink"]), zorder=6)
ax.text(0.947, 4.5, "  0.0007 apart on WER,\n  9.95 apart on BLEU-4",
        fontsize=FS["sub"], color=C["ink"], va="center", ha="left", style="italic")
ax.text(0.9497, -0.95, "null-model floor 0.950", fontsize=FS["sub"], color=C["base"],
        ha="left", va="bottom", weight="bold")
ax.text(1.0005, -0.95, "silence\n1.000", fontsize=FS["sub"], color="#5f6368",
        ha="right", va="bottom")
ax.text(0.7860, -0.95, "BLEU-4", fontsize=FS["label"], color=C["ink"], ha="right",
        va="bottom", weight="bold")

ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=FS["label"])
ax.invert_yaxis()
ax.set_xlim(1.021, 0.7745)
ax.set_ylim(len(rows) - 0.4, -1.25)
ax.set_xlabel("WER $\\longleftarrow$ better")
ax.grid(axis="y", visible=False)
ax.set_title("WER is nearly uninformative here: silence scores 1.000 and a constant\n"
             "string scores 0.950, leaving 0.093 of usable range",
             fontsize=FS["head"], weight="bold", pad=10)
leg = [Patch(fc=C["base"],   label="input-blind null model"),
       Patch(fc=C["pre"],    label="trained system"),
       Patch(fc=C["synth"],  label="trained system, output blindly truncated"),
       Patch(fc=C["ens"],    label="reported operating point"),
       Patch(fc=C["fusion"], label="oracle upper bound")]
ax.legend(handles=leg, loc="lower left", fontsize=FS["sub"], framealpha=0.96,
          borderpad=0.5)
fig.tight_layout()
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TARGET_W, 2.63), gridspec_kw={"width_ratios": [1.35, 1]})
ax1.plot(cov, [9.52, 10.37, 12.05, 17.42], "-o", color=C["ens"], lw=2.4, ms=7, zorder=5,
         label="confidence ranking, test")
ax1.plot(cov, [9.64, 10.37, 12.44, 18.42], "--o", color=C["synth"], lw=1.8, ms=6, zorder=4,
         label="confidence ranking, validation")
ax1.plot(cov, [9.52, 10.21, 11.54, 16.16], ":D", color=C["fusion"], lw=1.6, ms=5, zorder=3,
         label="oracle: true per-clip chrF, test")
ax1.plot(cov, [9.52, 9.90, 10.39, 10.48], "-s", color=C["grey"], lw=1.4, ms=5, zorder=3,
         label="keep the longest outputs, test")
ax1.plot(cov, [9.52, 9.50, 9.37, 9.34], "-x", color=C["base"], lw=1.4, ms=6, zorder=3,
         label="random abstention, test")
ax1.annotate("17.42", (50, 17.42), textcoords="offset points", xytext=(9, -1), ha="left", va="center",
             fontsize=FS["label"], color=C["ens"], weight="bold")
ax1.set_xlim(104, 43); ax1.set_xticks(cov); ax1.set_xticklabels([f"{c}%" for c in cov])
ax1.set_xlabel("Coverage (share of clips the model answers)")
ax1.set_ylabel("Corpus BLEU-4 on the answered clips")
ax1.set_ylim(8.5, 19.5)
ax1.set_title("All 800 clips", fontsize=FS["head"], weight="bold")
ax1.legend(loc="upper left", fontsize=FS["sub"], framealpha=0.95)
covu = [100, 75, 50]
ax2.plot(covu, [4.88, 5.83, 7.92], "-o", color=C["ens"], lw=2.4, ms=7, zorder=5,
         label="confidence ranking, test")
ax2.plot(covu, [4.27, 5.33, 7.24], "--o", color=C["synth"], lw=1.8, ms=6, zorder=4,
         label="confidence ranking, validation")
ax2.plot(covu, [4.88, 4.88, 4.68], "-x", color=C["base"], lw=1.4, ms=6, zorder=3,
         label="random abstention, test")
ax2.plot(covu, [4.27, 4.34, 4.20], ":x", color=C["base"], lw=1.2, ms=6, zorder=3,
         label="random abstention, validation")
ax2.invert_xaxis(); ax2.set_xticks(covu); ax2.set_xticklabels([f"{c}%" for c in covu])
ax2.set_xlabel("Coverage")
ax2.set_ylim(3.5, 9.0)
ax2.set_title("Unseen target sentences only", fontsize=FS["head"], weight="bold")
ax2.legend(loc="upper left", fontsize=FS["sub"], framealpha=0.95)
fig.suptitle("Declining the least-confident clips roughly doubles BLEU-4 at half coverage",
             fontsize=FS["head"], weight="bold")
fig.tight_layout()
fig.savefig("fig7_selective_prediction.png", dpi=300, bbox_inches="tight")
fig.savefig("fig7_selective_prediction.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# FIG 8 — CONTINUOUS IDENTIFICATION PROBE
# Source: results/CONTINUOUS_ID_PROBE_2026-09-09.md (45 glosses present in both datasets,
# 8 exemplars per class, clip-disjoint folds, nearest centroid, chance 2.2%).
# =====================================================================
arts = ["face", "body", "hands", "body + hands"]
iso = [11.1, 27.8, 40.0, 41.1]
cont = [3.3, 15.4, 18.1, 18.7]
x = np.arange(len(arts)); w = 0.36
fig, ax = plt.subplots(figsize=(TARGET_W, 3.78))
b1 = ax.bar(x - w/2, iso, w, color=C["pre"], zorder=3,
            label="isolated signs (citation form, MM-WLAuslan)")
b2 = ax.bar(x + w/2, cont, w, color=C["base"], zorder=3,
            label="continuous signing (conversational, Auslan-Daily)")
for bars_ in (b1, b2):
    for r in bars_:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.6, f"{r.get_height():.1f}%",
                ha="center", va="bottom", fontsize=FS["label"])
ax.plot([x[2] + w/2], [24.9], marker="*", ms=16, color=C["ens"], mec="white", ls="none", zorder=6,
        label="continuous hands, hand-local normalisation (24.9%)")
ax.axhline(2.2, ls=":", lw=1.4, color=C["grey"], zorder=2, label="chance (1 in 45 = 2.2%)")
ax.set_xticks(x); ax.set_xticklabels(arts)
ax.set_ylabel("Top-1 identification of 45 glosses (%)")
ax.set_ylim(0, 50)
ax.set_title("Pose carries sign identity in citation form;\n"
             "conversational signing loses more than half of it", fontsize=FS["head"], weight="bold", pad=10)
ax.legend(loc="upper left", fontsize=FS["sub"], framealpha=0.95, markerscale=0.7)
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TARGET_W, 2.62), sharey=True)
for ax, data, name in ((ax1, test_b, "Test BLEU-4 (beam-5)"),
                       (ax2, val_b, "Validation BLEU-4 (the selection metric)")):
    for s, v in data.items():
        ax.plot(ratios, v, "-o", color=C["grey"], lw=1.1, ms=4, alpha=0.9, zorder=3)
    placed = []  # seed labels at the line ends, nudged apart so close endpoints stay legible
    for y, s in sorted((v[-1], s) for s, v in data.items()):
        y_lab = max(y, placed[-1] + 0.13) if placed else y
        placed.append(y_lab)
        ax.text(ratios[-1] + 0.07, y_lab, f"s{s}", fontsize=FS["tiny"], color="#5f6368", va="center")
    ax.set_xlim(-0.12, 1.92)
    arr = np.array(list(data.values()), dtype=float)
    mean = [np.nan if np.isnan(col).all() else float(np.nanmean(col)) for col in arr.T]
    ax.plot(ratios, mean, "-o", color=C["synth"], lw=2.6, ms=7, zorder=5, label="mean of 3 seeds")
    ax.set_xticks(ratios); ax.set_xticklabels(["real only", "0.55:1", "1.1:1", "1.66:1"])
    ax.set_xlabel("Synthetic : real training clips")
    ax.set_title(name, fontsize=FS["head"], weight="bold")
ax1.plot([1.66], [8.54], "X", ms=11, color=C["base"], zorder=6,
         label="compute-matched control (real data repeated), s42")
ax2.text(0, 9.1, "control validation\nnot recorded", fontsize=FS["tiny"], color="#5f6368", ha="center")
ax1.set_ylabel("BLEU-4")
ax1.set_ylim(8.2, 11.9)
ax1.legend(loc="lower right", fontsize=FS["sub"], framealpha=0.95)
fig.suptitle("More correctly-timed synthetic data helps beyond 0.55:1 on test; "
             "validation does not separate the ratios", fontsize=FS["head"], weight="bold")
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
order = [("VLP", "CTC", "VLP pretraining vs DWPose + CTC"),
         ("RGBF", "CTC", "RGB fusion vs DWPose + CTC"),
         ("S6", "RGBF", "6-member ensemble vs RGB fusion"),
         ("F", "S6", "3 duration + 2 fusion (0.55:1) vs 6-member"),
         ("G1", "S6", "6 duration + 2 fusion (validation-best) vs 6-member"),
         ("Fprime", "S6", "3 duration + 2 fusion (1.1:1) vs 6-member"),
         ("Fprime", "F", "1.1:1 vs 0.55:1 duration members")]
missing = [f"{a}:{b}" for a, b, _ in order if (a, b) not in boot]
assert not missing, f"bootstrap output lacks {missing}"
fig, ax = plt.subplots(figsize=(TARGET_W, 3.15))
n_sig, sig_pairs = 0, []
for yi, (a, b, lab) in enumerate(order):
    obs, lo, hi, p = boot[(a, b)]
    sig = lo > 0 or hi < 0
    n_sig += sig
    if sig:
        sig_pairs.append((a, b))
    col = C["ens"] if sig else C["pre"]
    ax.plot([lo, hi], [yi, yi], color=col, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(obs, yi, "o", ms=8, color=col, mec="white", zorder=4)
    ax.text(2.95, yi, f"{obs:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", ha="left", fontsize=FS["sub"],
            color=C["ink"])
ax.axvline(0, color=C["ink"], lw=1.2, zorder=2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order], fontsize=FS["label"])
ax.invert_yaxis()
ax.set_xlim(-2.0, 4.6)
ax.set_xlabel("Difference in test BLEU-4 (95% paired-bootstrap interval)")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(fc=C["ens"], label="interval excludes zero"),
                   Patch(fc=C["pre"], label="not resolvable")], loc="upper center",
          bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=FS["sub"], framealpha=0.95)
title = ("Only the ensembling step is resolvable on an 800-sentence test split"
         if sig_pairs == [("S6", "RGBF")] else
         f"Paired-bootstrap resolution: {n_sig} of {len(order)} comparisons exclude zero")
ax.set_title(title, fontsize=FS["head"], weight="bold", pad=10)
fig.tight_layout()
fig.savefig("fig10_bootstrap_resolution.png", dpi=300, bbox_inches="tight")
fig.savefig("fig10_bootstrap_resolution.pdf", bbox_inches="tight")
plt.close(fig)

print("OK — wrote fig1..fig10 (png + pdf)")
import os
for f in sorted(os.listdir(".")):
    if f.startswith("fig") and f.endswith((".png", ".pdf")):
        print(f"  {f:34s} {os.path.getsize(f):>8d} bytes")
