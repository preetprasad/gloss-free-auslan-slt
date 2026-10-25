"""Figures corrected for the keypoint-extraction error (2026-10-25): fig4, fig14 and fig1.

  fig4_estimator_signer   AlphaPose (signer keypoints) against DWPose, test BLEU-4: bars = 3-seed means with 95% t-intervals,
                          dots = the seeds, the paired difference and its interval labelled. Read from readout.json.
  fig14_effect_sizes      the effect-size figure with its first row replaced by the corrected estimator difference
                          (the other rows are those of make_extra_figures.py, kept only if their numbers occur in the thesis text).
  fig1_results_ladder     the results ladder with the two first-listed-AlphaPose rungs relabelled and hatched (values unchanged).

usage: python make_signerfix_figures.py --out_dir DIR [--readout results/estimator_signer_keypoints_rerun/readout.json] [--tex_dir SignLanguage_Preet]
                                        [--only fig4,fig14,fig1]
readout.json format: PREP/schema.json -> files.readout.json_format. Nothing is written outside --out_dir. The ladder rows are
asserted against tab:res_ladder in --tex_dir/src15k/result.tex, so a stale ladder cannot be drawn."""
import argparse, json, math, os, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "src", "tools"))
import figstyle; figstyle.apply(plt)

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--readout", default=os.path.join(REPO, "results", "estimator_signer_keypoints_rerun", "readout.json"))
ap.add_argument("--tex_dir", default=os.path.join(REPO, "SignLanguage_Preet"))
ap.add_argument("--only", default="fig4,fig14,fig1")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
ONLY = set(args.only.split(","))

_FC = figstyle.C                                   # palette as in make_thesis_figures.py (fig1) / make_extra_figures.py
C = dict(grey=_FC["grey"], ink=_FC["ink"], pale=_FC["pale"], pose=_FC["body"], pre=_FC["accent"], fusion=_FC["good"],
         ens=figstyle.LADDER[6], base=_FC["warn"], synth=_FC["sup"])
TW = figstyle.TARGET_W; INK = C["ink"]
BLUE, ORANGE, GREY = _FC["accent"], _FC["warn"], _FC["grey"]
LAB, TICK = 9.0, 8.0
plt.rcParams.update({
    "axes.labelsize": LAB, "axes.titlesize": LAB, "figure.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": TICK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "grid.alpha": 1.0, "axes.grid.axis": "y", "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "hatch.linewidth": 0.6, "legend.frameon": False,
})
T975 = {2: 4.302652729911275, 3: 3.182446305284263, 4: 2.7764451051977987}
def check_overlaps(fig, name):
    """No eyes on the PNG? Then test it: report any two visible texts whose boxes overlap."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(r)) for t in ts]
    bad = [(a.get_text()[:28], b.get_text()[:28]) for i, (a, x) in enumerate(bb) for (b, y) in bb[i + 1:] if x.overlaps(y)]
    print(f"  [{name}] {len(ts)} text items, overlapping pairs: {bad if bad else 'none'}")
def save(fig, name, **kw):
    check_overlaps(fig, name)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(args.out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=300, **kw)
    plt.close(fig); print("wrote", os.path.join(args.out_dir, name + ".{pdf,png}"))
def tci(xs):
    m = st.mean(xs); h = T975[len(xs) - 1] * st.stdev(xs) / math.sqrt(len(xs)); return m, m - h, m + h

R = json.load(open(args.readout)) if ({"fig4", "fig14"} & ONLY) else None
if R is not None:
    prov = bool(R.get("PROVISIONAL"))
    eap, edw = R["test"]["eap"], R["test"]["edw"]
    assert len(eap) == len(edw) == len(R["seeds"]) >= 3
    diffs = [b - a for a, b in zip(eap, edw)]
    d_m, d_lo, d_hi = tci(diffs)
    print("test difference DWPose - AlphaPose: %+.2f [%+.2f, %+.2f]" % (d_m, d_lo, d_hi))

PROV = lambda ax: ax.set_title("PROVISIONAL (not independently recomputed)", loc="left", fontsize=6.5, color=ORANGE) if prov else None

# ------------------------------------------------------------------ fig4
if "fig4" in ONLY:
    fig, ax = plt.subplots(figsize=(5.8, 2.3), layout="constrained")
    series = [("AlphaPose\n(signer keypoints)", eap, GREY), ("DWPose", edw, C["pose"])]
    for i, (lab, v, col) in enumerate(series):
        m, lo, hi = tci(v)
        ax.bar(i, m, width=0.55, color=col, edgecolor="white", zorder=2)
        ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        for yv in (lo, hi): ax.plot([i - 0.07, i + 0.07], [yv, yv], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        ax.scatter([i + 0.0 + 0.13 * (k - 1) for k in range(len(v))], v, s=14, facecolors="white", edgecolors=INK, linewidths=0.8, zorder=5)
        ax.text(i + 0.33, m, f"{m:.2f}", ha="left", va="center", fontsize=LAB, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([s[0] for s in series]); ax.set_xlim(-0.6, 1.9)
    top = max(max(eap), max(edw), tci(eap)[2], tci(edw)[2]) + 0.35
    ax.set_ylim(0, top + 2.4)
    ax.plot([0, 0, 1, 1], [top, top + 0.3, top + 0.3, top], color=INK, lw=0.75, zorder=3)
    ax.text(0.5, top + 0.4, f"DWPose $-$ AlphaPose: ${d_m:+.2f}$\n95% interval ${d_lo:+.2f}$ to ${d_hi:+.2f}$",
            ha="center", va="bottom", fontsize=LAB, color=INK)
    ax.set_ylabel("Test BLEU-4 (mean of 3 seeds)"); ax.grid(False); ax.grid(True, axis="y")
    ax.annotate("dots: seeds; bars: means with\n95% $t$-intervals", (1.0, 0.3), xytext=(1.5, 0.6), fontsize=TICK - 0.5, color=INK, ha="left", va="bottom", annotation_clip=False)
    PROV(ax); save(fig, "fig4_estimator_signer")

# ------------------------------------------------------------------ fig14
if "fig14" in ONLY:
    texs = [os.path.join(args.tex_dir, d, f) for d, f in (("src", "result.tex"), ("src15k", "appendix_supp.tex"), ("src15k", "result.tex"))]
    THESIS = "".join(open(p).read() for p in texs if os.path.exists(p))
    NUMS = set(re.findall(r"\d+\.\d+|\d{1,3}(?:,\d{3})+", THESIS.replace("{,}", ",")))
    in_thesis = lambda *xs: all((f"{abs(x):.2f}" if isinstance(x, float) else str(x)) in NUMS for x in xs)
    E = [  # copied from make_extra_figures.py (2026-10-25); first row is the corrected estimator difference
        ("DWPose in place of AlphaPose (signer keypoints)", d_m, d_lo, d_hi, 1),
        ("Twice the real training pairs", 2.86, 1.78, 3.93, 1),
        ("Six-model ensemble against its strongest member type", 1.47, 0.40, 2.56, 1),
        ("Adopted recipe on the video-disjoint split", 0.94, 0.16, 1.72, 1),
        ("Pose augmentation", 0.70, -0.18, 1.57, 1),
        ("Stitched data and augmentation, News split", 0.34, 0.09, 0.58, 1),
        ("Wider synthetic vocabulary", 0.09, -0.45, 0.63, 0),
        ("Large synthetic pretraining (200,000 clips)", 0.04, -0.52, 0.60, 0),
        ("Sixty training epochs", -0.02, -0.74, 0.69, 0),
        ("Person tracking in pose extraction", -0.06, -0.66, 0.55, 0),
        ("Function-word signs removed", -0.09, -0.76, 0.59, 0),
        ("Two renderings of each synthetic sentence", -0.29, -1.15, 0.57, 0),
        ("No trimming of sign boundaries", -0.44, -0.90, 0.01, 0),
    ]
    kept = [e for i, e in enumerate(E) if i == 0 or in_thesis(e[1], e[2], e[3])]
    print("effect rows kept", len(kept), "of", len(E), "| dropped (numbers not in the thesis text):", [e[0] for e in E if e not in kept])
    try:
        import w11_readout
        runs = os.path.join(REPO, "results", "validation_selected_pretraining_rerun", "runs_pre")
        w = [w11_readout.load(runs, "P40", s)["beam5_test"] for s in (42, 101, 202)]
        c = [w11_readout.load(runs, "P0", s)["beam5_test"] for s in (42, 101, 202)]
        m, lo, hi = tci([a - b for a, b in zip(w, c)])
        kept.append(("Pose-text pretraining, 40 epochs (re-run)", m, lo, hi, 0))
    except Exception as ex:
        print("WARNING: P40-P0 row skipped (", ex, ")")
    kept.sort(key=lambda e: e[1])
    BOOT = "Six-model ensemble against its strongest member type"
    fig, ax = plt.subplots(figsize=(TW, 0.28 * len(kept) + 0.8), layout="constrained")
    for i, (lab, x, lo, hi, k) in enumerate(kept):
        col = BLUE if k else GREY
        ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot([x], [i], "o", ms=4.5, color=col, mfc=col if k else "white", mec=col, mew=1.2, zorder=3)
    ax.axvline(0, color=INK, lw=0.75, zorder=1)
    ax.set_yticks(range(len(kept)))
    ax.set_yticklabels([e[0] + (r" $^{\dagger}$" if e[0] == BOOT else "") for e in kept], fontsize=TICK)
    ax.set_ylim(-0.6, len(kept) - 0.4); ax.grid(False); ax.grid(True, axis="x")
    ax.set_xlabel("Difference in test BLEU-4 from paired control\n(95% interval)", ha="center")
    ia = [i for i, e in enumerate(kept) if e[4]]; ib = [i for i, e in enumerate(kept) if not e[4]]
    ax.text(0.02, sum(ia) / len(ia), "adopted or in\nreported system", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=LAB, color=BLUE)
    ax.text(0.98, sum(ib) / len(ib), "not adopted", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=LAB, color=GREY)
    PROV(ax); save(fig, "fig14_effect_sizes")

# ------------------------------------------------------------------ fig1
if "fig1" in ONLY:
    REF = C["grey"]
    models = [  # label, beam-5 test BLEU-4 (asserted against tab:res_ladder), colour key, unreplicated hatch, wrong-person hatch
        ("First-listed AlphaPose keypoints (control)", 2.87, "grey", False, True),
        ("+ pretraining, tuned decoding",              3.87, "grey", False, True),
        ("DWPose re-extraction (signer)",              8.78, "pose", False, False),
        ("+ CTC loss (3 seeds)",                       9.63, "pose", False, False),
        ("+ pretraining, 40 epochs",                  10.27, "pre", False, False),
        ("+ RGB fusion",                              10.33, "fusion", False, False),
        ("pretraining, 80 epochs*",                   10.71, "pre", True, False),
        ("+ checkpoint averaging*",                   10.89, "pre", True, False),
        ("stitched synthetic data, 0.55:1 (3 seeds)", 10.97, "synth", False, False),
        ("    at 1.1:1 synthetic:real (3 seeds)",     11.14, "synth", False, False),
        ("    + pose augmentation (3 seeds)",         11.34, "synth", False, False),
        ("6-member ensemble, beam width 5",           11.50, "ens", False, False),
        ("final ensemble: beam 10, length penalty 0.3", 11.80, "ens", False, False),
    ]
    tex = open(os.path.join(args.tex_dir, "src15k", "result.tex")).read()
    tab = tex.split("\\label{tab:res_ladder}")[1].split("\\end{tabular}")[0]
    rows = []
    for line in tab.splitlines():
        f = line.strip().removesuffix("\\\\").split("&")
        if len(f) == 4:
            m = re.fullmatch(r"(?:\\textbf\{)?([\d.]+)\}?", f[2].strip())
            if m: rows.append(float(m.group(1)))
    assert len(rows) == len(models), f"tab:res_ladder has {len(rows)} beam-5 rows, figure has {len(models)}"
    for t, (lab, v, *_r) in zip(rows, models): assert abs(t - v) < 1e-9, f"tab:res_ladder says {t} where the figure plots {v} ({lab})"
    labels = [m[0] for m in models]; vals = [m[1] for m in models]
    cols = [C[m[2]] for m in models]
    gap = np.cumsum([0] + [0.55 if models[i][2] != models[i - 1][2] else 0 for i in range(1, len(models))])
    y = np.arange(len(models)) + gap
    fig, ax = plt.subplots(figsize=(TW, 4.4))
    bars = ax.barh(y, vals, color=cols, edgecolor="white", height=0.68, zorder=3)
    for b, m in zip(bars, models):
        if m[3]: b.set_hatch("///")
        if m[4]: b.set_hatch("xxxx"); b.set_edgecolor("#5F6368")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.tick_params(axis="both", labelsize=TICK)
    ax.grid(False); ax.grid(True, axis="x"); ax.invert_yaxis()
    dur = y[labels.index("stitched synthetic data, 0.55:1 (3 seeds)")]
    for xv in (10.74, 11.08): ax.plot([xv, xv], [dur - 0.07, dur + 0.07], color=INK, lw=0.75, zorder=9, solid_capstyle="butt")
    ax.plot([10.74, 11.08], [dur, dur], color=INK, lw=0.75, zorder=9, solid_capstyle="butt")
    for v, n, h in ((9.20, "SL-Transformer 9.20", 0.16), (9.45, "SL-Luong 9.45", 0.10), (9.95, "GASLT 9.95", 0.04)):
        ax.axvline(v, ls="--", lw=0.75, color=REF, zorder=1)
        ax.plot([v, v], [1.0, 1.0 + h], ls="-", lw=0.75, color=REF, clip_on=False, transform=ax.get_xaxis_transform())
        ax.text(v + 0.08, 1.0 + h, n, fontsize=TICK, color=INK, ha="left", va="center", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylim(y[-1] + 0.6, y[0] - 0.6)
    ax.set_xlabel("Test BLEU-4, official Auslan-Daily Communication split"); ax.set_xlim(0, 12.5); ax.set_xticks([0, 3, 6, 9, 12])
    leg = [Patch(fc=C["grey"], ec="#5F6368", hatch="xxxx", label="AlphaPose, first-listed (mostly a non-signer)"),
           Patch(fc=C["pose"], label="DWPose keypoints (signer)"), Patch(fc=C["pre"], label="pretraining"),
           Patch(fc=C["fusion"], label="RGB fusion"), Patch(fc=C["synth"], label="synthetic data / augmentation"),
           Patch(fc=C["ens"], label="ensemble"), Patch(fc="white", ec=INK, hatch="///", label="* unreplicated"),
           Line2D([], [], ls="-", lw=0.75, c=INK, marker="|", ms=4, mew=0.75, label="range of 3 seeds")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=TICK, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); save(fig, "fig1_results_ladder")
