"""Four further figures for the dissertation (October 2026). Every plotted number is read from saved run
files or is asserted to occur in the Results chapter, so a figure cannot drift from the text.

  fig11_pretraining_rerun   pose-text pretraining length, three seeds per setting (wave 11)
  fig12_learning_curve      real-data learning curve
  fig13_latency_breakdown   time per clip by stage
  fig14_effect_sizes        paired differences in test BLEU-4 with 95% intervals, all controlled comparisons
"""
import os, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "src", "tools"))
import figstyle; figstyle.apply(plt)
import w11_readout
THESIS = "".join(open(os.path.join(REPO, "SignLanguage_Preet", d, f)).read()
                 for d, f in (("src", "result.tex"), ("src15k", "appendix_supp.tex"), ("src15k", "result.tex")))
NUMS = set(re.findall(r"\d+\.\d+|\d{1,3}(?:,\d{3})+", THESIS.replace("{,}", ",")))
def in_thesis(*xs): return all((f"{abs(x):.2f}" if isinstance(x, float) else str(x)) in NUMS for x in xs)
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ── presentation only: house colours from figstyle, and text sizes for a 12 pt body ──────────────
C = figstyle.C
TW = figstyle.TARGET_W                      # 6.30 in, the full text width
INK = C["ink"]
BLUE, ORANGE, GREY, GREEN = C["accent"], C["warn"], C["grey"], C["good"]
LAB, TICK = 9.0, 8.0                        # axis labels / annotations 9 pt, tick labels 8 pt (pt on page)
plt.rcParams.update({
    "axes.labelsize": LAB, "axes.titlesize": LAB, "figure.labelsize": LAB,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    # standard 1.3 / 1.4: thin spines, short outward ticks, light grid on the value axis only, behind the data
    "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "grid.alpha": 1.0, "axes.grid.axis": "y", "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5,
})

# ---------------------------------------------------------------- fig11: pretraining length, three seeds
runs = os.path.join(REPO, "results", "validation_selected_pretraining_rerun", "runs_pre")
arms = [("P0", 0), ("P40", 40), ("P80", 80), ("P128", 128), ("P256", 256)]
rows = {a: [w11_readout.load(runs, a, s) for s in (42, 101, 202)] for a, _ in arms}
EARLIER_DX = 0.18                           # display offset only: crosses sit beside the mean bar, not on it
fig, axes = plt.subplots(1, 2, figsize=(TW, 2.6), sharey=True, layout="constrained")
for ax, key, tag in ((axes[0], "beam5_val", "(a) validation"), (axes[1], "beam5_test", "(b) official test")):   # (a) validation, (b) test -- named in the caption
    xs = range(len(arms))
    for i, (a, _) in enumerate(arms):
        ax.scatter([i] * 3, [r[key] for r in rows[a]], s=18, color=BLUE, alpha=0.5, edgecolors="none", zorder=3)
    means = [st.mean(r[key] for r in rows[a]) for a, _ in arms]
    ax.plot(list(xs), means, "-", marker="_", ms=10, mew=1.5, color=BLUE, lw=1.5, zorder=6)
    ax.annotate("mean", (len(arms) - 1, means[-1]), textcoords="offset points", xytext=(9, 0), ha="left", va="center",
                fontsize=TICK, color=BLUE)
    if key == "beam5_test":
        ax.scatter([x + EARLIER_DX for x in [1, 2, 3, 4]], [10.27, 10.71, 9.69, 9.66], marker="x", s=20,
                   linewidths=1.2, color=INK, zorder=5)
        ax.annotate("earlier single runs", (2 + EARLIER_DX, 10.71), textcoords="offset points", xytext=(6, 0),
                    ha="left", va="center", fontsize=TICK, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels(["none", "40", "80", "128", "256"]); ax.set_xlim(-0.4, 5.0)
    ax.annotate(tag, (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(-26 if key == "beam5_val" else -10, 6), ha="left", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
axes[0].set_ylabel("BLEU-4 (beam width 5)")
fig.supxlabel("Pose-text pretraining (epochs)", fontsize=LAB)
save(fig, "fig11_pretraining_rerun")

# ---------------------------------------------------------------- fig12: real-data learning curve
pairs = [3141, 6241, 12439]; val = [6.80, 7.89, 10.51]; test = [6.03, 7.04, 9.90]; unseen = [2.27, 3.15, 5.00]
assert in_thesis(*val, *test, *unseen, 7.63, 3.53) and all(f"{p:,}" in NUMS for p in pairs)
EPI_X = 6241 * 1.05                         # display offset only, so the diamonds sit beside the 6,241 markers
fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
ax.plot(pairs, test, "-o", color=BLUE, lw=1.5, ms=4.5, zorder=2)
ax.plot(pairs, unseen, "--s", color=INK, lw=1.5, ms=4.5, zorder=2)
ax.scatter([EPI_X], [7.63], marker="D", s=22, facecolors="white", edgecolors=BLUE, linewidths=1.1, zorder=3)
ax.scatter([EPI_X], [3.53], marker="D", s=22, facecolors="white", edgecolors=INK, linewidths=1.1, zorder=3)
ax.set_xscale("log", base=2); ax.set_xticks(pairs); ax.set_xticklabels([f"{p:,}" for p in pairs]); ax.minorticks_off()
ax.set_xlim(2800, 22000)                    # room on the right for the direct line labels
ax.set_xlabel("Real training pairs (log scale)"); ax.set_ylabel("BLEU-4 (mean of five seeds)"); ax.set_ylim(0, 11)
for ys, txt, col in ((test, "official test\n(800 clips)", BLUE), (unseen, "unseen targets\n(570 clips)", INK)):
    ax.annotate(txt, (pairs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0), ha="left", va="center",
                fontsize=TICK, color=col)
ax.annotate("whole episodes", (EPI_X, 7.63), textcoords="offset points", xytext=(-6, 5), ha="right", va="center",
            fontsize=TICK, color=INK)
save(fig, "fig12_learning_curve")

# ---------------------------------------------------------------- fig13: time per clip by stage
stages = [("pose estimation", 1968.5, 4448.8, GREY), ("translation", 91.8, 217.7, BLUE), ("speech synthesis", 660.8, 697.1, ORANGE)]
assert all(f"{m:.1f}" in THESIS and f"{p:.1f}" in THESIS for _, m, p, _ in stages)
STAGE_STYLE = {  # inherited stages grey, the contribution in accent; hatching separates the two greys
    "pose estimation": dict(color=GREY, hatch=None, ec="white"),
    "translation": dict(color=BLUE, hatch=None, ec="white"),
    "speech synthesis": dict(color="#D5D9DD", hatch="///", ec="#5F6368"),
}
fig, ax = plt.subplots(figsize=(TW, 2.0), layout="constrained")
left = 0
for name, med, _p95, _ in stages:   # median bar only: stacked per-stage 95th percentiles are not the chain's 95th percentile
    sty = STAGE_STYLE[name]
    ax.barh(0, med, left=left, height=0.5, color=sty["color"], hatch=sty["hatch"], edgecolor=sty["ec"],
            linewidth=0.6, zorder=2)
    cx = left + med / 2
    if med > 500:                   # stage named directly above its segment, value inside
        ax.text(cx, 0, f"{med:.0f}", ha="center", va="center", color=INK, fontsize=LAB, zorder=3,
                bbox=dict(fc="white", ec="none", pad=1.2) if sty["hatch"] else None)
        ax.text(cx, 0.30, name, ha="center", va="bottom", color=INK, fontsize=LAB)
    else:                           # narrow segment: name and value on a raised leader, clear of the neighbour's name
        ax.annotate(f"{name}, {med:.0f}", xy=(cx, 0.25), xytext=(cx, 0.62), ha="center", va="bottom", fontsize=LAB,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.75))
    left += med
ax.set_ylim(-0.45, 0.95); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.grid(False); ax.grid(True, axis="x")
ax.set_xlim(left=0); ax.set_xlabel("Median time per clip (ms), laptop CPU")
save(fig, "fig13_latency_breakdown")

# ---------------------------------------------------------------- fig14: paired differences with 95% intervals
E = [  # label, difference, low, high, kind (adopted / not adopted)
    ("DWPose in place of AlphaPose keypoints", 4.97, 4.41, 5.53, 1),
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
kept = [e for e in E if in_thesis(e[1], e[2], e[3])]
print("effect rows kept", len(kept), "of", len(E), "| dropped:", [e[0] for e in E if e not in kept])
w = [r["beam5_test"] for r in rows["P40"]]; c = [r["beam5_test"] for r in rows["P0"]]
d = [a - b for a, b in zip(w, c)]; m = st.mean(d); h = 4.303 * st.stdev(d) / 3 ** 0.5
kept.append(("Pose-text pretraining, 40 epochs (re-run)", m, m - h, m + h, 0))
kept.sort(key=lambda e: e[1])
BOOT = "Six-model ensemble against its strongest member type"     # the one row with a paired-bootstrap interval
fig, ax = plt.subplots(figsize=(TW, 0.28 * len(kept) + 0.8), layout="constrained")
for i, (lab, x, lo, hi, k) in enumerate(kept):
    col = BLUE if k else GREY
    ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle="butt", zorder=2)
    ax.plot([x], [i], "o", ms=4.5, color=col, mfc=col if k else "white", mec=col, mew=1.2, zorder=3)
ax.axvline(0, color=INK, lw=0.75, zorder=1)
ax.set_yticks(range(len(kept)))
ax.set_yticklabels([e[0] + (r" $^{\dagger}$" if e[0] == BOOT else "") for e in kept], fontsize=TICK)
ax.set_ylim(-0.6, len(kept) - 0.4)
ax.grid(False); ax.grid(True, axis="x")      # the value axis is x here
ax.set_xlabel("Difference in test BLEU-4 from paired control\n(95% interval)", ha="center")
ia = [i for i, e in enumerate(kept) if e[4]]; ib = [i for i, e in enumerate(kept) if not e[4]]
ax.text(0.02, sum(ia) / len(ia), "adopted or in\nreported system", transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=LAB, color=BLUE)          # empty left margin beside the adopted rows
ax.text(0.98, sum(ib) / len(ib), "not adopted", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=LAB, color=GREY)         # empty right margin beside the other rows
save(fig, "fig14_effect_sizes")
print("P40 - P0 test: %.2f [%.2f, %.2f]" % (m, m - h, m + h))

# ---------------------------------------------------------------- fig15: ensemble size against decoding time
ens = {5: [(1, 9.53, 35), (3, 11.31, 103), (6, 12.19, 207)], 10: [(1, 9.78, 44), (3, 11.47, 133), (6, 12.47, 269)]}
assert all(re.search(rf"{b:.2f}\}}?\s*&\s*{ms}\s*&", THESIS) for v in ens.values() for _, b, ms in v), "ensemble cost table changed"
fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
for beam, col, fmt in ((5, BLUE, "-o"), (10, INK, "--s")):
    ax.plot([ms for *_, ms in ens[beam]], [b for _, b, _ in ens[beam]], fmt, color=col, lw=1.5, ms=4.5, zorder=3)
    for n, b, ms in ens[beam]:
        # ensemble size: beam 5 above-left of the marker; beam 10 directly below (checked against both lines)
        off, ha, va = ((-6, 5), "right", "bottom") if beam == 5 else ((0, -8), "center", "top")
        ax.annotate(f"{n}", (ms, b), textcoords="offset points", xytext=off, ha=ha, va=va, fontsize=TICK, color=col)
# direct series labels at the right ends; beam 5 sits up-left of its end because the beam-10 line passes to its right
ax.annotate("beam width 5", (ens[5][-1][2], ens[5][-1][1]), textcoords="offset points", xytext=(-16, 8),
            ha="right", va="bottom", fontsize=TICK, color=BLUE)
ax.annotate("beam width 10", (ens[10][-1][2], ens[10][-1][1]), textcoords="offset points", xytext=(7, 0),
            ha="left", va="center", fontsize=TICK, color=INK)
ax.set_xlim(20, 340); ax.set_ylim(9.2, 12.75); ax.set_yticks([10, 11, 12])
ax.set_xlabel("Median decoding time per clip (ms), laptop CPU"); ax.set_ylabel("Validation BLEU-4")
save(fig, "fig15_ensemble_cost")

# ---------------------------------------------------------------- fig16: the confidence gate on people who are not signing
gate = [("test-set signers", "53.5", BLUE), ("non-signing people\nin the same videos", "53.9", ORANGE),
        ("non-signers who\nhardly move", "50.3", ORANGE), ("signer's pose frozen\nat one frame", "46.5", GREY)]
assert all(v in NUMS for _, v, _ in gate), "non-signer figures changed"
HATCH = [None, "//", "//", ".."]
fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
for i, ((_, v, col), hat) in enumerate(zip(gate, HATCH)):
    ax.bar(i, float(v), color=col, hatch=hat, edgecolor="white", linewidth=0, width=0.6, zorder=2)
ax.axhline(50, color=INK, lw=0.75, ls="--", zorder=1)
ax.text(-0.42, 66, "dashed line: threshold fitted to voice 50% of validation clips", ha="left", va="center", fontsize=TICK, color=INK)
for i, (_, v, _) in enumerate(gate):   # labels sit above the bar or the 50% line, whichever is higher
    ax.text(i, max(float(v), 50) + 1.5, v + "%", ha="center", va="bottom", fontsize=LAB, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([n for n, *_ in gate], fontsize=TICK); ax.set_ylim(0, 70); ax.set_ylabel("Clips voiced (%)")
ax.set_yticks([0, 20, 40, 60])
ax.grid(axis="x", visible=False)
save(fig, "fig16_nonsigner_gate")

# ---------------------------------------------------------------- fig17: final ensemble by signer
sig = [("presenter", 14.39, 389), ("costumed character", 9.97, 350), ("other signers", 3.12, 61)]
BREAK = open(os.path.join(REPO, "results", "SIGNER_BREAKDOWN_2026-09-22.md")).read()
assert in_thesis(*[b for _, b, _ in sig], 11.80) and "ID 1 (389), ID 2 (350)" in BREAK and sum(n for *_, n in sig) == 800
fig, ax = plt.subplots(figsize=(4.6, 3.3), layout="constrained")
for i, ((_, b, _), col, hat) in enumerate(zip(sig, [BLUE, BLUE, GREY], [None, None, "//"])):
    ax.bar(i, b, color=col, hatch=hat, edgecolor="white", linewidth=0, width=0.6, zorder=2)
for i, (_, b, _) in enumerate(sig): ax.text(i, b + 0.3, f"{b:.2f}", ha="center", va="bottom", fontsize=LAB, color=INK)
ax.axhline(11.80, color=INK, lw=0.75, ls="--", zorder=1)
ax.text(2.42, 12.1, "all 800 clips: 11.80", ha="right", va="bottom", fontsize=TICK, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{n}\n({k} clips)" for n, _, k in sig], fontsize=TICK)
ax.set_ylabel("BLEU-4, final ensemble"); ax.set_ylim(0, 16.5); ax.set_yticks([0, 5, 10, 15]); ax.grid(axis="x", visible=False)
save(fig, "fig17_signer_breakdown")
