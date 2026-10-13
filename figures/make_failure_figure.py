"""FIG C10 -- a named failure taxonomy for the whole test split.

The figure survey behind results/FIGURE_STANDARD_2026-09-22.md examined roughly 35 dataset-paper
figures and found NOT ONE marked failure case; across 79 papers only OpenPose presents a labelled
failure taxonomy with named causes. This is that figure for gloss-free Auslan translation.

Every count is computed over all 800 clips of the official Communication test split from the saved
VLP-80 beam-5 decodes (results/semantic_adequacy_grading/hyps_vlp.jsonl), against the 12,439-sentence
training manifest (results/_kaya_train_manifest.tsv). Content words are the non-stopword alphabetic
tokens, using the same stopword list the stitching lexicon uses.

The modes are mutually exclusive and exhaustive; the examples are real rows, shortened only by
choosing among clips whose reference is 4-9 words so the pair fits the line.
"""
import collections, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "tools"))
import figstyle
figstyle.apply(plt)
C, FS, TARGET_W = figstyle.C, figstyle.FS, figstyle.TARGET_W
from stitch_lexicon import STOP

rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/semantic_adequacy_grading/hyps_vlp.jsonl"))]
train = {l.rstrip("\n").split("\t")[1].strip()
         for l in open(os.path.join(ROOT, "results/_kaya_train_manifest.tsv")) if "\t" in l}
content = lambda t: [w for w in t.split() if w not in STOP and w.isalpha()]

def category(ref, hyp):
    rc, hc = content(ref), content(hyp)
    if not rc:                          return "reference has no content word"
    hit = [w for w in rc if w in hc]
    if hyp.strip() == ref.strip():      return "exact match"
    if len(hit) == len(rc):             return "all content words, reworded"
    if hit:                             return "partial identification"
    if hyp.strip() in train:            return "memorised fallback"
    return "novel, nothing correct"

counts = collections.Counter(category(r["ref"], r["hyp_beam"]) for r in rows)
N = sum(counts.values())

# order: a meaningful sequence from most to least correct (exact > reworded > partial > nothing
# correct), with the unscoreable remainder last; colour by outcome, hatch as the redundant channel
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "src").is_dir())
_sys.path.insert(0, str(_root / "src" / "tools"))
from restore_references import sentence as S   # dataset sentences are named by SHA-1 digest and read from the spreadsheet
SPEC = [
    ("exact match",                 C["good"],  None,  "",
     S("1f3e2360b132d560b732896727cddd648aee6428"), S("1f3e2360b132d560b732896727cddd648aee6428")),
    ("all content words, reworded", C["good"],  "///", "every content word present, wording differs",
     S("8b67b2afd9ca0d8a9d4860e8c5512070aae6d82c"), S("cef9405048a1eed5e2d70d1846f2ef159fe905fb")),
    ("partial identification",      C["sup"],   "...", "some signs found, some not",
     S("c8e0066243d1df8c0adcbee85fbb13c094715bbd"), S("91717878befa63e0e6356961da72c0a1d53fcec0")),
    ("memorised fallback",          C["warn"],  "xx",  "a verbatim training sentence, nothing correct",
     S("0580869881cfa0555295bdd2be5956120250be5c"), S("bbf795dc99e2fb23d150da80ee659b22318efc91")),
    ("novel, nothing correct",      C["warn"],  "\\\\\\", "not a training sentence, still nothing correct",
     S("5294ec8ef8ae492153698430bf6c25a4d6e491d1"), "sally how do we make it now ."),
    ("reference has no content word", C["grey"], "--", "not scoreable",
     S("b29480caa2d65e6e72aeedcefeef56af29189be5"), "thank you ."),
]

# every example is a real (reference, output) pair of the test set and falls in the category it illustrates
_pairs = {(r["ref"].strip(), r["hyp_beam"].strip()) for r in rows}
for _name, _, _, _, _ref, _hyp in SPEC:
    assert (_ref.strip(), _hyp.strip()) in _pairs and category(_ref, _hyp) == _name, (_name, _ref, _hyp)

# built at the printed width with explicit margins (no tight bbox), so the PDF is exactly 160 mm wide
fig = plt.figure(figsize=(TARGET_W, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.06,
                      left=0.25, right=0.995, bottom=0.13, top=0.94)
ax = fig.add_subplot(gs[0, 0])
ys = np.arange(len(SPEC))[::-1]
Z95 = 1.96                                   # 95% Wilson score interval for a binomial share
for y, (name, col, hatch, _, _, _) in zip(ys, SPEC):
    pct = 100 * counts[name] / N
    b = ax.barh(y, pct, color=col, edgecolor="white", height=0.66, zorder=3)
    if hatch:
        b[0].set_hatch(hatch); b[0].set_edgecolor("white")
    _p = counts[name] / N
    _d = 1 + Z95 ** 2 / N
    _c = (_p + Z95 ** 2 / (2 * N)) / _d
    _h = Z95 * np.sqrt(_p * (1 - _p) / N + Z95 ** 2 / (4 * N * N)) / _d
    _lo, _hi = 100 * max(_c - _h, 0.0), 100 * (_c + _h)
    ax.errorbar(pct, y, xerr=[[max(pct - _lo, 0.0)], [max(_hi - pct, 0.0)]], fmt="none",
                ecolor=C["ink"], elinewidth=0.75, capsize=2, capthick=0.75, zorder=4)
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in SPEC], fontsize=8)
ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel("Share of test clips (%)", fontsize=9)
ax.grid(axis="x", color="#DDDDDD", linewidth=0.4, alpha=1.0); ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(0.6)
ax.tick_params(axis="x", direction="out", width=0.6, length=3, labelsize=8)
ax.tick_params(axis="y", length=0, pad=4)
ax.set_ylim(-0.6, len(SPEC) - 0.4)

axe = fig.add_subplot(gs[0, 1]); axe.axis("off"); axe.set_xlim(0, 1)
axe.set_ylim(-0.6, len(SPEC) - 0.4)
for y, (name, col, hatch, note, ref, hyp) in zip(ys, SPEC):
    # swatch repeating the bar's colour AND hatch, so each example is tied to its mode without colour
    sw = Rectangle((0.0, y - 0.38), 0.035, 0.76, facecolor=col, edgecolor="white",
                   hatch=hatch, linewidth=0.0)
    axe.add_patch(sw)
    off = 0.0 if note else 0.15          # centre two-line entries on the swatch
    axe.text(0.055, y + 0.30 - off, f"signed  “{ref}”", fontsize=8, color=C["ink"], va="center")
    axe.text(0.055, y + 0.0 - off, f"model   “{hyp}”", fontsize=8, color=C["ink"], va="center")
    if note:
        axe.text(0.055, y - 0.30, note, fontsize=8, color=C["ink"], va="center", style="italic")

# panel labels, 10 pt bold, top-left and outside the axes
fig.text(0.005, 0.99, "(a)", fontsize=10, fontweight="bold", ha="left", va="top")
fig.text(axe.get_position().x0, 0.99, "(b)", fontsize=10, fontweight="bold", ha="left", va="top")

fig.savefig(os.path.join(HERE, "figC10_failure_modes.png"), dpi=300)
fig.savefig(os.path.join(HERE, "figC10_failure_modes.pdf"))
print("figC10 written")
for name, *_ in SPEC:
    _p = counts[name] / N
    _d = 1 + 1.96 ** 2 / N
    _c = (_p + 1.96 ** 2 / (2 * N)) / _d
    _h = 1.96 * np.sqrt(_p * (1 - _p) / N + 1.96 ** 2 / (4 * N * N)) / _d
    print(f"  {name:32s} {counts[name]:4d}  {100*_p:5.1f}%  "
          f"95% Wilson CI [{100*max(_c-_h,0):4.1f}, {100*(_c+_h):4.1f}]")
