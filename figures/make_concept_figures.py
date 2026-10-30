"""Conceptual figures for the Introduction, Literature Review and Methodology chapters.

The Results-chapter figures live in make_thesis_figures.py. These are the figures that make the
TASK legible to a non-specialist reader -- the gap the literature-review marker identified
("some key concepts (gloss, CTC) are not adequately explained or explained too late";
"good presentation, but missed opportunity to include graphics").

Every panel that shows data is built from real project files, named per figure. Nothing here is
traced from a published figure.

Run:  python make_concept_figures.py      (writes PNG + PDF beside this file)
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POSE_DIR = os.path.join(ROOT, "results", "dwpose_experiment", "dwpose_clips")
RGB_DIR = os.path.join(ROOT, "results", "dwpose_experiment", "rgb_clips", "Signer")

import sys as _sys0
_sys0.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as _fsty          # the one house style shared with make_thesis_figures.py
_fsty.apply(plt)

# ── HOUSE STYLE ──────────────────────────────────────────────────────────────────────
# Measured, not guessed. Across 75 published papers from CVPR/ICCV/ECCV/NeurIPS/ICLR/ACL/EMNLP
# (results/figure_standard/pdfs, font sizes read out of the PDFs with PyMuPDF): body text is
# 10.0 pt, the median text INSIDE a figure is 7.2 pt (0.70 of body), and the 5th percentile is
# 5.2 pt. Most papers do go below 6 pt somewhere -- 52 of 75 -- so a blanket "nothing under 6pt"
# rule is not the real convention; the real convention is that the TYPICAL label sits near 7 pt.
#
# A figure is therefore built AT ITS PRINTED WIDTH, so that a fontsize of 8 means 8 pt on paper.
# Building a 11.6in-wide canvas and letting LaTeX shrink it to 6.3in was turning 8.4 pt labels
# into 4.6 pt, i.e. our median label was as small as a published paper's smallest.
#   thesis  \linewidth on A4 with 25mm margins = 160mm = 6.30in
#   a two-column conference figure spanning both columns is ~7.0in, close enough to reuse
TARGET_W = _fsty.TARGET_W
# The printed review asked for ~9 pt in-figure text; nothing in these figures goes below 7.5 pt.
# Standard 1.2: one size for tick labels (8 pt), one for labels and annotations (9 pt), panel
# letters 10 pt bold. Nothing is set below 7.6 pt.
FS = dict(head=9.0, label=9.0, sub=8.0, tiny=8.0)   # final points, on the page
PANEL_FS = 10.0
# Standard 1.3/1.4 on top of the house style: 0.6 pt spines, 0.6 x 3 pt outward ticks, no grid
# unless asked for, and then #DDDDDD at 0.4 pt on the value axis only.
plt.rcParams.update({
    "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "grid.alpha": 1.0,
})

# The figures must be set in the DOCUMENT's font, not the plotting tool's default sans. Latin
# Modern is the OpenType form of Computer Modern, which is what this thesis's report class uses.
_LM = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm"
if os.path.isdir(_LM):
    import matplotlib.font_manager as _fm
    for _f in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf",
               "lmroman10-bolditalic.otf", "lmsans10-regular.otf", "lmsans10-bold.otf"):
        _p = os.path.join(_LM, _f)
        if os.path.exists(_p):
            _fm.fontManager.addfont(_p)

C = dict(_fsty.C)
# legacy names used below, mapped onto the house palette. pose/pre/fusion are the frozen
# anatomical colours and are used ONLY for body/hands/face from here on.
C.update(pose=_fsty.ART["body"], pre=_fsty.ART["hands"], fusion=_fsty.ART["face"],
         synth=_fsty.C["sup"], base=_fsty.C["warn"], ens=_fsty.LADDER[-1])

# ── COCO-WholeBody-133 layout, as produced by DWPose (RTMW-DW-x-l) ──────────────
BODY, FEET, FACE, LHAND, RHAND = slice(0, 17), slice(17, 23), slice(23, 91), slice(91, 112), slice(112, 133)
BODY_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
              (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
CONF = 0.3


def load_clip(clip):
    kps = np.load(os.path.join(POSE_DIR, f"{clip}.npz"))["KPS"]
    return kps


def read_frames(clip, idxs):
    cap = cv2.VideoCapture(os.path.join(RGB_DIR, f"{clip}_signer.mp4"))
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def draw_skeleton(ax, kp, lw=1.25, body_ms=8, hand_ms=2.5, face_ms=1.5, alpha=1.0, hand_lw=None):
    """Draw one frame's 133 keypoints. Colour by articulator so the reader can see that the
    hands carry fine structure the body does not."""
    def seg(sl, edges, col, w, ms):
        pts = kp[sl]
        for a, b in edges:
            if pts[a, 2] > CONF and pts[b, 2] > CONF:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        color=col, lw=w, zorder=3, alpha=alpha, solid_capstyle="round")
        m = pts[:, 2] > CONF
        ax.scatter(pts[m, 0], pts[m, 1], s=ms, color=col, zorder=4,
                   edgecolors="white", linewidths=0.35 if ms >= 6 else 0.0, alpha=alpha)
    hl = lw * 0.6 if hand_lw is None else hand_lw
    seg(BODY, BODY_EDGES, C["pose"], lw, body_ms)
    seg(LHAND, HAND_EDGES, C["pre"], hl, hand_ms)
    seg(RHAND, HAND_EDGES, C["pre"], hl, hand_ms)
    fp = kp[FACE]
    m = fp[:, 2] > CONF
    ax.scatter(fp[m, 0], fp[m, 1], s=face_ms, color=C["fusion"], zorder=4, alpha=alpha)


def tidy(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C["grey"]); s.set_linewidth(0.6)


def panel_label(fig, ax, letter, ha="right"):
    """Rule 4.1: 10 pt bold panel letter at the top-left, outside the axes."""
    ax.apply_aspect()
    pos = ax.get_position()
    dx = (-3.0 if ha == "right" else 0.0) / 72.0 / fig.get_figwidth()
    dy = 3.0 / 72.0 / fig.get_figheight()
    fig.text(pos.x0 + dx, pos.y1 + dy, f"({letter})", fontsize=PANEL_FS, fontweight="bold",
             color=C["ink"], ha=ha, va="bottom")


def save(fig, stem):
    """Write PNG + PDF at exactly the built width (rule 1.1): the box is tight vertically but
    always spans the full figure width, so LaTeX never rescales the figure."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())          # inches
    wf = fig.get_figwidth()
    x0, x1 = min(0.0, bb.x0), max(wf, bb.x1)
    if x1 - x0 > wf + 0.01:
        print(f"WARNING {stem}: content is {x1 - x0:.2f} in wide, built at {wf:.2f} in")
    box = Bbox.from_extents(x0, bb.y0 - 0.03, x1, bb.y1 + 0.03)
    fig.savefig(stem + ".png", dpi=300, bbox_inches=box)
    fig.savefig(stem + ".pdf", bbox_inches=box)


# =====================================================================
# FIG C1 — WHAT THE MODEL ACTUALLY SEES, AND WHAT IT IS TOLD
# Real clip video_16_130 (Auslan-Daily Communication, train split).
#   pose      results/dwpose_experiment/dwpose_clips/video_16_130.npz  (125 frames, 133 kpts)
#   video     results/dwpose_experiment/rgb_clips/Signer/video_16_130_signer.mp4
#   subtitle  data/auslan-daily/translation/Auslan-Daily_Communication.xlsx
#   gloss     Dataset Split Table/Sign Spotting/Sign_Spotting.xlsx -> OIL, frames 82-110
# The point of the figure: the model consumes row (b) only, and the sole supervision is the
# English sentence. One sign in this clip carries a gloss label; the other eight words do not.
# Clip chosen for stable framing: many Auslan-Daily clips span a camera cut (video_14_73, for
# instance, switches to a close-up at frame ~30), which would make the panels incomparable.
# Selection criteria, over the 37 local clips that have both a subtitle and a spotted gloss:
# both shoulders tracked in >=95% of frames, and the largest single-frame jump in shoulder
# width or torso centre below 4% of shoulder width (video_16_130: 3.4%, the lowest).
# =====================================================================
CLIP = "video_16_130"
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "src").is_dir())
_sys.path.insert(0, str(_root / "src" / "tools"))
from restore_references import sentence as S   # dataset sentences are named by SHA-1 digest and read from the spreadsheet
SUBTITLE = S("3a03098af76f86fdd1f74ae0c61e2d6441a216db")
GLOSS, G0, G1 = "OIL", 82, 110

kps = load_clip(CLIP)
T = len(kps)
show = [6, 34, 62, 90, 118]
frames = read_frames(CLIP, show)

# A common crop for both rows, so the video frame and the skeleton are directly comparable.
# Box: everything above the hips, plus wherever the hands travel, with a 12% margin.
vis = kps[..., 2] > CONF
pts = kps[:, :133, :2][vis[:, :133]]
head_hands = np.concatenate([kps[:, :11, :2][vis[:, :11]], kps[:, 91:133, :2][vis[:, 91:133]]])
x0, x1 = head_hands[:, 0].min(), head_hands[:, 0].max()
y0, y1 = head_hands[:, 1].min(), head_hands[:, 1].max()
mx, my = 0.12 * (x1 - x0), 0.12 * (y1 - y0)
side = max(x1 - x0 + 2 * mx, y1 - y0 + 2 * my)
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
BX0, BX1 = cx - side / 2, cx + side / 2
BY0, BY1 = cy - side / 2, cy + side / 2

fig = plt.figure(figsize=(TARGET_W, 4.60))
gs = fig.add_gridspec(3, len(show), height_ratios=[1.0, 1.0, 0.46], hspace=0.30, wspace=0.05)

H, W = frames[show[0]].shape[:2]
ix0, ix1 = int(max(0, BX0)), int(min(W, BX1))
iy0, iy1 = int(max(0, BY0)), int(min(H, BY1))
for j, f in enumerate(show):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(frames[f][iy0:iy1, ix0:ix1]); tidy(ax)
    ax.set_title(f"frame {f}", fontsize=FS["sub"], color=C["ink"], pad=3)
    if j == 0:
        ax.set_ylabel("(a) video\nas recorded", fontsize=FS["head"], labelpad=6)

for j, f in enumerate(show):
    ax = fig.add_subplot(gs[1, j])
    draw_skeleton(ax, kps[f], lw=2.0, body_ms=16, hand_ms=7, face_ms=2.0)
    ax.set_xlim(BX0, BX1); ax.set_ylim(BY1, BY0); ax.set_aspect("equal")
    tidy(ax)
    if j == 0:
        ax.set_ylabel("(b) the 133\nkeypoints the\nmodel consumes", fontsize=FS["head"], labelpad=6)
    if j == 0:
        _row_b_top = ax.get_position().y1

# articulator key in the gap ABOVE row (b), outside every panel. Line weight and mark size
# repeat the colour code (thick line + large dots / thin line + small dots / dots only), so the
# three articulators stay separable in greyscale.
from matplotlib.lines import Line2D
_key = [Line2D([], [], color=C["pose"], lw=2.0, marker="o", ms=5, mec="white", mew=0.4, label="body"),
        Line2D([], [], color=C["pre"], lw=1.1, marker="o", ms=2.8, label="hands"),
        Line2D([], [], color=C["fusion"], lw=0, marker="o", ms=3.0, label="face")]
fig.legend(handles=_key, loc="lower center", bbox_to_anchor=(0.5, _row_b_top + 0.004), ncol=3,
           frameon=False, fontsize=FS["label"], handlelength=2.2, columnspacing=1.6,
           borderaxespad=0.0, borderpad=0.1)

# (c) the supervision timeline
axt = fig.add_subplot(gs[2, :])
axt.set_xlim(0, T - 1); axt.set_ylim(0, 1); axt.grid(False)
axt.set_yticks([]); axt.spines["left"].set_visible(False)
axt.spines["top"].set_visible(False); axt.spines["right"].set_visible(False)
axt.spines["bottom"].set_color(C["grey"])
axt.add_patch(Rectangle((0, 0.42), T - 1, 0.30, fc=C["pale"], ec=C["grey"], lw=0.6, zorder=1))
axt.add_patch(Rectangle((G0, 0.42), G1 - G0, 0.30, fc=C["sup"], ec="none", zorder=2))
axt.text((G0 + G1) / 2, 0.57, GLOSS, ha="center", va="center", fontsize=FS["sub"],
         color="white", fontweight="bold", zorder=3)
# set inside the unlabelled span rather than to the right of the bar, which used to push the
# tight bounding box past the printed width
axt.text(G0 / 2, 0.57, "no sign-level label here", ha="center", va="center",
         fontsize=FS["sub"], color=C["ink"], zorder=3)
for f in show:
    axt.plot([f, f], [0.72, 0.80], color=C["grey"], lw=0.8, zorder=2)
axt.set_xlabel(f"frame (clip is {T} frames at 25 fps, {T/25:.1f} s)", fontsize=9)
axt.text(0, 0.16, "the only supervision:", fontsize=FS["label"], color=C["ink"], va="center")
axt.text(T * 0.26, 0.16, f"“{SUBTITLE}”", fontsize=FS["head"], color=C["ink"],
         va="center", style="italic")
axt.set_ylabel("(c)", fontsize=FS["head"], rotation=0, labelpad=14, va="center")

fig.savefig("figC1_what_the_model_sees.png", dpi=300, bbox_inches="tight")
fig.savefig("figC1_what_the_model_sees.pdf", bbox_inches="tight")
plt.close(fig)
print("figC1 written")


# =====================================================================
# FIG C2 — WHAT CTC DOES, AND WHAT THIS PROJECT'S CTC TARGET ACTUALLY IS
# Schematic. The literature-review marker circled "CTC" and wrote that key concepts are "not
# adequately explained or explained too late", so this panel explains the mechanism before the
# text relies on it. Panel (b) states the project's own approximation honestly: the auxiliary
# CTC head is trained against the ENGLISH content words in ENGLISH order, which is not the order
# the signs occur in, because Auslan-Daily ships no continuous gloss sequence to use instead.
# =====================================================================
fig, (axa, axb) = plt.subplots(2, 1, figsize=(TARGET_W, 3.70),
                               gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.30))
fig.subplots_adjust(left=0.01, right=0.99)      # the drawing fills the printed width

# ── (a) the alignment problem and the collapse rule ──────────────────────────
# Latin Modern Roman has no U+2205 glyph and drops it silently, so the blank symbol -- the one
# mark this panel exists to define -- rendered as nothing. Mathtext supplies it from CM.
BLANK = r"$\emptyset$"
emis = [BLANK, BLANK, "O", "O", BLANK, "I", "I", "I", BLANK, "L", "L", BLANK]
n = len(emis)
axa.set_xlim(-4.6, n + 0.6); axa.set_ylim(0.20, 3.12); axa.axis("off")
LX = -4.5          # row labels live left of the boxes, not on top of them

def row(y, label):
    axa.text(LX, y + 0.31, label, fontsize=FS["label"], color=C["ink"], ha="left", va="center")

row(2.35, "video frames")
for i in range(n):
    axa.add_patch(FancyBboxPatch((i, 2.35), 0.86, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc=C["pale"], ec=C["grey"], lw=0.7))
    axa.text(i + 0.43, 2.66, f"{i+1}", ha="center", va="center", fontsize=FS["sub"], color=C["ink"])

row(1.35, "per-frame output")
for i, e in enumerate(emis):
    blank = e == BLANK
    axa.add_patch(FancyBboxPatch((i, 1.35), 0.86, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc="white" if blank else C["sup"], ec=C["ink"] if blank else C["sup"], lw=0.7))
    axa.text(i + 0.43, 1.66, e, ha="center", va="center", fontsize=FS["head"],
             color=C["ink"] if blank else "white", fontweight="normal" if blank else "bold")
    axa.annotate("", xy=(i + 0.43, 2.01), xytext=(i + 0.43, 2.32),
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C["grey"]))

row(0.35, "target")
for i, ch in enumerate("OIL"):
    axa.add_patch(FancyBboxPatch((i, 0.35), 0.86, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc=C["sup"], ec="none"))
    axa.text(i + 0.43, 0.66, ch, ha="center", va="center", fontsize=FS["head"], color="white", fontweight="bold")
axa.annotate("", xy=(1.3, 1.03), xytext=(1.3, 1.31),
             arrowprops=dict(arrowstyle="-|>", lw=0.75, color=C["ink"]))
axa.text(1.75, 1.17, "collapse repeated labels, then drop " + BLANK, fontsize=FS["sub"], color=C["ink"], va="center")
# the explanation of the blank symbol and of summing over alignments now lives in the caption
panel_label(fig, axa, "a", ha="left")

# ── (b) the target this project can actually build ──────────────────────────
axb.set_xlim(0, 10); axb.set_ylim(0, 2.25); axb.axis("off")
panel_label(fig, axb, "b", ha="left")
axb.text(0.05, 1.82, "Auslan gloss sequence, as signed:", fontsize=FS["label"], color=C["ink"])
for i, (g, x) in enumerate(zip(["OIL", "ADD", "NOW"], [0.25, 1.65, 3.05])):
    axb.add_patch(FancyBboxPatch((x, 1.16), 1.22, 0.46, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc="white", ec=C["ink"], lw=0.75, ls=(0, (3, 2))))
    axb.text(x + 0.61, 1.39, g, ha="center", va="center", fontsize=FS["head"], color=C["ink"])
axb.text(4.45, 1.39, "not annotated in Auslan-Daily",
         fontsize=FS["label"], color=C["ink"], va="center")
axb.text(0.05, 0.86, "CTC target used: English content words, in English order",
         fontsize=FS["label"], color=C["ink"])
for g, x in zip(["now", "add", "olive", "oil"], [0.25, 1.65, 3.05, 4.45]):
    axb.add_patch(FancyBboxPatch((x, 0.20), 1.22, 0.46, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc=C["sup"], ec="none"))
    axb.text(x + 0.61, 0.43, g, ha="center", va="center", fontsize=FS["head"], color="white", fontweight="bold")
axb.text(5.95, 0.43, "proxy: English word order",
         fontsize=FS["label"], color=C["ink"], va="center", style="italic")

save(fig, "figC2_ctc_explained")
plt.close(fig)
print("figC2 written")


# =====================================================================
# FIG C3 — THE TRANSLATION MODEL
# Read off the code, not from memory:
#   slt/model.py   PoseSLT: input_proj Linear(feat->d)+LayerNorm+ReLU+Dropout, sinusoidal
#                  positional encoding, nn.Transformer(d_model, nhead, enc, dec, ffn),
#                  generator Linear(d->vocab)
#   config.py      d_model 256, nhead 8, num_encoder_layers 4, num_decoder_layers 4,
#                  dim_feedforward 1024, dropout 0.1
#   train_ctc.py   ctc_head = nn.Linear(cfg.d_model, len(gloss2idx)+1) on the ENCODER memory,
#                  nn.CTCLoss(blank=0), weighted by cfg.ctc_weight = 0.3
#   slt/rgb_fusion.py  optional RGB stream concatenated to the pose memory along time; the CTC
#                  head stays on the unmodified pose memory
# The method chapter previously had no figure at all; this is the system it describes.
# =====================================================================
def box(ax, x, y, w, h, label, fc, tc="white", fs=9, ec="none", ls="solid", lw=1.0, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.055 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.115, sub, ha="center", va="center",
                fontsize=fs - 1.8, color=tc, zorder=4)


def arrow(ax, x0, y0, x1, y1, ls="solid", col=None, lw=1.3, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col or C["ink"], linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"))


# Built at the printed width, with the box sub-labels moved into the caption -- which is where
# published architecture figures put their specifications anyway. At 6.3in a box cannot carry a
# name and a spec line and still be legible.
fig, ax = plt.subplots(figsize=(TARGET_W, 2.60))
fig.subplots_adjust(left=0.01, right=0.99)      # the drawing fills the printed width
ax.set_xlim(-1.80, 10.9); ax.set_ylim(0.30, 4.40); ax.axis("off"); ax.grid(False)

Y, H = 2.00, 0.72
# Rule 3.3: three fills, one meaning each, keyed below the drawing:
#   pale = not trained in this project, blue = trained in this project, white = data.
# A dashed outline marks what is absent at inference (training only) or present only in the
# RGB-fusion models. The anatomical colours are not borrowed for boxes.
DATA_FC = "white"
DASH = (0, (3.5, 2))
MAIN = [(0.05, 1.60, "DWPose", C["pale"], C["ink"]),
        (2.00, 2.25, "133 keypoints", DATA_FC, C["ink"]),
        (4.60, 1.75, "encoder \u00d74", C["accent"], "white"),
        (6.70, 1.75, "decoder \u00d74", C["accent"], "white"),
        (8.80, 1.55, "English", DATA_FC, C["ink"])]
for x, w, lab, col, tc in MAIN:
    box(ax, x, Y, w, H, lab, col, tc=tc, fs=FS["label"],
        ec=C["grey"] if col in (C["pale"], DATA_FC) else "none", lw=0.75)
for a, b in zip(MAIN, MAIN[1:]):
    arrow(ax, a[0] + a[1], Y + H / 2, b[0], Y + H / 2, lw=0.75)
arrow(ax, -0.45, Y + H / 2, 0.05, Y + H / 2, lw=0.75)
ax.text(-0.50, Y + H / 2, "video", ha="right", va="center", fontsize=FS["label"], color=C["ink"])
# tensor shapes on the arrows (rule 3.2)
ax.text(-0.50, Y + H / 2 - 0.30, "T\u00d7H\u00d7W\u00d73", ha="right", va="center",
        fontsize=FS["sub"], color=C["ink"])
ax.text(1.825, Y + H + 0.10, "T\u00d7133\u00d73", ha="center", va="bottom",
        fontsize=FS["sub"], color=C["ink"])
ax.text(6.525, Y + H + 0.10, "encoder memory\nT\u00d7256", ha="center", va="bottom",
        fontsize=FS["sub"], color=C["ink"], linespacing=1.2)
ax.text(8.625, Y - 0.10, "L tokens", ha="center", va="top", fontsize=FS["sub"], color=C["ink"])

# auxiliary CTC head, off the encoder memory, discarded at inference
box(ax, 4.55, 0.55, 1.85, H, "CTC head", C["accent"], fs=FS["label"], ec=C["ink"], ls=DASH, lw=0.75)
arrow(ax, 5.48, Y - 0.02, 5.48, 0.55 + H + 0.02, ls=DASH, lw=0.75)
ax.text(6.58, 0.91, "training only", fontsize=FS["sub"], color=C["ink"], va="center")

# VLP pretraining stage (Section 3.5.3): initialises the encoder, dashed because it is a
# separate earlier stage, not part of the model at inference
box(ax, 0.75, 0.42, 3.05, 0.95, "", "none", tc=C["ink"], ec=C["ink"], ls=DASH, lw=0.75)
ax.text(0.75 + 3.05 / 2, 0.42 + 0.64, "VLP pretraining", ha="center", va="center",
        fontsize=FS["label"], color=C["ink"], fontweight="bold", zorder=4)
ax.text(0.75 + 3.05 / 2, 0.42 + 0.30, "text encoder, contrastive", ha="center", va="center",
        fontsize=FS["sub"], color=C["ink"], zorder=4)
arrow(ax, 3.80, 1.10, 4.80, Y - 0.02, ls=DASH, col=C["ink"], lw=0.75)
ax.text(2.25, 1.64, "initialises the encoder", ha="center", va="center",
        fontsize=FS["sub"], color=C["ink"])

# optional appearance stream, above, dashed throughout
box(ax, 2.00, 3.35, 2.25, H, "DINOv2", C["pale"], tc=C["ink"], ls=DASH, ec=C["grey"], lw=0.75, fs=FS["label"])
box(ax, 4.60, 3.35, 1.75, H, "RGB enc \u00d72", C["accent"], tc="white", ls=DASH, ec=C["ink"], lw=0.75, fs=FS["label"])
# the appearance stream reads the video itself (crops located by the keypoints), so it branches off
# the video input, not off the DWPose box
ax.plot([-0.20, -0.20], [Y + H / 2, 3.71], ls=DASH, color=C["grey"], lw=0.75, solid_capstyle="butt")
arrow(ax, -0.20, 3.71, 2.00, 3.71, ls=DASH, col=C["grey"], lw=0.75)
arrow(ax, 4.25, 3.71, 4.60, 3.71, ls=DASH, col=C["grey"], lw=0.75)
arrow(ax, 6.35, 3.71, 7.57, 3.71, ls=DASH, col=C["grey"], lw=0.75)
arrow(ax, 7.57, 3.71, 7.57, Y + H + 0.02, ls=DASH, col=C["grey"], lw=0.75)
ax.text(8.00, 3.71, "appearance stream,\nRGB-fusion models only",
        fontsize=FS["sub"], color=C["ink"], style="italic", va="center")

from matplotlib.patches import Patch
_k3 = [Patch(fc=C["pale"], ec=C["grey"], lw=0.75, label="not trained here"),
       Patch(fc=C["accent"], ec="none", label="trained in this project"),
       Patch(fc=DATA_FC, ec=C["grey"], lw=0.75, label="data"),
       Patch(fc="none", ec=C["ink"], lw=0.75, ls=DASH, label="training or RGB-fusion only")]
fig.legend(handles=_k3, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=4, frameon=False,
           fontsize=FS["sub"], handlelength=1.6, columnspacing=1.4, borderaxespad=0.0)

save(fig, "figC3_architecture")
plt.close(fig)
print("figC3 written")


# =====================================================================
# FIG C4 — THE SAME SIGN, IN CITATION FORM AND IN CONVERSATION
# The project's central diagnosis is that conversational Auslan is the hard case: signs are
# shorter, faster and less repeatable than the dictionary forms a stitching corpus is built
# from. Nothing published shows this for Auslan, so this figure is built from our own data.
#   citation        MM-WLAuslan train clips labelled GOAL, DWPose, 30 fps
#                   (results/mmwl_sample/, pulled from the Kaya backup)
#   conversational  Auslan-Daily Communication spans labelled GOAL in Sign_Spotting.xlsx,
#                   DWPose, 25 fps (results/dwpose_experiment/dwpose_clips/)
# Skeletons are drawn after the same per-frame normalisation the model applies (centre on the
# shoulders, scale by shoulder width), so the two rows are directly comparable.
# =====================================================================
import glob
import pandas as pd

GLOSS = "goal"
MM_DIR = os.path.join(ROOT, "results", "mmwl_sample")
SPOT_XLSX = os.path.join(ROOT, "data", "raw", "Auslan-Daily", "Dataset Split Table",
                         "Sign Spotting", "Sign_Spotting.xlsx")


def norm_frames(kp):
    """Centre on the shoulder midpoint and scale by shoulder width, per frame -- the same
    normalisation slt/data.py applies before the encoder sees the clip."""
    out = kp.copy()
    ls, rs = kp[:, 5, :2], kp[:, 6, :2]
    mid = (ls + rs) / 2.0
    wid = np.linalg.norm(ls - rs, axis=1)
    wid = np.where(wid < 1e-3, np.nan, wid)
    out[:, :, :2] = (kp[:, :, :2] - mid[:, None, :]) / wid[:, None, None]
    return out


spot = pd.read_excel(SPOT_XLSX)
spot["clip"] = spot["Video_Name_Clip"].str.replace("_signer.mp4", "", regex=False)
local = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(POSE_DIR, "*.npz"))}
conv = spot[(spot["Query_Gloss"].str.lower() == GLOSS) & (spot["clip"].isin(local))]
conv = conv.assign(n=lambda d: d["End_Frame"] - d["Begin_Frame"])
cit_files = sorted(glob.glob(os.path.join(MM_DIR, "*.npz")))
cit_len = np.array([len(np.load(f)["KPS"]) for f in cit_files]) / 30.0        # MM-WLAuslan is 30 fps
conv_len = conv["n"].to_numpy() / 25.0                                        # Auslan-Daily is 25 fps

# The exemplars are the BEST-TRACKED instance in each group, not the median-duration one: a
# thumbnail of a poorly estimated clip shows the estimator's failure rather than the sign. The
# duration claim in panel (c) uses every instance, so the choice here affects only the pictures.
def arm_hand_conf(k):
    return float(k[:, [5, 6, 7, 8, 9, 10], 2].mean() + k[:, 91:133, 2].mean()) / 2.0


cit_pick = max(cit_files, key=lambda f: arm_hand_conf(np.load(f)["KPS"]))
_conv_scores = []
for _, _r in conv.iterrows():
    _k = np.load(os.path.join(POSE_DIR, f"{_r['clip']}.npz"))["KPS"]
    _seg = _k[int(_r["Begin_Frame"]):int(_r["End_Frame"]) + 1]
    _conv_scores.append(arm_hand_conf(_seg))
conv_row = conv.iloc[int(np.argmax(_conv_scores))]
cit_kp = norm_frames(np.load(cit_pick)["KPS"])
conv_full = np.load(os.path.join(POSE_DIR, f"{conv_row['clip']}.npz"))["KPS"]
conv_kp = norm_frames(conv_full[int(conv_row["Begin_Frame"]):int(conv_row["End_Frame"]) + 1])

NP = 5
# panel (c) gets the full width as its own row, so its axis can start at 0 and its labels sit
# inside the plotting area
fig = plt.figure(figsize=(TARGET_W, 4.70))
gs = fig.add_gridspec(3, NP, height_ratios=[1.0, 1.0, 0.80], hspace=0.45, wspace=0.06,
                      right=0.98)


# Only the articulators that carry the sign: shoulders, arms and hands, plus a head marker.
# The face contour and the legs add nothing at this size and make the thumbnails unreadable.
ARM_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]


def view_box(kp, idx):
    """A common square window per row, fitted to the arms, hands and head across the frames
    shown. A fixed window clips low arm positions, which is what conversational signing has."""
    pts = []
    for f in idx:
        k = kp[f]
        for sl in (slice(0, 11), LHAND, RHAND):
            seg = k[sl]
            pts.append(seg[seg[:, 2] > CONF, :2])
    pts = np.concatenate([p for p in pts if len(p)])
    x0, x1 = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
    y0, y1 = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.18
    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def draw_row(r, kp, title, fps, col, view):
    idx = np.linspace(0, len(kp) - 1, NP).round().astype(int)
    vx0, vx1, vy0, vy1 = view
    for j, f in enumerate(idx):
        ax = fig.add_subplot(gs[r, j])
        k = kp[f]
        for a, b in ARM_EDGES:
            if k[a, 2] > CONF and k[b, 2] > CONF:
                ax.plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], color=col, lw=2.4, solid_capstyle="round")
        if k[0, 2] > CONF:                                   # nose -> a simple head marker
            ax.scatter([k[0, 0]], [k[0, 1]], s=190, color=col, alpha=0.28, zorder=2)
            ax.scatter([k[0, 0]], [k[0, 1]], s=24, color=col, zorder=3)
        for sl in (LHAND, RHAND):
            pts = k[sl]
            for a, b in HAND_EDGES:
                if pts[a, 2] > CONF and pts[b, 2] > CONF:
                    ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                            color=C["pre"], lw=1.25, solid_capstyle="round")
            m = pts[:, 2] > CONF
            ax.scatter(pts[m, 0], pts[m, 1], s=3.5, color=C["pre"], zorder=4)
        ax.set_xlim(vx0, vx1); ax.set_ylim(vy1, vy0); ax.set_aspect("equal")
        tidy(ax)
        ax.set_xlabel(f"{f/fps:.2f} s", fontsize=FS["sub"], color=C["grey"], labelpad=1)
        if j == 0:
            ax.set_ylabel(title, fontsize=FS["head"], labelpad=6)


# ONE window for both rows, so the excursion of the arms is comparable between them: the
# citation form travels from rest into the sign and back, the conversational one does not.
_idx = lambda kp: np.linspace(0, len(kp) - 1, NP).round().astype(int)
_a, _b = view_box(cit_kp, _idx(cit_kp)), view_box(conv_kp, _idx(conv_kp))
VIEW = (min(_a[0], _b[0]), max(_a[1], _b[1]), min(_a[2], _b[2]), max(_a[3], _b[3]))
_s = max(VIEW[1] - VIEW[0], VIEW[3] - VIEW[2]) / 2
_cx, _cy = (VIEW[0] + VIEW[1]) / 2, (VIEW[2] + VIEW[3]) / 2
VIEW = (_cx - _s, _cx + _s, _cy - _s, _cy + _s)

# arms in the body colour in both rows (anatomical meaning only); the rows are named, not hued
draw_row(0, cit_kp, "(a) citation form\nMM-WLAuslan", 30.0, C["pose"], VIEW)
draw_row(1, conv_kp, "(b) in conversation\nAuslan-Daily", 25.0, C["pose"], VIEW)

axd = fig.add_subplot(gs[2, :])
rng = np.random.default_rng(0)
_xmax = 1.08 * max(cit_len.max(), conv_len.max())
# marker shape separates the groups as well as fill: open-grey circles vs dark triangles
for vals, y, mk, fc, lab, side in [(cit_len, 1.0, "o", C["grey"], "citation", "left"),
                                   (conv_len, 0.0, "^", C["ink"], "conversation", "right")]:
    # median bar drawn BEHIND the instances
    axd.plot([np.median(vals)] * 2, [y - 0.30, y + 0.30], color=C["ink"], lw=1.6, zorder=2)
    axd.scatter(vals, y + rng.uniform(-0.09, 0.09, len(vals)), s=30, marker=mk, color=fc,
                alpha=0.9, edgecolors=C["ink"], linewidths=0.6, zorder=3)
    # a clear gap to the nearest point, and a white box so the x-grid does not run through it
    _lx = vals.min() - 0.07 * _xmax if side == "left" else vals.max() + 0.07 * _xmax
    axd.text(_lx, y, f"{lab}, n={len(vals)}, median {np.median(vals):.2f}\u2009s",
             ha="right" if side == "left" else "left", va="center",
             fontsize=FS["sub"], color=C["ink"], zorder=4,
             bbox=dict(fc="white", ec="none", pad=1.5))
axd.set_yticks([0, 1])
axd.set_yticklabels(["in conversation", "citation form"], fontsize=FS["sub"])
axd.set_xlim(0, _xmax)
axd.set_ylim(-0.55, 1.55); axd.set_xlabel("duration of the sign GOAL (s)", fontsize=FS["label"])
axd.set_title("(c) duration of every instance held", fontsize=FS["head"], loc="left", pad=4)
axd.grid(False); axd.grid(axis="x", alpha=0.25); axd.set_axisbelow(True)

fig.savefig("figC4_citation_vs_conversational.png", dpi=300, bbox_inches="tight")
fig.savefig("figC4_citation_vs_conversational.pdf", bbox_inches="tight")
plt.close(fig)
print(f"figC4 written  (citation median {np.median(cit_len):.2f}s n={len(cit_len)}, "
      f"conversational median {np.median(conv_len):.2f}s n={len(conv_len)})")


# =====================================================================
# FIG C5 — BLEU-4 AND WER, WORKED ON ONE REAL TEST SENTENCE
# The literature-review marker wrote "Explain BLEU" (p7) and "Define metrics!" (p9) after
# scores had been quoted since p3. This computes both metrics on a real pair, so the reader
# sees where a single-digit corpus score comes from.
#   pair      results/semantic_adequacy_grading/hyps_vlp.jsonl, idx 104 (VLP-80, beam-5, official test)
#   harness   scored through src/slt/metrics.py, which reproduces the recorded
#             corpus BLEU-4 of 10.27 for this decode set to four decimals before anything here
#             was drawn.
# =====================================================================
REF_W = S("9759272241498eb0366c3653d820a7979f7a1ce4").split()   # test reference, read by digest
HYP_W = ["that", "is", "right", "possum", "."]
PREC = [(1, 4, 5, ["that", "is", "right", "."]),
        (2, 2, 4, ["that is", "is right"]),
        (3, 1, 3, ["that is right"]),
        (4, 0, 2, [])]
BP = 0.6703
_OPS = [("C", "that"), ("C", "is"), ("C", "right"), ("D", ""), ("D", ""), ("S", "possum"), ("C", ".")]
ALIGN = [(op, r, h) for (op, h), r in zip(_OPS, REF_W)]

fig, (axb, axw) = plt.subplots(2, 1, figsize=(TARGET_W, 3.60),
                               gridspec_kw=dict(height_ratios=[1.55, 0.90], hspace=0.28))
fig.subplots_adjust(left=0.01, right=0.99)      # the drawing fills the printed width

# ── (a) BLEU-4 ──────────────────────────────────────────────────────────────
axb.set_xlim(0, 12.6); axb.set_ylim(0.45, 5.60); axb.axis("off"); axb.grid(False)
axb.set_title("(a) BLEU-4", fontsize=FS["head"], loc="left", pad=4)
axb.text(0.0, 5.12, "reference", fontsize=FS["sub"], color=C["ink"])
axb.text(1.55, 5.12, " ".join(REF_W), fontsize=FS["head"], color=C["ink"], style="italic")
axb.text(0.0, 4.62, "translation", fontsize=FS["sub"], color=C["ink"])
axb.text(1.55, 4.62, " ".join(HYP_W), fontsize=FS["head"], color=C["ink"], style="italic")

for row, (n, m, tot, matched) in enumerate(PREC):
    y = 3.85 - row * 0.72
    axb.text(0.0, y, f"{n}-grams", fontsize=FS["sub"], color=C["ink"], va="center")
    hyp_ngrams = [" ".join(HYP_W[i:i + n]) for i in range(len(HYP_W) - n + 1)]
    x = 1.55
    for g in hyp_ngrams:
        hit = g in matched
        w = 0.135 * len(g) + 0.30
        axb.add_patch(FancyBboxPatch((x, y - 0.20), w, 0.40,
                                     boxstyle="round,pad=0.02,rounding_size=0.06",
                                     fc=C["good"] if hit else "white",
                                     ec=C["ink"] if hit else C["grey"], lw=1.3 if hit else 0.7,
                                     linestyle="solid" if hit else (0, (2, 1.5))))
        # a match is marked by fill, a heavy solid outline AND bold text; a miss is dashed
        axb.text(x + w / 2, y, g, ha="center", va="center", fontsize=FS["sub"],
                 color="white" if hit else C["ink"], fontweight="bold" if hit else "normal")
        x += w + 0.14
    axb.text(9.55, y, f"$p_{n} = {m}/{tot}$", fontsize=9.6,
             color=C["ink"] if m else C["base"], va="center")
axb.text(10.85, 3.85 - 3 * 0.72, "← none match", fontsize=FS["label"],
         color=C["base"], va="center")

# only the worked numbers stay in the figure; the explanation moves to the caption
axb.text(0.0, 0.85,
         f"brevity penalty $\\exp(1-7/5) = {BP:.2f}$;   "
         f"BLEU-4 $= {BP:.2f} \\times (p_1 p_2 p_3 p_4)^{{1/4}} = 0$",
         fontsize=FS["label"], color=C["ink"], va="center")

# ── (b) WER ─────────────────────────────────────────────────────────────────
axw.set_xlim(0, 12.6); axw.set_ylim(0.25, 2.5); axw.axis("off"); axw.grid(False)
axw.set_title("(b) Word error rate", fontsize=FS["head"], loc="left", pad=4)
TAG = dict(C=(C["pale"], C["grey"]), S=(C["base"], "white"),
           D=(C["synth"], "white"), I=(C["ens"], C["ink"]))
x = 1.55
axw.text(0.0, 1.78, "reference", fontsize=FS["sub"], color=C["ink"])
axw.text(0.0, 1.16, "translation", fontsize=FS["sub"], color=C["ink"])
for op, rw, hw in ALIGN:
    w = 0.135 * max(len(rw), len(hw), 3) + 0.34
    for y, word in ((1.78, rw), (1.16, hw)):
        axw.add_patch(FancyBboxPatch((x, y - 0.20), w, 0.40,
                                     boxstyle="round,pad=0.02,rounding_size=0.06",
                                     fc=C["pale"] if word else "white",
                                     ec=C["grey"], lw=0.6, linestyle="solid" if word else (0, (2, 2))))
        axw.text(x + w / 2, y, word if word else "—", ha="center", va="center",
                 fontsize=FS["sub"], color=C["ink"] if word else C["grey"])
    if op != "C":
        fc, tc = TAG[op]
        axw.add_patch(FancyBboxPatch((x + w / 2 - 0.16, 2.14), 0.32, 0.30,
                                     boxstyle="round,pad=0.01,rounding_size=0.05", fc=fc, ec="none"))
        axw.text(x + w / 2, 2.29, op, ha="center", va="center", fontsize=FS["sub"], color=tc, fontweight="bold")
    x += w + 0.12
axw.text(0.0, 0.48,
         "One substitution, two deletions, no insertions, against a 7-word reference: "
         "$\\mathrm{WER} = (1+2+0)/7 = 0.43$.",
         fontsize=FS["sub"], color=C["ink"], va="center")

fig.savefig("figC5_metrics_worked.png", dpi=300, bbox_inches="tight")
fig.savefig("figC5_metrics_worked.pdf", bbox_inches="tight")
plt.close(fig)
print("figC5 written")


# =====================================================================
# FIG C6 — QUALITATIVE RESULTS, WITH THE MODEL'S OWN CONFIDENCE
# Vision and NLP venues expect to see real model output, successes and failures side by side.
# Five clips from the official Communication test split, decoded by VLP-80 at beam 5, ordered by
# the model's self-confidence (mean per-token cross-entropy of its own output; lower = more
# confident), which is the signal behind the selective-prediction result.
#   decodes     results/semantic_adequacy_grading/hyps_vlp.jsonl  (idx aligns row-for-row with the manifest)
#   manifest    results/TEST_manifest_official.tsv       (800 clips; verified 800/800 have local video)
#   confidence  results/self_reliance_vlp.json -> self_ce (800 values, same ordering)
#   video       results/dwpose_experiment/rgb_clips/Signer/<clip>_signer.mp4
# Clips are drawn only from those with stable framing (largest single-frame jump in shoulder width
# or torso centre below 25% of shoulder width), so the frames show signing rather than the pose
# extractor changing its mind about who the signer is.
# The five are chosen to span the outcome space, NOT to flatter: one exact match, one long sentence
# whose content words fail, one semantic near-miss the metric cannot credit, one confidently wrong,
# one unconfidently wrong. Selection recorded in results/figures/qual_examples.json.
# =====================================================================
import sys as _sys
import json as _json
_sys.path.insert(0, os.path.join(ROOT, "src", "tools"))
from stitch_lexicon import STOP as _STOP

QUAL = sorted(_json.load(open(os.path.join(os.path.dirname(__file__), "qual_examples.json"))),
              key=lambda q: q["ce"])
NF = 3                                                   # frames shown per clip
TEST_POSE = os.path.join(ROOT, "results", "test_clips_dwpose")


def content_words(t):
    return [w for w in t.split() if w not in _STOP and w.isalpha()]


fig = plt.figure(figsize=(TARGET_W, 4.35))
fig.canvas.draw()          # a renderer must exist before text can be measured
gs = fig.add_gridspec(len(QUAL), NF + 1, width_ratios=[1] * NF + [5.6],
                      hspace=0.13, wspace=0.035)

for r, q in enumerate(QUAL):
    kp = np.load(os.path.join(TEST_POSE, q["clip"] + ".npz"))["KPS"]
    idx = (np.linspace(0.12, 0.88, NF) * (len(kp) - 1)).round().astype(int)
    # one crop per clip, fitted to where the head and hands actually go
    m = kp[..., 2] > CONF
    pts = np.concatenate([kp[:, :11, :2][m[:, :11]], kp[:, 91:133, :2][m[:, 91:133]]])
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    side = max(x1 - x0, y1 - y0) * 1.15
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cap = cv2.VideoCapture(os.path.join(RGB_DIR, q["clip"] + "_signer.mp4"))
    for j, f in enumerate(idx):
        ax = fig.add_subplot(gs[r, j])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            im = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            H, W = im.shape[:2]
            ax.imshow(im[int(max(0, cy - side / 2)):int(min(H, cy + side / 2)),
                         int(max(0, cx - side / 2)):int(min(W, cx + side / 2))])
        tidy(ax)
        if r == 0:
            ax.set_title(["start", "middle", "end"][j], fontsize=FS["sub"], color=C["ink"], pad=3)
    cap.release()

    axt = fig.add_subplot(gs[r, NF])
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off"); axt.grid(False)
    ref_c, hyp_c = content_words(q["ref"]), content_words(q["hyp"])

    # sentence column starts after the widest row label as it actually renders, plus a gap
    _probe = axt.text(0.0, 0.5, "signed", fontsize=FS["tiny"])
    TX0 = max(0.10, _probe.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(axt.transData.inverted()).x1 + 0.025)
    _probe.remove()

    def render(y, label, text, other, col_miss):
        """Lay the sentence out word by word and advance by the width the word ACTUALLY renders
        at. All text is dark. Function words are regular weight; content words are bold;
        recovered content words are bold AND underlined; content words produced but not signed
        are bold italic in the error colour. If a sentence would run past the column it is
        re-set at 7.6 pt (never smaller) so it ends inside the canvas."""
        axt.text(0.0, y, label, fontsize=FS["tiny"], color=C["ink"], va="center")
        rend = fig.canvas.get_renderer()
        inv = axt.transData.inverted()
        for size in (FS["sub"], 7.6):
            arts = []
            x = TX0
            for w in text.split():
                func = (w in _STOP) or (not w.isalpha())
                hit = (not func) and (w in other)
                colour = col_miss if (not func and not hit) else C["ink"]
                t = axt.text(x, y, w, fontsize=size, color=colour, va="center",
                             fontweight="normal" if func else "bold",
                             style="italic" if (not func and not hit and col_miss == C["base"]) else "normal")
                arts.append(t)
                bb = t.get_window_extent(renderer=rend).transformed(inv)
                if hit:
                    arts += axt.plot([bb.x0, bb.x1], [bb.y0 - 0.005] * 2, color=C["ink"],
                                     lw=0.7, clip_on=False)
                x = bb.x1 + (0.005 if w in ".,?!" else 0.011)
            if x <= 0.975 or size == 7.6:
                break
            for a in arts:
                a.remove()

    render(0.78, "signed", q["ref"], hyp_c, C["ink"])
    render(0.46, "model", q["hyp"], ref_c, C["base"])
    hit = sum(1 for w in ref_c if w in hyp_c)
    axt.text(TX0, 0.14,
             f"{hit}/{len(ref_c)} content words recovered     "
             f"self-confidence {q['ce']:.2f} nats     {len(kp)} frames",
             fontsize=FS["tiny"], color=C["ink"], va="center")
    if r < len(QUAL) - 1:
        axt.plot([0, 1], [-0.07, -0.07], color=C["pale"], lw=1.0, clip_on=False)

# The colour/weight code is defined in the caption (it has to be, for the figure to be
# self-contained), so an in-figure legend would only repeat it -- and being wider than the grid
# it inflated the tight bounding box, pushing the real content below the printed width.
fig.savefig("figC6_qualitative.png", dpi=300, bbox_inches="tight")
fig.savefig("figC6_qualitative.pdf", bbox_inches="tight")
plt.close(fig)
print("figC6 written")


# =====================================================================
# FIG C7 — HOW A SYNTHETIC TRAINING SENTENCE IS BUILT
# The second-largest replicated gain in this thesis (+1.48 BLEU-4 over three paired seeds) comes
# from training on sentences stitched out of isolated dictionary clips, and the method chapter
# describes it in prose alone. This shows one real example end to end.
#   synthetic clip   results/stitch_sample/stitch2_000007.npz   (from the Kaya build, 97 frames)
#   provenance       stitch_audit/corpus/audit.jsonl -- per-sign gloss, source clip id, source
#                    length before and after trimming, and the span it occupies in the output
#   source clips     results/stitch_sample/<id>.npz  (MM-WLAuslan train, DWPose, 30 fps)
#   target text      "hey there friends how old are you ."
# The numbers on the arrows are the real compression applied to each sign.
# =====================================================================
SEGS = [  # gloss, source id, source frames (original), output frames, start, end
    ("hey",     "22938", 74, 11,  0, 10),
    ("friend",  "28431", 68, 10, 13, 22),
    ("how old", "83334", 69, 19, 26, 44),
    ("are",     "29306", 59, 23, 45, 67),
    ("you",     "82513", 80, 26, 71, 96),
]
STITCH_DIR = os.path.join(ROOT, "results", "stitch_sample")
TARGET = "hey there friends how old are you ."
SYN = np.load(os.path.join(STITCH_DIR, "stitch2_000007.npz"))["KPS"]
SEG_COLS = [C["pose"], C["pre"], C["fusion"], C["synth"], C["ens"]]


def norm_upper(kp):
    """Centre on the shoulder midpoint, scale by shoulder width -- the encoder's own view."""
    out = kp.copy()
    ls, rs = kp[:, 5, :2], kp[:, 6, :2]
    mid = (ls + rs) / 2.0
    wid = np.linalg.norm(ls - rs, axis=1)
    wid = np.where(wid < 1e-3, np.nan, wid)
    out[:, :, :2] = (kp[:, :, :2] - mid[:, None, :]) / wid[:, None, None]
    return out


def mini_skeleton(ax, k, col):
    for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]:
        if k[a, 2] > CONF and k[b, 2] > CONF:
            ax.plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], color=col, lw=2.0, solid_capstyle="round")
    if k[0, 2] > CONF:
        ax.scatter([k[0, 0]], [k[0, 1]], s=130, color=col, alpha=0.28)
        ax.scatter([k[0, 0]], [k[0, 1]], s=18, color=col)
    for sl in (LHAND, RHAND):
        pts = k[sl]
        for a, b in HAND_EDGES:
            if pts[a, 2] > CONF and pts[b, 2] > CONF:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=C["pre"], lw=1.0)
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(1.2, -2.0); ax.set_aspect("equal")
    tidy(ax)


fig = plt.figure(figsize=(TARGET_W, 3.45))
gs = fig.add_gridspec(3, len(SEGS), height_ratios=[1.0, 0.34, 1.28], hspace=0.06, wspace=0.06)

# (a) the dictionary clips this sentence is assembled from
for j, (gloss, sid, src_n, out_n, s0, s1) in enumerate(SEGS):
    ax = fig.add_subplot(gs[0, j])
    k = norm_upper(np.load(os.path.join(STITCH_DIR, f"{sid}.npz"))["KPS"])
    mini_skeleton(ax, k[len(k) // 2], SEG_COLS[j])
    ax.set_title(gloss.upper(), fontsize=FS["head"], color=SEG_COLS[j], fontweight="bold", pad=3)
    ax.set_xlabel(f"{src_n} frames\n({src_n/30:.1f} s at 30 fps)", fontsize=FS["tiny"],
                  color=C["grey"], labelpad=2)
    if j == 0:
        ax.set_ylabel("(a) isolated\ndictionary clips\nMM-WLAuslan", fontsize=FS["label"], labelpad=6)

# (b) the compression applied to each sign
axm = fig.add_subplot(gs[1, :])
axm.set_xlim(0, len(SEGS)); axm.set_ylim(0, 1); axm.axis("off"); axm.grid(False)
for j, (gloss, sid, src_n, out_n, s0, s1) in enumerate(SEGS):
    # each arrow ends on the middle of its own span in (b); both rows share the same width
    _xm = ((s0 + s1) / 2 + 0.5) / len(SYN) * len(SEGS)
    axm.annotate("", xy=(_xm, 0.02), xytext=(j + 0.5, 0.52),
                 arrowprops=dict(arrowstyle="-|>", lw=1.2, color=SEG_COLS[j], shrinkA=0, shrinkB=0))
    axm.text(j + 0.56, 0.30, f"{src_n}→{out_n}", fontsize=FS["sub"], color=C["base"], va="center")
# the explanation of what the arrows mean lives in the caption, not on top of the
# clip-length labels it used to collide with

# (c) the stitched clip: spans along the top, skeletons sampled beneath
axb = fig.add_subplot(gs[2, :])
T = len(SYN)
axb.set_xlim(-0.5, T - 0.5); axb.set_ylim(0, 1); axb.grid(False)
axb.set_yticks([]); axb.set_xlabel(f"frame of the synthetic clip  ({T} frames, 25 fps, "
                                  f"{T/25:.1f} s)", fontsize=9)
for sp in ("left", "top", "right"):
    axb.spines[sp].set_visible(False)
axb.spines["bottom"].set_color(C["grey"])
axb.add_patch(Rectangle((-0.5, 0.60), T, 0.26, fc=C["pale"], ec=C["grey"], lw=0.6, zorder=1))
from matplotlib.colors import to_rgb as _to_rgb


def _text_on(col):
    """Dark text on light fills, white only on dark ones."""
    r, g, b = _to_rgb(col)
    return C["ink"] if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


for j, (gloss, sid, src_n, out_n, s0, s1) in enumerate(SEGS):
    axb.add_patch(Rectangle((s0, 0.60), s1 - s0 + 1, 0.26, fc=SEG_COLS[j], ec="none",
                            alpha=0.85, zorder=2))
    if (s1 - s0 + 1) < 2.0 * len(gloss):
        # too narrow to hold its label with a margin: set it just above the bar instead
        axb.text((s0 + s1) / 2, 0.885, gloss.upper(), ha="center", va="bottom",
                 fontsize=FS["sub"], color=C["ink"], fontweight="bold", zorder=3)
    else:
        axb.text((s0 + s1) / 2, 0.73, gloss.upper(), ha="center", va="center", fontsize=FS["sub"],
                 color=_text_on(SEG_COLS[j]), fontweight="bold", zorder=3)
synn = norm_upper(SYN)
# thumbnails narrower than before, and nudged right only as far as needed so that adjacent
# insets never overlap; each still shows the middle frame of its own sign
_IW, _prev = 13.0, -1e9
for j, (gloss, sid, src_n, out_n, s0, s1) in enumerate(SEGS):
    f = (s0 + s1) // 2
    _c = max(f, -0.5 + _IW / 2, _prev + _IW + 1.0)
    _prev = _c
    sub = axb.inset_axes([(_c + 0.5 - _IW / 2) / T, 0.02, _IW / T, 0.50], transform=axb.transAxes)
    mini_skeleton(sub, synn[f], SEG_COLS[j])
axb.set_ylabel("(b) one stitched\ntraining clip", fontsize=FS["label"], labelpad=18)

fig.savefig("figC7_stitching.png", dpi=300, bbox_inches="tight")
fig.savefig("figC7_stitching.pdf", bbox_inches="tight")
plt.close(fig)
print("figC7 written")
