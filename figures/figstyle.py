"""One house style for every figure in the thesis. Imported by make_concept_figures.py and
make_thesis_figures.py so the two sets cannot drift apart.

SIZING, measured rather than assumed. Across 75 published papers from CVPR/ICCV/ECCV/NeurIPS/
ICLR/ACL/EMNLP (results/figure_standard/, font sizes read out of the PDFs with PyMuPDF): body
text is 10.0pt, the median text inside a figure is 7.2pt (0.70 of body), the 5th percentile is
5.2pt, and 52 of 75 papers go below 6pt somewhere. So the rule is not "never below 6pt" -- it is
that the TYPICAL label sits near 0.70-0.90 of body size.
A figure is therefore built AT ITS PRINTED WIDTH, so a fontsize of 8.5 means 8.5pt on paper.
Building wide and letting LaTeX shrink the result turned 8.4pt labels into 4.6pt.
  thesis  \\linewidth on A4 with 25mm margins = 160mm = 6.30in; body 11pt

COLOUR. One colour means one concept, in every figure of the thesis. The anatomical three are
frozen: a reader is taught them in Figure 1.1 and they must not be reused for anything else.
Every colour distinction is backed by mark type, weight or printed text, so nothing is lost in
greyscale or to a colour-vision deficiency.
"""
import os

TARGET_W = 6.30
BODY_PT = 11.0
FS = dict(head=9.0, label=8.5, sub=7.8, tiny=7.2)     # final points, on the page

# ── frozen anatomical palette: taught once, never reused ────────────────────────────────
ART = dict(body="#E8710A", hands="#1A237E", face="#7FBFBF")

# ── everything else ─────────────────────────────────────────────────────────────────────
C = dict(
    grey="#9AA0A6", ink="#202124", pale="#F1F3F4",
    body=ART["body"], hands=ART["hands"], face=ART["face"],
    sup="#6A1B9A",        # supervision artefacts: gloss spans, CTC targets, reference hits
    accent="#1565C0",     # the contribution, in any diagram; inherited parts stay grey
    warn="#B0332C",       # errors, wrong output, harmful effects
    good="#0F766E",       # agreement between a system and its reference
)

# results-ladder ramp: sequential, luminance-ordered, so bars stay separable in greyscale
LADDER = ["#BFC4C9", "#F5B77A", "#E8710A", "#8FB8E8", "#1565C0", "#7E57C2", "#F4B400"]


def apply(plt):
    """Set the document's font and the house defaults on a matplotlib pyplot module."""
    lm = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm"
    if os.path.isdir(lm):
        import matplotlib.font_manager as fm
        for f in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf",
                  "lmroman10-bolditalic.otf"):
            p = os.path.join(lm, f)
            if os.path.exists(p):
                fm.fontManager.addfont(p)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": FS["label"],
        "axes.titlesize": FS["head"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["sub"], "ytick.labelsize": FS["sub"],
        "legend.fontsize": FS["sub"],
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,        # embed TrueType, never Type 3
    })
