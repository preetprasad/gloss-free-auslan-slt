"""Extract figures from the reference papers, for inclusion in the thesis WITH ATTRIBUTION.

The literature-review marker's presentation criticism was "missed opportunity to include
graphics". UWA permits reproducing a figure from a cited work for academic purposes provided it
is attributed in the caption, so this tool pulls each candidate figure out of its source PDF at
print resolution and records exactly where it came from, so the caption can cite it precisely.

For each PDF it finds every "Figure N"/"Fig. N" caption, infers the figure's region on the page
(the drawings and images sitting above the caption and horizontally overlapping it, bounded
above by the nearest text block), renders that region at 400 dpi, and writes a manifest row with
the paper, page, figure number and full caption text.

Run:  python extract_paper_figures.py                 # all papers
      python extract_paper_figures.py Auslan-Daily.pdf CorrNet.pdf
"""
import json
import os
import re
import sys

import fitz  # PyMuPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "borrowed")
DPI = 400
CAP_RE = re.compile(r"^\s*(Figure|Fig\.?)\s*([0-9]+)\s*[.:]", re.I)
MIN_AREA_FRAC = 0.010          # ignore slivers
MAX_AREA_FRAC = 0.92           # a "figure" filling the whole page is a mis-parse


def caption_blocks(page):
    """Text blocks that begin like a figure caption, as (fig_no, caption_text, bbox)."""
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        m = CAP_RE.match(text.strip())
        if m:
            out.append((int(m.group(2)), " ".join(text.split()), fitz.Rect(x0, y0, x1, y1)))
    return out


def graphic_rects(page):
    """Every drawing and raster image on the page, as rectangles."""
    rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                rects.append(fitz.Rect(r))
        except Exception:
            pass
    return [r for r in rects if r.width > 2 and r.height > 2]


def figure_region(page, cap_rect):
    """The region the figure occupies: graphics above the caption that overlap it horizontally,
    clipped below by the caption and above by the nearest unrelated text block."""
    page_r = page.rect
    cand = []
    for r in graphic_rects(page):
        if r.y1 > cap_rect.y0 + 2:                       # must sit above the caption
            continue
        overlap = min(r.x1, cap_rect.x1) - max(r.x0, cap_rect.x0)
        if overlap > 0.18 * min(r.width, cap_rect.width):
            cand.append(r)
    if not cand:
        return None
    reg = cand[0]
    for r in cand[1:]:
        reg = reg | r
    # Do not run up into prose. A figure's own labels are short; a title block or a paragraph is
    # not, so any block of more than 12 words that overlaps the region at all is a hard ceiling.
    # (The earlier 0.25-width overlap test let paper titles through on single-column first pages.)
    top = page_r.y0
    for b in page.get_text("blocks"):
        bx0, by0, bx1, by1, txt = b[0], b[1], b[2], b[3], b[4]
        t = txt.strip()
        if not t or CAP_RE.match(t):
            continue
        prose = len(t.split()) > 12
        overlap = min(bx1, reg.x1) - max(bx0, reg.x0)
        if by1 <= cap_rect.y0 and overlap > (0.10 if prose else 0.25) * reg.width:
            if by1 <= reg.y1:
                top = max(top, by1)
    reg.y0 = max(reg.y0, top + 1)
    reg.y1 = min(cap_rect.y0 - 1, reg.y1 + 2)
    reg.x0 = max(page_r.x0, reg.x0 - 3)
    reg.x1 = min(page_r.x1, reg.x1 + 3)
    if reg.height < 24 or reg.width < 40:
        return None
    frac = (reg.width * reg.height) / (page_r.width * page_r.height)
    if not (MIN_AREA_FRAC < frac < MAX_AREA_FRAC):
        return None
    return reg


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(name)[0].lower()).strip("_")[:44]


def run(pdfs):
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name in pdfs:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            print(f"  !! missing {name}")
            continue
        doc = fitz.open(path)
        s, n = slug(name), 0
        for pno in range(len(doc)):
            page = doc[pno]
            for fig_no, cap, cap_rect in caption_blocks(page):
                reg = figure_region(page, cap_rect)
                if reg is None:
                    continue
                out_png = os.path.join(OUT, f"{s}__fig{fig_no}_p{pno+1}.png")
                page.get_pixmap(clip=reg, dpi=DPI).save(out_png)
                manifest.append(dict(
                    source_pdf=name, slug=s, figure=fig_no, page=pno + 1,
                    caption=cap[:600], png=os.path.relpath(out_png, os.path.dirname(__file__)),
                    px=[round(v, 1) for v in (reg.x0, reg.y0, reg.x1, reg.y1)],
                ))
                n += 1
        doc.close()
        print(f"  {name}: {n} figures")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest)} figures -> {OUT}")
    return manifest


DEFAULT = [
    "Auslan-Daily.pdf",
    "MM-WLAuslan_Multi-View_Multi-Modal_Word-Level_Australian_Sign_Language_Recognition_Dataset.pdf",
    "TwoStream-SLR.pdf", "CorrNet.pdf", "VAC.pdf", "CrossLingual-SLR.pdf",
    "Online-CSLR-Translation.pdf", "GFSLT-VLP.pdf", "SignCL.pdf", "SignLLM.pdf",
    "LLMs_are_Good_Sign_Language_Translators.pdf",
    "Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation.pdf",
    "SignGen_End-to-End_Sign_Language_Video_Generation_with_Latent_Diffusion.pdf",
    "GlossFree-UnbiasedEval.pdf",
    "From DataGloves to Deep Networks- A Survey on Sign Language Recognition.pdf",
    "Visual_Hand_Gesture_Recognition_with_Deep_Learning_A_Comprehensive_Review_of_Methods_Datasets_Challenges_and_Future_Research_Directions.pdf",
    "American_Sign_Language_Recognition_using_Deep_Learning_and_Computer_Vision.pdf",
]

if __name__ == "__main__":
    run(sys.argv[1:] or DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────
# CURATED CROPS
# The automatic region finder is a triage tool: it is right often enough to choose from, and
# wrong often enough that anything going into the thesis is cropped by hand. Each entry below is
# a box in PDF points (x0, y0, x1, y1) read off the rendered page and checked by eye.
# `run_curated()` writes these to borrowed/curated/ and is the only output the thesis uses.
# ─────────────────────────────────────────────────────────────────────────────
CURATED = {
    # id: (pdf, page, rect, short attribution used in the thesis caption)
    "gloss_vs_glossfree": (
        "GFSLT-VLP.pdf", 1, (300, 240, 556, 528),
        "Zhou et al., GFSLT-VLP (ICCV 2023), Figure 1"),
    "wholebody_keypoints": (
        "TwoStream-SLR.pdf", 15, (125, 66, 529, 283),
        "Chen et al., TwoStream-SLR (NeurIPS 2022), Figure 4"),
    "auslan_daily_diversity": (
        "Auslan-Daily.pdf", 1, (103, 223, 509, 378),
        "Shen et al., Auslan-Daily (NeurIPS 2023), Figure 1"),
    "auslan_daily_stats": (
        "Auslan-Daily.pdf", 6, (122, 69, 490, 222),
        "Shen et al., Auslan-Daily (NeurIPS 2023), Figure 3"),
    "auslan_daily_tasks": (
        "Auslan-Daily.pdf", 7, (120, 66, 490, 272),
        "Shen et al., Auslan-Daily (NeurIPS 2023), Figure 4"),
    "mmwl_test_subsets": (
        "MM-WLAuslan_Multi-View_Multi-Modal_Word-Level_Australian_Sign_Language_Recognition_Dataset.pdf",
        5, None, "Shen et al., MM-WLAuslan (2024), Figure 2"),
    "reported_vs_reproduced": (
        "GlossFree-UnbiasedEval.pdf", 2, (101, 162, 494, 311),
        "Sincan et al., unbiased evaluation of gloss-free SLT (CVIU 2026), Figure 1"),
    "ctc_forced_alignment": (
        "Online-CSLR-Translation.pdf", 12, None,
        "Zuo et al., online CSLR (EMNLP 2024), Figure 5"),
    "online_vs_offline": (
        "Online-CSLR-Translation.pdf", 2, (64, 64, 530, 302),
        "Zuo et al., online CSLR (EMNLP 2024), Figure 1"),
    "crosslingual_signs": (
        "CrossLingual-SLR.pdf", 1, (300, 398, 552, 482),
        "Wei and Chen, cross-lingual signs (ICCV 2023), Figure 1"),
    "representation_density": (
        "SignCL.pdf", 2, None,
        "Ye et al., SignCL (NeurIPS 2024), Figure 1"),
}

CUR_OUT = os.path.join(OUT, "curated")


def run_curated(only=None):
    os.makedirs(CUR_OUT, exist_ok=True)
    rows = []
    for key, (pdf, page, rect, attrib) in CURATED.items():
        if only and key not in only:
            continue
        if rect is None:
            print(f"  .. {key}: no box set yet (page {page} of {pdf})")
            continue
        doc = fitz.open(os.path.join(ROOT, pdf))
        pg = doc[page - 1]
        r = fitz.Rect(*rect) & pg.rect
        out_png = os.path.join(CUR_OUT, f"{key}.png")
        pg.get_pixmap(clip=r, dpi=DPI).save(out_png)
        rows.append(dict(id=key, source_pdf=pdf, page=page, rect=list(rect), attribution=attrib,
                         png=os.path.relpath(out_png, os.path.dirname(__file__))))
        print(f"  {key}: {r.width:.0f}x{r.height:.0f}pt -> {out_png}")
        doc.close()
    with open(os.path.join(CUR_OUT, "attribution.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows
