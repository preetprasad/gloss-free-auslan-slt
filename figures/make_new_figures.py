"""Three figures added on 10 October 2026 after a review of the dissertation's figures.

  fig18_pose_estimators        the same frames with the distributed AlphaPose keypoints and the DWPose re-extraction
  fig19_dataset_lengths        clip length and reference length in the two Auslan-Daily subsets (replaces a borrowed raster)
  fig20_published_reimplemented published against re-implemented BLEU-4 (replaces a borrowed radar chart)

Run with the interpreter that has OpenCV. Every number drawn is read from a file or asserted to occur in the Literature Review."""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import figstyle; figstyle.apply(plt)
# Line weights, ticks and grid from the figure standard (rules 1.3, 1.4). Fonts and colours stay as figstyle sets them.
plt.rcParams.update({
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "grid.alpha": 1.0,
    "axes.axisbelow": True,
})
C, FS, W = figstyle.C, figstyle.FS, figstyle.TARGET_W
INK = C["ink"]
def save(fig, name, tight=True):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight" if tight else None, dpi=300)
    plt.close(fig); print("wrote", name)

# ------------------------------------------------------------------ fig18: the two pose estimators on the same frames
def pose_figure(clip="video_11_6"):
    import cv2
    ap = np.load(os.path.join(REPO, "full_run", "clips", clip + ".npz"))["KPS"]                                  # (T, 136, 3) HALPE-136
    dw = np.load(os.path.join(REPO, "results", "dwpose_experiment", "dwpose_clips", clip + ".npz"))["KPS"]        # (T, 133, 3) COCO-WholeBody
    assert ap.shape[0] == dw.shape[0] and ap.shape[1] == 136 and dw.shape[1] == 133
    # The distributed keypoints are in the coordinates of the whole video and the re-extraction in those of the signer crop.
    # The 11 upper-body keypoints that both layouts share (nose to wrists) give the scale and shift between the two.
    ap = ap.copy(); n = min(ap.shape[0], dw.shape[0])
    A, B = ap[:n, :11, :2].reshape(-1, 2), dw[:n, :11, :2].reshape(-1, 2)
    keep = (ap[:n, :11, 2].reshape(-1) > 0.3) & (dw[:n, :11, 2].reshape(-1) > 0.5)
    Am, Bm = A[keep].mean(0), B[keep].mean(0)
    scale = float(((A[keep] - Am) * (B[keep] - Bm)).sum() / ((A[keep] - Am) ** 2).sum()); shift = Bm - scale * Am
    ap[:, :, :2] = scale * ap[:, :, :2] + shift
    off = float(np.median(np.linalg.norm(ap[:n, :11, :2].reshape(-1, 2)[keep] - B[keep], axis=1)))
    print(f"alignment of the two keypoint sets: scale {scale:.3f}, shift ({shift[0]:.0f}, {shift[1]:.0f}) px, median residual on shared body keypoints {off:.1f} px")
    assert off < 12, "the shared body keypoints do not align; the crop is not a fixed box"
    AH, DH = slice(94, 136), slice(91, 133)                                                                        # both hands, 2 x 21 keypoints
    cap = cv2.VideoCapture(os.path.join(REPO, "results", "dwpose_experiment", "rgb_clips", "Signer", clip + "_signer.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    T = min(len(frames), ap.shape[0])
    moving = [t for t in range(T) if dw[t, DH, 2].mean() > 0.5]
    picks = [moving[int(len(moving) * q)] for q in (0.15, 0.4, 0.62, 0.85)]
    pts = dw[picks][:, list(range(0, 13)) + list(range(91, 133)), :2].reshape(-1, 2)
    x0, y0 = pts.min(0) - 28; x1, y1 = pts.max(0) + 28
    chains = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    THR = 0.3                                                                          # one confidence threshold for both estimators
    FINGER = ["#C62828", "#FFB300", "#43A047", "#0D47A1", "#CE93D8"]                   # thumb, index, middle, ring, little: OpenPose hues, lightness 43/78/59/32/70
    FNAME = ["thumb", "index", "middle", "ring", "little"]
    HNAME = {0: "signer\u2019s left hand", 1: "signer\u2019s right hand"}            # first / second 21 hand keypoints of both layouts
    HCOL = {0: "#7986CB", 1: C["hands"]}                                               # left = light indigo, right = the frozen navy: differ in lightness, so they survive greyscale
    def hand(kps, hands, t, i): return kps[t, hands][21 * i:21 * (i + 1)]
    def soften(img):                                                                   # one uniform treatment for every panel: 35 % desaturation, 15 % dim, nothing local, nothing brightened
        f = img.astype(float) / 255.0; g = f @ np.array([0.299, 0.587, 0.114])
        return np.clip(0.85 * (0.65 * f + 0.35 * g[..., None]), 0, 1)
    # one square pixel window per frame and hand, shared by both estimators, fitted tightly to the union of both estimators' hand keypoints
    cen, side = {}, {}
    for t in picks:
        for i in (0, 1):
            h = hand(dw, DH, t, i); okk = h[:, 2] >= THR; qd = h[okk, :2] if okk.sum() >= 3 else h[:, :2]
            q = np.vstack([qd, hand(ap, AH, t, i)[:, :2]]); lo, hi = q.min(0), q.max(0)
            s = float(np.ptp(q, axis=0).max()) * 1.15 + 10; cap = float(np.ptp(qd, axis=0).max()) * 2.2 + 16
            if s <= cap: cen[t, i] = (lo + hi) / 2                                     # both skeletons fit: centre on their union
            else: s = cap; cen[t, i] = np.median(qd, axis=0)                           # a stray AlphaPose point may leave the window; the hand may not
            side[t, i] = max(s, 48.0)
    # keep every hand window 2 px inside the frame (same pixel window for both estimators), so its outline is never cut
    FH, FW = frames[picks[0]].shape[:2]
    for k in list(cen.keys()):
        s = side[k] = min(side[k], float(min(FH, FW)) - 4); cx, cy = cen[k]
        cen[k] = np.array([min(max(cx, s / 2 + 2), FW - s / 2 - 2), min(max(cy, s / 2 + 2), FH - s / 2 - 2)])
    S = max(side.values())                                                             # the largest window sets the minimum magnification
    # widen the context crop so that both boxes lie fully inside it, without leaving the image
    bx = np.array([[cen[k][0] - side[k] / 2, cen[k][1] - side[k] / 2, cen[k][0] + side[k] / 2, cen[k][1] + side[k] / 2] for k in cen])
    x0 = max(0.0, min(float(x0), bx[:, 0].min() - 4)); y0 = max(0.0, min(float(y0), bx[:, 1].min() - 4))
    x1 = min(float(FW), max(float(x1), bx[:, 2].max() + 4)); y1 = min(float(FH), max(float(y1), bx[:, 3].max() + 4))
    order = sorted((0, 1), key=lambda i: np.mean([cen[t, i][0] for t in picks]))      # columns follow image position
    # layout in inches at the printed size
    H = 4.6; top, bot, vgap = 0.46, 0.30, 0.07
    rh = (H - top - bot - 3 * vgap) / 4                                                # side of a hand panel = row height
    aspect = (x1 - x0) / (y1 - y0); ch_in = rh; cw_in = ch_in * aspect
    if cw_in > 1.35: cw_in = 1.35; ch_in = cw_in / aspect
    mag = (rh / S) / (cw_in / (x1 - x0))
    if mag < 3.2: cw_in *= mag / 3.2; ch_in *= mag / 3.2; mag = 3.2
    lab_w, g1, g_in, g2 = 0.80, 0.14, 0.04, 0.18
    total = lab_w + cw_in + g1 + 4 * rh + 2 * g_in + g2
    xc = (W - total) / 2 + lab_w
    hx = [xc + cw_in + g1, xc + cw_in + g1 + rh + g_in]; hx += [hx[1] + rh + g2, hx[1] + 2 * rh + g2 + g_in]
    fig = plt.figure(figsize=(W, H))
    def add(x, ytop, w, h):
        a = fig.add_axes([x / W, (H - ytop - h) / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([]); a.grid(False); return a
    def draw_hand(ax, h):                                                              # no white casing under the bones: it bleached the hand
        for chain, col in zip(chains, FINGER):
            for a, b in zip(chain[:-1], chain[1:]):
                good = h[a, 2] >= THR and h[b, 2] >= THR; xy = h[[a, b], :2]
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.9 if good else 0.75, ls="-" if good else (0, (1.6, 1.0)),
                        solid_capstyle="round", dash_capstyle="butt", zorder=4)
        for j in range(21):                                                            # joints 1.6 pt (< 2x the 0.9 pt bone), thin white casing; hollow below THR
            col = INK if j == 0 else FINGER[(j - 1) // 4]
            if h[j, 2] >= THR: ax.plot(h[j, 0], h[j, 1], "o", ms=1.6, mfc=col, mec="white", mew=0.3, zorder=5)
            else: ax.plot(h[j, 0], h[j, 1], "o", ms=1.6, mfc="none", mec=col, mew=0.55, zorder=5)
    for r, t in enumerate(picks):
        ytop = top + r * (rh + vgap); img = soften(frames[t])
        ax = add(xc, ytop + (rh - ch_in) / 2, cw_in, ch_in)                           # context: the signer crop, small
        ax.imshow(img, interpolation="bilinear"); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color(C["grey"])
        for i in order:                                                                # both boxes labelled in every frame, tag in the inner top corner facing the crop centre
            cx, cy = cen[t, i]; s = side[t, i]
            ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec=HCOL[i], lw=0.75, zorder=3))
            rightside = cx > (x0 + x1) / 2
            ax.annotate("L" if i == 0 else "R", xy=(cx - s / 2 if rightside else cx + s / 2, cy - s / 2), xycoords="data",
                        xytext=(1.6 if rightside else -1.6, -1.4), textcoords="offset points", fontsize=FS["tiny"], fontweight="bold", color=HCOL[i],
                        ha="left" if rightside else "right", va="top", zorder=4, annotation_clip=False,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        yc = H - (ytop + rh / 2)
        fig.text((xc - 0.08) / W, (yc + 0.08) / H, f"frame {t}", fontsize=FS["sub"], color=INK, ha="right", va="bottom")
        fig.text((xc - 0.08) / W, (yc + 0.04) / H, f"mean hand conf.\nAlphaPose {ap[t, AH, 2].mean():.2f}\nDWPose {dw[t, DH, 2].mean():.2f}",
                 fontsize=FS["tiny"], color=INK, ha="right", va="top", linespacing=1.25)
        for g, i in enumerate(order):
            cx, cy = cen[t, i]
            for e, (kps, hands) in enumerate(((ap, AH), (dw, DH))):
                ax = add(hx[2 * g + e], ytop, rh, rh)
                ax.imshow(img, interpolation="bilinear"); draw_hand(ax, hand(kps, hands, t, i))
                s = side[t, i]; ax.set_xlim(cx - s / 2, cx + s / 2); ax.set_ylim(cy + s / 2, cy - s / 2)
                for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.75); sp.set_color(HCOL[i])
                if r == 0: ax.text(0.5, 1.04, ("AlphaPose", "DWPose")[e], transform=ax.transAxes, fontsize=FS["sub"], color=INK, ha="center", va="bottom")
    fig.text((xc + cw_in / 2) / W, (H - top + 0.037) / H, "signer crop", fontsize=FS["sub"], color=INK, ha="center", va="bottom")
    for g, i in enumerate(order):
        xa, xb = hx[2 * g], hx[2 * g + 1] + rh
        fig.text((xa + xb) / 2 / W, (H - top + 0.22) / H, HNAME[i], fontsize=FS["label"], color=INK, ha="center", va="bottom")
        fig.add_artist(Line2D([xa / W, xb / W], [(H - top + 0.19) / H] * 2, transform=fig.transFigure, color=HCOL[i], lw=0.75))
    hs = [Line2D([], [], color=c, lw=0.9, marker="o", ms=2.4, mfc=c, mec="white", mew=0.3) for c in FINGER]
    hs += [Line2D([], [], color=INK, lw=0.9, marker="o", ms=2.4, mfc=INK, mec="white", mew=0.3),
           Line2D([], [], color=INK, lw=0.75, ls=(0, (1.6, 1.0)), marker="o", ms=2.4, mfc="none", mec=INK, mew=0.55)]
    fig.legend(hs, FNAME + [r"confidence $\geq$ 0.3", r"confidence $<$ 0.3"], loc="center", bbox_to_anchor=(0.5, 0.13 / H), ncol=7,
               frameon=False, fontsize=FS["sub"], handlelength=1.8, columnspacing=1.1, handletextpad=0.4)
    save(fig, "fig18_pose_estimators", tight=False)
    print(f"hand windows {min(side.values()):.0f} to {S:.0f} px square, each shared by both estimators; magnification at least {mag:.1f}x relative to the signer crop")
    print(f"clip {clip}: frames {picks}; median body offset {off:.1f} px; mean hand confidence over the four frames: AlphaPose {ap[picks][:, AH, 2].mean():.2f}, DWPose {dw[picks][:, DH, 2].mean():.2f}; whole clip {ap[:T, AH, 2].mean():.2f} / {dw[:T, DH, 2].mean():.2f}")

# ------------------------------------------------------------------ fig19: clip and reference lengths of the two subsets
def lengths_figure():
    import openpyxl
    d = os.path.join(REPO, "data", "raw", "Auslan-Daily", "Dataset Split Table", "Sign Language Alignment")
    data = {}
    for name, f in (("Communication", "Auslan_Daily_Communication.xlsx"), ("News", "Auslan_Daily_News.xlsx")):
        ws = openpyxl.load_workbook(os.path.join(d, f), read_only=True).worksheets[0]; it = ws.iter_rows(values_only=True); h = list(next(it))
        s, b, e = h.index("Subtitle"), h.index("Begin_Frame"), h.index("End_Frame")
        rows = [(int(r[e]) - int(r[b]), len(str(r[s]).split())) for r in it if r[s] is not None and r[b] is not None]
        data[name] = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.70), gridspec_kw=dict(wspace=0.30))   # 69 mm: inside the 50-70 mm band (rule 1.11)
    for ax, col, xlabel, bins, limit in ((axes[0], 0, "Clip length (frames at 25 frames per second)", np.arange(0, 701, 20), 256),
                                         (axes[1], 1, "Reference sentence length (tokens)", np.arange(0, 61, 2), None)):
        for (name, a), colour, hatch in zip(data.items(), (C["accent"], C["body"]), ("", "////")):
            ax.hist(np.clip(a[:, col], bins[0], bins[-1] - 1e-6), bins=bins, weights=np.ones(len(a)) / len(a) * 100, histtype="stepfilled", alpha=0.45, color=colour,
                    hatch=hatch, edgecolor=colour, linewidth=1.25, label=f"{name} (median {int(np.median(a[:, col]))})")
        if limit:
            ax.axvline(limit, color=INK, lw=0.75, ls="--", zorder=1); ax.text(limit + 8, ax.get_ylim()[1] * 0.97, "input limit, 256 frames", fontsize=FS["sub"], color=INK, va="top")
        ax.set_xlabel(xlabel, fontsize=FS["label"]); ax.set_ylabel("Share of clips (%)", fontsize=FS["label"]); ax.tick_params(labelsize=FS["sub"])
        ax.grid(axis="x", visible=False); ax.grid(axis="y", visible=True)
        ax.legend(frameon=False, fontsize=FS["sub"], loc="lower right", bbox_to_anchor=(1.0, 1.01), borderaxespad=0)
        ax.text(-0.16, 1.01, "(a)" if col == 0 else "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="left", va="bottom")
    save(fig, "fig19_dataset_lengths")
    for n, a in data.items(): print(f"{n}: {len(a)} clips; frames mean {a[:, 0].mean():.1f} median {np.median(a[:, 0]):.0f}, over 256 frames {100 * (a[:, 0] > 256).mean():.1f}%; tokens mean {a[:, 1].mean():.1f} median {np.median(a[:, 1]):.0f}")

# ------------------------------------------------------------------ fig20: published against re-implemented BLEU-4
def reimplemented_figure():
    review = open(os.path.join(REPO, "SignLanguage_Preet", "src", "review.tex")).read()
    rows = [("Vision-language pretraining\n(GFSLT-VLP)", (21.44, 21.97, 0.27), (11.00, 10.41, 0.17)),
            ("Frame contrastive loss\n(GFSLT-VLP + SignCL)", (22.74, 22.35, 0.28), (16.16, 11.22, 0.18))]
    assert all(f"{v:.2f}" in review for _, a, b in rows for v in a + b), "a value is not in the Literature Review table"
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.00), sharey=True, gridspec_kw=dict(wspace=0.08))   # 51 mm: at least the 50 mm of rule 1.11
    for ax, k, title in ((axes[0], 1, "PHOENIX14T"), (axes[1], 2, "CSL-Daily")):
        for y, row in enumerate(rows):
            pub, re_, sd = row[k]
            ax.plot([pub, re_], [y, y], color=C["grey"], lw=1.25, zorder=1)
            ax.scatter([pub], [y], s=20, facecolors="white", edgecolors=INK, linewidths=1.0, zorder=3, label="as published" if y == 0 else None)
            ax.errorbar([re_], [y], xerr=[sd], fmt="s", ms=4.5, color=C["accent"], ecolor=C["accent"], elinewidth=0.75, capsize=2, capthick=0.75, zorder=4, label="re-implemented in one pipeline (mean and standard deviation)" if y == 0 else None)
            ax.text(pub, y + 0.26, f"{pub:.2f}", ha="center", fontsize=FS["sub"], color=INK); ax.text(re_, y - 0.42, f"{re_:.2f}", ha="center", fontsize=FS["sub"], color=C["accent"])
        ax.text(0.0, 1.03, "(a)" if k == 1 else "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="left", va="bottom")
        ax.annotate(title, xy=(0.0, 1.03), xycoords="axes fraction", xytext=(17, 0), textcoords="offset points", fontsize=FS["label"], color=INK, ha="left", va="bottom")
        ax.set_ylim(-0.75, 1.65); ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=FS["sub"])
        ax.set_xlabel("BLEU-4", fontsize=FS["label"]); ax.tick_params(labelsize=FS["sub"]); ax.margins(x=0.18); ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=FS["sub"], ncol=2, loc="upper center", bbox_to_anchor=(0.56, 1.16))
    save(fig, "fig20_published_reimplemented")

if __name__ == "__main__":
    # NOTE (10 Oct 2026). Of 570 local clips that have both keypoint sets, 44 register to within 8 px on the 11 shared body keypoints
    # (scale 0.99 to 1.01, so the crop is a pure shift there); in most of the others the stored first detection is not the same person or
    # the crop moves. pose_figure() asserts the registration, so it can only be drawn on a registered clip (video_16_130 is not one).
    which = sys.argv[1:] or ["lengths", "reimpl", "pose"]
    if "lengths" in which: lengths_figure()
    if "reimpl" in which: reimplemented_figure()
    if "pose" in which: pose_figure()
