"""FIG C8 -- what the encoder actually localises, inside the pipeline, on one real test clip.

Not a schematic: the numbers are read out of a trained checkpoint by running the clip through it.
  checkpoint  results/ckpts/comm_dwpose_vlp_ctc.pt  (VLP-80 -> CTC, the recorded 10.27 model)
  clip        video_66_70, official Communication test split
  video       results/dwpose_experiment/rgb_clips/Signer/video_66_70_signer.mp4
HARNESS VALIDATION: the greedy decode produced here reproduces the hypothesis recorded in
results/semantic_adequacy_grading/hyps_vlp.jsonl for this clip, character for character, before any
internal state is plotted.

What it shows: the auxiliary CTC head assigns a probability to each of 2,988 pseudo-gloss labels
at every frame. For the content words the model goes on to produce, that probability has a clear
peak at a plausible moment and ranks first of 2,988 there. For the two content words the model
misses, it is flat and near zero -- rank 1731 and rank 173 at their best frame. The failure is
not decoding or word order; the encoder has no evidence for those signs anywhere in the clip.
"""
import json, os, sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
import figstyle
figstyle.apply(plt)
C, FS, TARGET_W = figstyle.C, figstyle.FS, figstyle.TARGET_W
from config import Config
from slt.data import normalize_clip
from slt.reference_model import build_model

CLIP, CKPT = "video_66_70", os.path.join(ROOT, "results/ckpts/comm_dwpose_vlp_ctc.pt")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config().to_dict()})
vocab, g2i = ck["vocab"], ck["gloss2idx"]

npz = np.load(os.path.join(ROOT, "results/test_clips_dwpose", CLIP + ".npz"))
feats, mask = normalize_clip(npz["KPS"], npz["BBOX"], cfg)
model = build_model(feats.shape[1], len(vocab), cfg); model.load_state_dict(ck["model"]); model.eval()
head = torch.nn.Linear(cfg.d_model, len(g2i) + 1); head.load_state_dict(ck["ctc_head"]); head.eval()

with torch.no_grad():
    mem = model.encode(torch.from_numpy(feats).unsqueeze(0), ~torch.from_numpy(mask).unsqueeze(0))
    P = torch.softmax(head(mem), -1)[0].numpy()                 # (T, 1 + |G|)
    ys = torch.tensor([[1]])
    for _ in range(40):
        nxt = int(model.decode_step(ys, mem, ~torch.from_numpy(mask).unsqueeze(0), None)[:, -1].argmax())
        ys = torch.cat([ys, torch.tensor([[nxt]])], 1)
        if nxt == 2: break
GREEDY = " ".join(vocab[int(t)] for t in ys[0, 1:] if int(t) not in (0, 1, 2))

rec = {json.loads(l)["idx"]: json.loads(l) for l in open(os.path.join(ROOT, "results/semantic_adequacy_grading/hyps_vlp.jsonl"))}
man = [l.split("\t")[0] for l in open(os.path.join(ROOT, "results/TEST_manifest_official.tsv"))]
idx = next(i for i, p in enumerate(man) if os.path.basename(p)[:-4] == CLIP)
assert GREEDY.strip() == rec[idx]["hyp_greedy"].strip(), "harness mismatch: decode does not reproduce the record"
REF, HYP = rec[idx]["ref"], rec[idx]["hyp_beam"]
print(f"harness OK - greedy reproduces the recorded hypothesis for {CLIP}")

T = P.shape[0]
WORDS = ["why", "watch", "some", "children", "learning", "reflections", "mirrors"]
rows, ranks = [], []
for w in WORDS:
    col = P[:, g2i[w]]
    t = int(col.argmax())
    ranks.append(int((P[t, 1:] > col[t]).sum()) + 1)
    rows.append(col)
rows = np.array(rows)
produced = set(HYP.split())

# ── frames ──────────────────────────────────────────────────────────────────
NF = 6
show = (np.linspace(0.04, 0.96, NF) * (T - 1)).round().astype(int)
kp = npz["KPS"]; m = kp[..., 2] > 0.3
pts = np.concatenate([kp[:, :11, :2][m[:, :11]], kp[:, 91:133, :2][m[:, 91:133]]])
x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
side = max(x1 - x0, y1 - y0) * 1.12
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
cap = cv2.VideoCapture(os.path.join(ROOT, "results/dwpose_experiment/rgb_clips/Signer", CLIP + "_signer.mp4"))
frames = {}
for f in show:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
    if ok:
        im = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); H, W = im.shape[:2]
        frames[f] = im[int(max(0, cy - side/2)):int(min(H, cy + side/2)),
                       int(max(0, cx - side/2)):int(min(W, cx + side/2))]
cap.release()

fig = plt.figure(figsize=(TARGET_W, 4.95))
# Two rows, one narrow column for the colourbar, so the frame strip and the heatmap share
# exactly one time axis -- the alignment-ribbon idiom this literature expects.
gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.55], width_ratios=[1, 0.022],
                      hspace=0.38, wspace=0.02)

axf = fig.add_subplot(gs[0, 0]); axf.set_xlim(0, T - 1); axf.set_yticks([]); axf.grid(False)
for _sp in axf.spines.values(): _sp.set_visible(False)
axf.tick_params(labelbottom=False, bottom=False)
strip = np.concatenate([frames[f] for f in show], axis=1)
axf.imshow(strip, extent=[0, T - 1, 0, 1], aspect="auto")
for k in range(1, NF):
    axf.axvline((T - 1) * k / NF, color="white", lw=1.0)
# number each tile and give its frame index, so it can be found on the frame axis of (b)
for k, f in enumerate(show):
    axf.text((T - 1) * (k + 0.5) / NF, -0.03, f"({k + 1})  frame {int(f)}",
             ha="center", va="top", fontsize=FS["sub"], color=C["ink"], clip_on=False)
axf.set_title("(a) the clip, six frames evenly spaced", fontsize=FS["sub"], loc="left", pad=3,
              color=C["ink"])

axh = fig.add_subplot(gs[1, 0], sharex=axf)
im = axh.imshow(rows, aspect="auto", cmap="magma_r", vmin=0, vmax=rows.max(),
                extent=[0, T - 1, len(WORDS) - 0.5, -0.5], interpolation="nearest")
axh.set_yticks(range(len(WORDS)))
axh.set_yticklabels([f"{w}   (rank {r})" for w, r in zip(WORDS, ranks)], fontsize=FS["label"])
# leave a fixed gutter between the tick labels and the heatmap for the status markers
axh.tick_params(axis="y", length=0, pad=15)
for i, w in enumerate(WORDS):
    got = w in produced
    # Latin Modern has no U+25CB; mathtext supplies both marks from the CM symbol font
    axh.annotate(r"$\bullet$" if got else r"$\circ$", xy=(0, i), xycoords=axh.get_yaxis_transform(),
                 xytext=(-7, 0), textcoords="offset points", ha="center", va="center",
                 fontsize=FS["label"], color=C["good"] if got else C["warn"],
                 annotation_clip=False)
    if not got:
        inset = (T - 1) * 0.004
        axh.add_patch(Rectangle((inset, i - 0.42), T - 1 - 2 * inset, 0.84, fill=False,
                                ec=C["warn"], lw=1.1, ls=(0, (3, 2)), zorder=5))
# the six frames shown in (a): ticks on the frame axis, numbered as in (a), and a mark on top
axh.set_xticks(show)
axh.set_xticklabels([f"{int(f)}\n({k + 1})" for k, f in enumerate(show)], fontsize=FS["sub"])
axh.plot(show, np.full(NF, 1.0), ls="none", marker="v", ms=4.5, color=C["ink"],
         clip_on=False, zorder=6,
         transform=offset_copy(axh.get_xaxis_transform(), fig=fig, y=4.5, units="points"))
axh.set_xlabel(f"frame  ({T} frames, 25\u2009fps); ticks mark the frames shown in (a)",
               fontsize=FS["label"])
axh.grid(False)
axh.set_title("(b) probability the auxiliary CTC head gives each reference content word, per frame",
              fontsize=FS["sub"], loc="left", pad=13, color=C["ink"])
axc = fig.add_subplot(gs[1, 1])
cb = fig.colorbar(im, cax=axc)
cb.set_label("probability", fontsize=FS["sub"]); cb.ax.tick_params(labelsize=FS["sub"])

axh.annotate(f"signed:  “{REF}”\nmodel:   “{HYP}”",
             xy=(0, 0), xycoords="axes fraction", xytext=(0, -58), textcoords="offset points",
             fontsize=FS["label"], color=C["ink"], ha="left", va="top", linespacing=1.4,
             annotation_clip=False)
fig.savefig(os.path.join(HERE, "figC8_inside_the_pipeline.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figC8_inside_the_pipeline.pdf"), bbox_inches="tight")
print("figC8 written |", {w: r for w, r in zip(WORDS, ranks)})
