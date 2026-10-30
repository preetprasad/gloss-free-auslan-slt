"""6 evidence panels for arm G: index-0 (red) vs G (lime) on frames where they differ, with the AlphaPose signer mapped into the crop by the
fitted similarity transform (white). Top row: arm signature vs time (AlphaPose / index-0 / G). Run with the twin python (frames via ~/.venvs/ds).
Choice of clips (seeded, fixed rule, before looking): 2 clips index-0 fails and G passes the r > 0.5 test, 2 clips where index-0 passes but
G differs on >= 10 frames (partial bystander switches), 2 clips where G differs from tracker v2 on >= 5 frames (what G adds over v2). T >= 40."""
import os, csv, json, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
H = "."; OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{OUT}/panels", exist_ok=True)
sys.path.insert(0, f"{H}/src/tools")
import wG_signerguided as W
tv2 = W.load_tv2(f"{H}/results/tracker_v2")
VID = f"{H}/results/dwpose_experiment/rgb_clips/Signer"; DW = f"{H}/results/dwpose_experiment/alldet_local"; AP = f"{H}/full_run/clips_signer"
rows = list(csv.DictReader(open(f"{OUT}/per_clip_g.csv")))
f = lambda r, k: float(r[k]); multi = [r for r in rows if r["multi"] == "1" and int(r["T"]) >= 40]
rng = np.random.default_rng(0)
A = [r for r in multi if not f(r, "r_idx0") > 0.5 and f(r, "r_g") > 0.5]
B = [r for r in multi if f(r, "r_idx0") > 0.5 and int(r["c0_ne_g"]) >= 10]
C = [r for r in multi if int(r["v2_ne_g"]) >= 5]
pick = [("idx0_fails_G_ok", A[i]) for i in sorted(rng.choice(len(A), 2, replace=False))] + \
       [("partial_switch", B[i]) for i in sorted(rng.choice(len(B), 2, replace=False))] + \
       [("G_differs_from_v2", C[i]) for i in sorted(rng.choice(len(C), 2, replace=False))]
ARMS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]


def rel(k):
    ok_ = (k[:, 5, 2] > .3) & (k[:, 6, 2] > .3); mid = (k[:, 5, :2] + k[:, 6, :2]) / 2
    w = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)[:, None]
    out = np.full((len(k), 4, 2), np.nan); out[ok_] = ((k[:, [7, 8, 9, 10], :2] - mid[:, None]) / np.maximum(w, 1e-6)[:, None])[ok_]
    return out


manifest = []
for pi, (kind, r) in enumerate(pick):
    n = r["clip"]; a = np.load(f"{AP}/{n}.npz")["KPS"]; d = dict(np.load(f"{DW}/{n}.npz")); T = len(d["NDET"])
    g = W.build_g(d, a, tv2); G = g["KPS"]
    P0 = np.zeros((T, 133, 3), np.float32); P0[:, :, :2] = d["KP_ALL"][:, 0]; P0[:, :, 2] = d["SC_ALL"][:, 0]
    diff = np.where(~np.isclose(P0, G).all(axis=(1, 2)))[0]
    fr = diff[np.unique(np.linspace(0, len(diff) - 1, 3).round().astype(int))] if len(diff) else np.array([0])
    cands, DK, DC, DV, AK, AC = W.prepare(d, a, tv2); st, info = W.fit_transform(AK, AC, DK, DC, DV)
    s_, t_ = (info["s_"], info["tr_"]) if st == "ok" else (None, None)
    tmp = f"{OUT}/panels/_frames_{n}"; os.makedirs(tmp, exist_ok=True)
    subprocess.run([os.path.expanduser("~/.venvs/ds/bin/python"), "-c",
        "import cv2,sys\nc=cv2.VideoCapture(sys.argv[1])\nfor i in map(int,sys.argv[3:]):\n c.set(1,i);ok,f=c.read()\n if ok: cv2.imwrite(sys.argv[2]+'/f%d.png'%i,f)",
        f"{VID}/{n}_signer.mp4", tmp] + [str(int(i)) for i in fr], check=True)
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    ra, r0, rg = rel(a), rel(P0), rel(G)
    for c, ttl in enumerate(["wrists, y (rel. shoulder-mid / width)", "wrists, x", "elbows, y"]):
        ax = fig.add_subplot(gs[0, c]); comp = 1 if c in (0, 2) else 0
        for src, col, ls, lab in ((ra, "k", "-", "AlphaPose Signer"), (r0, "tab:red", "-", "DWPose index-0 (training today)"), (rg, "tab:green", "--", "G (signer-guided)")):
            for side, (idx, al) in enumerate(((2 if c < 2 else 0, 1.0), (3 if c < 2 else 1, 0.5))):
                ax.plot(src[:, idx, comp], color=col, ls=ls, alpha=al, lw=1.3, label=lab if side == 0 else None)
        ax.set_title(ttl, fontsize=9)
        if comp == 1: ax.invert_yaxis()
        if c == 0: ax.legend(fontsize=7)
        ax.set_xlabel("frame (solid = side 1, faded = side 2)", fontsize=7)
    for c, i in enumerate(fr):
        ax = fig.add_subplot(gs[1, c]); p = f"{tmp}/f{int(i)}.png"
        if os.path.exists(p): ax.imshow(Image.open(p).convert("RGB"))
        def draw(k, col, lab, tr=None):
            for u, v in ARMS:
                xy = k[i][[u, v], :2]
                if tr is not None: xy = tr[0] * xy + tr[1]
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2, label=lab)
        draw(P0, "red", "index-0"); draw(G, "lime", "G")
        if s_ is not None and i < len(a) and a[i, 5:7, 2].min() > .3: draw(a, "white", "AlphaPose signer (fitted transform)", (s_, t_))
        ax.set_title(f"frame {int(i)}  ndet={int(d['NDET'][i])} G-det={int(g['chosen'][i])} source={['ap','ap_mask','carry','v2prior','v2_fallback'][g['source'][i]]}", fontsize=8); ax.axis("off")
        if c == 0:
            h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7, loc="lower left")
    fig.suptitle(f"{n} class={kind}  T={T}  frames G != index-0: {len(diff)}  r(AP,idx0)={f(r, 'r_idx0'):.2f}  r(AP,v1)={f(r, 'r_v1'):.2f}  r(AP,v2)={f(r, 'r_v2'):.2f}  r(AP,G)={f(r, 'r_g'):.2f}  "
                 f"fit s={info.get('s', float('nan')):.2f} inliers {info.get('n_inlier')}/{info.get('n_fit')}  fallback='{g['info']['reason']}'", fontsize=9)
    fig.tight_layout(); fn = f"{OUT}/panels/panel_{pi + 1}_{kind}_{n}.png"; fig.savefig(fn, dpi=80); plt.close(fig)
    manifest.append(fn); print(fn)
    for x in os.listdir(tmp): os.remove(f"{tmp}/{x}")
    os.rmdir(tmp)
json.dump(manifest, open(f"{OUT}/panels/panels.json", "w"), indent=1)
