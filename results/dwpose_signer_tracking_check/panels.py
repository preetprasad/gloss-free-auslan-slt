"""6 evidence panels. Run with the twin python; frame decode is delegated to ~/.venvs/ds (cv2, no numpy work there)."""
import os, csv, json, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
H = "."; OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{OUT}/panels", exist_ok=True)
VID = f"{H}/results/dwpose_experiment/rgb_clips/Signer"
rows = list(csv.DictReader(open(f"{OUT}/per_clip.csv")))
rng = np.random.default_rng(0)
ok = lambda r: int(r["T_dw"]) >= 40
unexp = [r for r in rows if r["catA_stored_KPS"] == "5_unexplained" and ok(r) and r["split"] in ("train", "val", "test")]
swit = [r for r in rows if r["catB_index0"] == "3_switch_explained" and ok(r)]
pick = [("unexplained", unexp[i]) for i in sorted(rng.choice(len(unexp), 3, replace=False))] + [("switch", swit[i]) for i in sorted(rng.choice(len(swit), 3, replace=False))]
ARMS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
def rel(k):
    ok_ = (k[:, 5, 2] > .3) & (k[:, 6, 2] > .3); mid = (k[:, 5, :2] + k[:, 6, :2]) / 2
    w = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)[:, None]
    out = np.full((len(k), 4, 2), np.nan)
    out[ok_] = ((k[:, [7, 8, 9, 10], :2] - mid[:, None]) / np.maximum(w, 1e-6)[:, None])[ok_]
    return out
def offset(a, KA, SA):
    """crop offset (AP original-frame coords -> crop coords) by torso alignment with the DW person whose shoulder mid-point offset is most constant"""
    best = None
    for j in range(KA.shape[1]):
        m = (a[:, 5, 2] > .3) & (a[:, 6, 2] > .3) & (SA[:, j, 5] > .3) & (SA[:, j, 6] > .3)
        if m.sum() < 8: continue
        ma = ((a[:, 5, :2] + a[:, 6, :2]) / 2)[m]; md = ((KA[:, j, 5] + KA[:, j, 6]) / 2)[m]
        w = np.median(np.linalg.norm(a[m, 5, :2] - a[m, 6, :2], axis=1)); d = ma - md
        s = float(np.mean(np.std(d, 0)) / w)
        if best is None or s < best[0]: best = (s, np.median(d, 0), j)
    return best
manifest = []
for pi, (kind, r) in enumerate(pick):
    n = r["clip"]; a = np.load(f"{H}/full_run/clips_signer/{n}.npz")["KPS"]; d = np.load(f"{H}/results/dwpose_experiment/alldet_local/{n}.npz")
    KA, SA, K = d["KP_ALL"], d["SC_ALL"], d["KPS"]; T = len(K)
    P0 = np.zeros((T, 133, 3), np.float32); P0[:, :, :2] = KA[:, 0]; P0[:, :, 2] = SA[:, 0]
    best = offset(a, KA, SA); T_ap = len(a)
    fr = np.linspace(0, min(T, T_ap) - 1, 3).astype(int)
    tmp = f"{OUT}/panels/_frames_{n}"; os.makedirs(tmp, exist_ok=True)
    subprocess.run([os.path.expanduser("~/.venvs/ds/bin/python"), "-c",
        "import cv2,sys\nc=cv2.VideoCapture(sys.argv[1])\nfor i in map(int,sys.argv[3:]):\n c.set(1,i);ok,f=c.read()\n if ok: cv2.imwrite(sys.argv[2]+'/f%d.png'%i,f)",
        f"{VID}/{n}_signer.mp4", tmp] + [str(i) for i in fr], check=True)
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    ra, r0, rk = rel(a), rel(P0), rel(K)
    for c, (ci, nm) in enumerate([(2, "right/left wrist y"), (2, "wrist x"), (1, "elbow y")]):
        ax = fig.add_subplot(gs[0, c]); comp = 1 if c in (0, 2) else 0
        for src, col, ls, lab in ((ra, "k", "-", "AlphaPose Signer"), (r0, "tab:red", "-", "DWPose index-0 (training)"), (rk, "tab:blue", "--", "DWPose v1-tracked (stored KPS)")):
            for side, (idx, al) in enumerate(((ci if c < 2 else 0, 1.0), (ci + 1 if c < 2 else 1, 0.5))):
                ax.plot(src[:, idx, comp], color=col, ls=ls, alpha=al, lw=1.3, label=lab if side == 0 else None)
        ax.set_title(["wrists, y (rel. shoulder-mid / width)", "wrists, x", "elbows, y"][c], fontsize=9)
        if comp == 1: ax.invert_yaxis()
        if c == 0: ax.legend(fontsize=7)
        ax.set_xlabel("frame (solid = sides 1, faded = sides 2)", fontsize=7)
    for c, i in enumerate(fr):
        ax = fig.add_subplot(gs[1, c]); p = f"{tmp}/f{i}.png"
        if os.path.exists(p): ax.imshow(Image.open(p).convert("RGB"))
        def draw(k, col, lab, off=(0, 0)):
            if np.all(k[i, 5:7, 2] > .3) or True:
                for u, v in ARMS: ax.plot([k[i, u, 0] - off[0], k[i, v, 0] - off[0]], [k[i, u, 1] - off[1], k[i, v, 1] - off[1]], color=col, lw=2, label=lab)
        draw(P0, "red", "index-0"); 
        if not np.array_equal(K[i, :, :2], P0[i, :, :2]): draw(K, "cyan", "v1-tracked")
        if best and best[0] < 0.3 and i < T_ap:
            draw(a, "lime", "AlphaPose (shifted)", best[1])
        ax.set_title(f"frame {i}  ndet={int(d['NDET'][i])} chosen={int(d['CHOSEN'][i])}", fontsize=8); ax.axis("off")
        if c == 0:
            h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7, loc="lower left")
    fig.suptitle(f"{n} [{r['split']}] class={kind}  T_ap={r['T_ap']} T_dw={r['T_dw']}  r(AP,idx0)={float(r['r_idx0']):.2f}  r(AP,stored KPS)={float(r['r_kps']):.2f}  "
                 f"r(AP,v2 path)={float(r['r_v2_path']):.2f}  best other={max(float(r['r_best_index']), float(r['r_best_tracklet']) if r['r_best_tracklet']!='nan' else -1):.2f}\\n"
                 f"switch frames={r['n_switch']}  multi-det share={float(r['share_multidet']):.2f}  rms AP/idx0={float(r['rms_ap']):.2f}/{float(r['rms_idx0']):.2f}  "
                 f"AP-shift torso-alignment sd={best[0]:.2f} (DW person {best[2]})" if best else n, fontsize=9)
    fig.tight_layout(); fn = f"{OUT}/panels/panel_{pi+1}_{kind}_{n}.png"; fig.savefig(fn, dpi=80); plt.close(fig)
    manifest.append(fn); print(fn)
    for f in os.listdir(tmp): os.remove(f"{tmp}/{f}")
    os.rmdir(tmp)
json.dump(manifest, open(f"{OUT}/panels/panels.json", "w"), indent=1)
