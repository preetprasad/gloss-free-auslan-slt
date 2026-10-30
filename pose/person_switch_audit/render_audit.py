"""Blind audit of the person tracker (job 6590) on official val+test clips. For sampled frames, draws the stored
index-0 skeleton and the tracked skeleton in colours assigned at random per frame (A = red, B = blue), plus four
context frames, so a labeller can say which skeleton is on the signer without knowing which method drew it.
Strata (seed 0): 40 override frames (CHOSEN != 0) from each of light (0,0.1], medium (0.1,0.3], heavy (>0.3) clips,
one frame per clip; plus 40 agreement frames (CHOSEN == 0) from affected clips, drawn with one skeleton only.
Writes panels + key.json OUTSIDE the repo (dataset frames). Usage: render_audit.py OUT_DIR TRACKED_DIR"""
import csv, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
out, tdir = Path(sys.argv[1]), Path(sys.argv[2])
assert ROOT not in out.resolve().parents; (out / "panels").mkdir(parents=True, exist_ok=True)
FRAC = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(open(ROOT / "results/ndet_per_clip.csv"))}
clips = []
for split, sdir in (("val", "val_clips_dwpose"), ("test", "test_clips_dwpose")):
    for l in open(ROOT / f"results/real_data_learning_curve/{split}.tsv"):
        c = l.split("\t")[0].rsplit("/", 1)[1][:-4]; clips.append((c, split, ROOT / "results" / sdir / f"{c}.npz"))
rng = random.Random(0)
def bin_of(f): return "light" if 0 < f <= 0.1 else "medium" if 0.1 < f <= 0.3 else "heavy" if f > 0.3 else "none"
strata = {"light": [], "medium": [], "heavy": []}
for c, sp, p in clips:
    b = bin_of(FRAC[c])
    if b != "none": strata[b].append((c, sp, p))
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
def draw(img, k, col):
    for a, b in EDGES:
        if k[a, 2] > 0.3 and k[b, 2] > 0.3: cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), col, 4)
    for i in list(range(91, 133)) + [0]:
        if k[i, 2] > 0.3: cv2.circle(img, tuple(map(int, k[i, :2])), 4, col, -1)
def frames_of(c):
    cap = cv2.VideoCapture(str(ROOT / f"results/dwpose_experiment/rgb_clips/Signer/{c}_signer.mp4")); fs = []
    while True:
        ok, f = cap.read()
        if not ok: return fs
        fs.append(f)
items, key = [], {}
for b, pool in strata.items():
    for c, sp, p in rng.sample(pool, 40):
        ch = np.load(tdir / f"{c}.npz")["CHOSEN"]; t = rng.choice([i for i in range(len(ch)) if ch[i] != 0])
        items.append((b, c, sp, p, t, False))
aff = [x for x in clips if FRAC[x[0]] > 0]
for c, sp, p in rng.sample(aff, 40):
    Tz = np.load(tdir / f"{c}.npz"); ch, Sz = Tz["CHOSEN"], np.load(p)["KPS"]
    # CHOSEN == 0 AND the stored skeleton really is that detection (0.14% of such frames differ >50 px across runs)
    zs = [i for i in range(len(ch)) if ch[i] == 0 and np.abs(Sz[i, :, :2] - Tz["KPS"][i, :, :2]).max() < 5.0]
    if zs: items.append(("agree", c, sp, p, rng.choice(zs), True))
rng.shuffle(items)
for n, (b, c, sp, p, t, agree) in enumerate(items):
    S = np.load(p)["KPS"]; T = np.load(tdir / f"{c}.npz")["KPS"]; fs = frames_of(c)
    assert len(fs) == len(S) == len(T), (c, len(fs), len(S), len(T))
    img = fs[t].copy(); pid = f"p{n:03d}"
    if agree:
        assert np.abs(S[t, :, :2] - T[t, :, :2]).max() < 5.0
        draw(img, S[t], (0, 0, 255)); key[pid] = dict(stratum=b, clip=c, split=sp, frame=t, agree=True)
    else:
        a_is_tracked = rng.random() < 0.5
        draw(img, T[t] if a_is_tracked else S[t], (0, 0, 255)); draw(img, S[t] if a_is_tracked else T[t], (255, 80, 0))
        key[pid] = dict(stratum=b, clip=c, split=sp, frame=t, agree=False, A="tracked" if a_is_tracked else "index0")
    h = 700; main = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    ctx = [cv2.resize(fs[i], (int(fs[i].shape[1] * 170 / fs[i].shape[0]), 170)) for i in np.linspace(0, len(fs) - 1, 4).astype(int)]
    ctx = np.vstack(ctx[:4]) if sum(x.shape[0] for x in ctx) <= h else np.vstack(ctx)
    ctx = cv2.copyMakeBorder(ctx, 0, max(0, h - ctx.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:h]
    panel = np.hstack([main, np.zeros((h, 12, 3), np.uint8), ctx])
    cap = f"{pid}  A = RED" + ("" if agree else ", B = BLUE") + f"  (frame {t+1}/{len(fs)})   right: 4 context frames, no skeleton"
    panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(panel, cap, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / "panels" / f"{pid}.png"), panel)
json.dump(key, open(out / "key.json", "w"), indent=1)
from collections import Counter
print(len(items), "panels;", Counter(k["stratum"] for k in key.values()), "| A=tracked in", sum(k.get("A") == "tracked" for k in key.values()), "of", sum(not k["agree"] for k in key.values()), "override panels")
