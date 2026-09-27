"""Ground-truth panels for tracker v2: which detected person is the signer?

For each clip in gt_clips.json, about 10 frames (every ceil(T/10)-th frame) are drawn with EVERY detection from the
all-detections extraction: each person's skeleton, hands and detector box in its own colour, labelled with a letter.
Letters are assigned in a random order per frame (random.Random(7)), so a rater cannot tell which detection came first
or which any tracker would choose. Four context frames (no drawing) show whom the crop follows.
Writes panels and key.json (panel -> clip, frame, letter -> detection index) OUTSIDE the repo (dataset frames).
Usage: render_gt.py ALLDET_NPZ_DIR OUT_DIR
"""
import json, math, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert ROOT not in out.resolve().parents, "frames are dataset material; write outside the repo"
(out / "panels").mkdir(parents=True, exist_ok=True)
G = json.load(open(ROOT / "results/tracker_v2/gt_clips.json"))
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]
COLS = [(0, 0, 255), (255, 90, 0), (0, 220, 255), (255, 0, 255), (0, 200, 0), (255, 255, 0), (128, 0, 255)]
LET = "ABCDEFG"
rng = random.Random(7); key = {}; n = 0
for part in ("dev", "heldout"):
    for clip, stratum in G[part]:
        d = np.load(src / f"{clip}.npz"); KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]
        cap = cv2.VideoCapture(str(ROOT / f"results/dwpose_experiment/rgb_clips/Signer/{clip}_signer.mp4")); fs = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fs.append(f)
        assert len(fs) == len(ND), (clip, len(fs), len(ND))
        T = len(fs); stride = max(1, math.ceil(T / 10))
        ctx = [cv2.resize(fs[i], (int(fs[i].shape[1] * 170 / fs[i].shape[0]), 170)) for i in np.linspace(0, T - 1, 4).astype(int)]
        for t in range(0, T, stride):
            img = fs[t].copy(); m = int(ND[t]); order = list(range(m)); rng.shuffle(order)
            letters = {}
            for slot, det in enumerate(order):
                col = COLS[slot % len(COLS)]; k, s = KA[t, det], SA[t, det]
                x0, y0, x1, y1 = map(int, DB[t, det]); cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
                for a, b in EDGES:
                    if s[a] > 0.3 and s[b] > 0.3: cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), col, 4)
                for i in range(91, 133):
                    if s[i] > 0.3: cv2.circle(img, tuple(map(int, k[i])), 3, col, -1)
                letters[LET[slot]] = det
            # labels last, so no box or skeleton covers a letter; nudge down any label that would overlap another
            placed = []
            for slot, det in enumerate(order):
                col = COLS[slot % len(COLS)]; x0, y0 = int(DB[t, det][0]), int(DB[t, det][1])
                lx, ly = min(max(4, x0 + 4), img.shape[1] - 40), max(34, y0 + 34)
                while any(abs(lx - px) < 40 and abs(ly - py) < 40 for px, py in placed):
                    ly += 42
                placed.append((lx, ly))
                cv2.rectangle(img, (lx - 4, ly - 30), (lx + 30, ly + 6), col, -1)
                cv2.rectangle(img, (lx - 4, ly - 30), (lx + 30, ly + 6), (255, 255, 255), 2)
                cv2.putText(img, LET[slot], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            h = 700; main = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
            c = np.vstack(ctx); c = cv2.copyMakeBorder(c, 0, max(0, h - c.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:h]
            panel = np.hstack([main, np.zeros((h, 12, 3), np.uint8), c])
            pid = f"g{n:04d}"; n += 1
            cap_txt = f"{pid}  people detected: {m} ({', '.join(LET[:m]) or 'none'})  frame {t + 1}/{T}   right: context frames"
            panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(panel, cap_txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(out / "panels" / f"{pid}.png"), panel)
            key[pid] = dict(part=part, stratum=stratum, clip=clip, frame=t, ndet=m, letters=letters)
json.dump(key, open(out / "key.json", "w"), indent=1)
print(n, "panels;", sum(v["part"] == "dev" for v in key.values()), "dev;", sum(v["ndet"] == 0 for v in key.values()), "with no detection")
