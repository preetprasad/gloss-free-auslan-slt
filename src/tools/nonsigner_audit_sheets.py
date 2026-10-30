"""B4 visual audit sheets for the non-signer gate: 3 frames per crop with the saved skeleton drawn on, 10 crops per
sheet, labelled PASS (voiced at tau_0.50) or DECL. The crops are the tracked worker's own inputs, so keypoint pixel
coordinates align with the video frames. Usage: python tools/nonsigner_audit_sheets.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

NS = Path(__file__).resolve().parents[2] / "results" / "nonsigner"
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]
HAND_L, HAND_R = range(91, 112), range(112, 133)


def frames(mp4, idx):
    cap, out = cv2.VideoCapture(str(mp4)), {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            out[i] = f
        i += 1
    return out


def draw(img, kp):
    for a, b in EDGES:
        if kp[a, 2] >= 0.3 and kp[b, 2] >= 0.3:
            cv2.line(img, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 255, 0), 2)
    for j in list(HAND_L) + list(HAND_R):
        if kp[j, 2] >= 0.3:
            cv2.circle(img, tuple(map(int, kp[j, :2])), 2, (0, 0, 255), -1)
    for j in range(23, 91):
        if kp[j, 2] >= 0.3:
            cv2.circle(img, tuple(map(int, kp[j, :2])), 1, (255, 200, 0), -1)
    return img


def main():
    r = json.load(open(NS / "nonsigner_gate_results.json"))
    D = json.load(open(NS / "decodes.json"))["neg"]
    items = [(c, "PASS") for c in r["B4_audit"]["passing"]] + [(c, "DECL") for c in r["B4_audit"]["declined"]]
    rows = []
    for n, (c, tag) in enumerate(items):
        kps = np.load(NS / "npz" / f"{c}.npz")["KPS"]
        T = len(kps)
        idx = sorted({int(T * q) for q in (0.2, 0.5, 0.8)})
        fr = frames(NS / "b4_mp4" / f"{c}.mp4", set(idx))
        tiles = []
        for i in idx:
            img = fr.get(i, np.zeros((360, 480, 3), np.uint8)).copy()
            img = draw(img, kps[i])
            tiles.append(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))))
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        row = np.hstack(tiles)
        label = f"#{n:02d} {tag} ce={D[c]['ce']:.2f} '{D[c]['hyp']}' {c}"
        row = cv2.copyMakeBorder(row, 26, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(row, label[:110], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        rows.append(row)
    w = max(x.shape[1] for x in rows)
    rows = [cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows]
    out = NS / "b4_sheets"
    out.mkdir(exist_ok=True)
    for s in range(0, len(rows), 10):
        cv2.imwrite(str(out / f"sheet_{s // 10}.jpg"), np.vstack(rows[s:s + 10]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"wrote {len(range(0, len(rows), 10))} sheets to {out}")


if __name__ == "__main__":
    main()
