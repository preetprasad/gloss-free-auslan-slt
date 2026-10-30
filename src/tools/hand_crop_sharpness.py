"""Does FAST HAND MOTION blur the hands, and does that predict sign identifiability?

The 2026-08 classical-CV pass measured sharpness as `cv2.Laplacian(whole_grayscale_frame).var()`.
That is dominated by the static background and is STRUCTURALLY INCAPABLE of seeing motion blur on a
fast-moving hand -- so its null result cannot be used to rule hand blur out. This measures the hand
crops themselves.

Design note: absolute sharpness is not comparable across datasets (different cameras, resolutions,
codecs), so everything here is WITHIN the continuous corpus:
    does hand-crop sharpness fall as hand speed rises, and does it predict identification?
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

L_HAND, R_HAND = slice(91, 112), slice(112, 133)


def crop_sharpness(frame, pts, pad=0.35, size=64):
    """Contrast-normalised focus measure on a fixed-size hand crop.

    Dividing by the crop's own intensity variance decouples sharpness from contrast/brightness, and
    resizing to a fixed size decouples it from how many pixels the hand happens to occupy -- so a
    low value means genuinely smeared edges, not a small or dim hand.
    """
    h, w = frame.shape[:2]
    x0, y0 = np.nanmin(pts, 0)
    x1, y1 = np.nanmax(pts, 0)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return None
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1 = int(max(0, x0 - px)), int(min(w, x1 + px))
    y0, y1 = int(max(0, y0 - py)), int(min(h, y1 + py))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    c = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    c = cv2.resize(c, (size, size)).astype(np.float64)
    v = c.var()
    if v < 1e-6:
        return None
    return float(cv2.Laplacian(c, cv2.CV_64F).var() / v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot_csv", required=True)
    ap.add_argument("--pose_dirs", nargs="+", required=True)
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--rgb_suffix", default="_signer.mp4")
    ap.add_argument("--max_spans", type=int, default=350)
    ap.add_argument("--max_frames_per_span", type=int, default=8)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    spot = pd.read_csv(a.spot_csv)
    spot["cid"] = spot.Video_Name_Clip.map(lambda v: re.sub(r"_signer\.mp4$", "", str(v)))
    spot["g"] = spot.Query_Gloss.astype(str).str.strip().str.lower()
    pose = {}
    for d in a.pose_dirs:
        for f in glob.glob(os.path.join(d, "*.npz")):
            pose.setdefault(os.path.basename(f)[:-4], f)

    rows = []
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(spot))
    for i in order:
        if len(rows) >= a.max_spans:
            break
        r = spot.iloc[int(i)]
        p, v = pose.get(r.cid), os.path.join(a.rgb_dir, r.cid + a.rgb_suffix)
        if not p or not os.path.exists(v):
            continue
        with np.load(p) as z:
            k = z["KPS"].astype(np.float32)
        b, e = max(0, int(r.Begin_Frame)), min(len(k), max(int(r.Begin_Frame) + 3, int(r.End_Frame)))
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            continue
        picks = np.linspace(b, e - 1, min(a.max_frames_per_span, e - b)).astype(int)
        sh, sp = [], []
        scale = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=-1)
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            for hs in (L_HAND, R_HAND):
                pts = k[fi, hs, :2]
                conf = k[fi, hs, 2]
                pts = pts[conf > 0.3]
                if len(pts) < 8:
                    continue
                s = crop_sharpness(frame, pts)
                if s is not None:
                    sh.append(s)
            if 0 < fi < len(k) - 1 and scale[fi] > 1e-3:
                d = np.linalg.norm(k[fi + 1, L_HAND, :2].mean(0) - k[fi - 1, L_HAND, :2].mean(0))
                sp.append(d / (2 * scale[fi]))
        cap.release()
        if sh:
            rows.append({"cid": r.cid, "gloss": r.g, "sharp": float(np.mean(sh)),
                         "sharp_p10": float(np.percentile(sh, 10)),
                         "speed": float(np.mean(sp)) if sp else np.nan,
                         "span_frames": int(e - b)})
    d = pd.DataFrame(rows)
    print(f"measured {len(d)} spans over {d.cid.nunique()} clips\n")
    print("=== hand-crop sharpness (contrast-normalised, fixed 64x64 crop) ===")
    print(f"  mean {d.sharp.mean():.4f}   median {d.sharp.median():.4f}   "
          f"p10 {d.sharp.quantile(.1):.4f}   p90 {d.sharp.quantile(.9):.4f}")
    ok = d.dropna(subset=["speed"])
    print(f"\n=== DOES FAST HAND MOTION BLUR THE HANDS?  (n={len(ok)}) ===")
    print(f"  corr(hand speed, sharpness)       = {ok.speed.corr(ok.sharp):+.3f}")
    print(f"  corr(hand speed, worst-decile)    = {ok.speed.corr(ok.sharp_p10):+.3f}")
    q = ok.speed.quantile([.25, .75])
    slow, fast = ok[ok.speed <= q.iloc[0]], ok[ok.speed >= q.iloc[1]]
    print(f"  slowest quartile sharpness {slow.sharp.mean():.4f}  vs  fastest {fast.sharp.mean():.4f}"
          f"   ratio {fast.sharp.mean()/slow.sharp.mean():.2f}x")
    if a.out_csv:
        d.to_csv(a.out_csv, index=False)
        print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
