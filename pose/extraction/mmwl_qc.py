"""MM-WLAuslan DWPose QC — per-clip pose quality from the extracted npz confidence.

Cheap quality gate for the ISLR pretraining set (the DWPose npz already carries the
confidence channel, so no re-processing of pixels). For each clip:
  mean_hand_conf : mean confidence over the 42 hand keypoints (91:133) on valid frames
  mean_conf      : mean confidence over all 133 keypoints
  zero_frac      : fraction of all-zero (failed) frames
  n_frames       : frame count
A clip passes if mean_hand_conf >= --min_hand_conf, zero_frac <= --max_zero_frac and
n_frames >= --min_frames. Writes qc.csv (all clips + pass flag) and keep.txt (passing ids).
Signing is hand-dominated, so hand confidence is the decisive signal (same rationale that
drove the Auslan-Daily DWPose jump: hand-keypoint confidence 0.13->0.82).
"""
import os, glob, csv, argparse
import numpy as np

HAND = slice(91, 133)   # COCO-WholeBody: left 91-111, right 112-132


def stats(f):
    with np.load(f) as z:
        k = np.asarray(z["KPS"], dtype=np.float32)      # (T,133,3)
    if k.ndim != 3 or k.shape[1] < 133:
        return None
    T = k.shape[0]
    conf = k[..., 2]
    nonzero = np.abs(k).reshape(T, -1).sum(1) > 1e-6
    valid = conf[nonzero] if nonzero.any() else conf
    hand = valid[:, HAND]
    return {
        "n_frames": int(T),
        "zero_frac": round(float((~nonzero).mean()), 4),
        "mean_conf": round(float(valid.mean()) if valid.size else 0.0, 4),
        "mean_hand_conf": round(float(hand.mean()) if hand.size else 0.0, 4),
        "hand_conf_p10": round(float(np.percentile(hand, 10)) if hand.size else 0.0, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.csv and <out>_keep.txt)")
    ap.add_argument("--min_hand_conf", type=float, default=0.30)
    ap.add_argument("--max_zero_frac", type=float, default=0.30)
    ap.add_argument("--min_frames", type=int, default=8)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pose_dir, "*.npz")))
    rows, keep = [], []
    npass = 0
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        s = stats(f)
        if s is None:
            rows.append({"id": sid, "n_frames": 0, "zero_frac": 1, "mean_conf": 0,
                         "mean_hand_conf": 0, "hand_conf_p10": 0, "pass": 0})
            continue
        ok = int(s["mean_hand_conf"] >= a.min_hand_conf and s["zero_frac"] <= a.max_zero_frac
                 and s["n_frames"] >= a.min_frames)
        rows.append({"id": sid, **s, "pass": ok})
        if ok:
            keep.append(sid); npass += 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "n_frames", "zero_frac", "mean_conf",
                                           "mean_hand_conf", "hand_conf_p10", "pass"])
        w.writeheader(); w.writerows(rows)
    open(a.out + "_keep.txt", "w").write("\n".join(keep) + ("\n" if keep else ""))
    n = len(rows)
    if n:
        mh = np.array([r["mean_hand_conf"] for r in rows], float)
        print(f"[qc] clips={n} pass={npass} ({100*npass/n:.1f}%) "
              f"mean_hand_conf: mean={mh.mean():.3f} median={np.median(mh):.3f} "
              f"p10={np.percentile(mh,10):.3f} | thr hand>={a.min_hand_conf} "
              f"zero<={a.max_zero_frac} frames>={a.min_frames}", flush=True)
    print(f"[qc] wrote {a.out}.csv and {a.out}_keep.txt", flush=True)


if __name__ == "__main__":
    main()
