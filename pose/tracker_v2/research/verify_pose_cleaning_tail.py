"""Verifier check of one lens claim: "the raw hand SG(5,2) residual tail (p95 26.9 px) is identity errors (person
switches in index-0 pose), which tracking fixes, not smoothing".

Same residual as pose_cleaning_headroom.pixel_jitter (val DWPose clips, hand keypoints conf >= 0.5). A frame is
"near a switch" if it lies within +-2 frames (the SG window half-width) of a switch-sized jump: a single-frame move of
the torso centre (mean of shoulders 5,6 and hips 11,12) or change in shoulder width exceeding 40% of the clip's median
shoulder width (the 2026-09-22b jump definition). Reports what share of the above-p95 residuals sit near a switch vs
the share of all residual samples that do, and the same split by hand speed.
Run: ~/.pyenv/versions/3.14.3/bin/python verify_pose_cleaning_tail.py
"""
import glob

import numpy as np
from scipy.signal import savgol_filter

VAL = "./results/val_clips_dwpose/*.npz"


def near_switch(k, frac=0.4, halo=2):
    """(T,) bool: frames within +-halo of a switch-sized single-frame jump."""
    sh = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=-1)
    sw = np.median(sh)
    c = k[:, [5, 6, 11, 12], :2].mean(1)
    jump = np.zeros(len(k), bool)
    jump[1:] = (np.linalg.norm(np.diff(c, axis=0), axis=-1) > frac * sw) | (np.abs(np.diff(sh)) > frac * sw)
    out = np.zeros(len(k), bool)
    for t in np.flatnonzero(jump):               # transition between t-1 and t touches frames t-1-halo .. t+halo
        out[max(0, t - 1 - halo): t + halo + 1] = True
    return out


def self_check():
    T = 20
    k = np.zeros((T, 133, 3))
    k[:, 5, :2], k[:, 6, :2], k[:, 11, :2], k[:, 12, :2] = [100, 100], [200, 100], [110, 300], [190, 300]
    k[:, :, 2] = 0.9
    assert not near_switch(k).any()
    k[10:, :, :2] += 300                          # a different person from frame 10 on
    m = near_switch(k)
    assert m[7:13].all() and not m[:7].any() and not m[13:].any(), np.flatnonzero(m)
    print("self-check passed")


def main():
    res, near, speed = [], [], []
    paths = sorted(glob.glob(VAL))
    assert len(paths) == 800
    for p in paths:
        k = np.load(p)["KPS"].astype(np.float64)
        if len(k) < 7:
            continue
        h = k[:, 91:133]
        r = np.linalg.norm(h[..., :2] - savgol_filter(h[..., :2], 5, 2, axis=0), axis=-1)   # (T, 42)
        sp = np.zeros_like(r)
        sp[1:] = np.linalg.norm(np.diff(h[..., :2], axis=0), axis=-1)
        ok = h[..., 2] >= 0.5
        ns = np.repeat(near_switch(k)[:, None], 42, 1)
        res.append(r[ok]); near.append(ns[ok]); speed.append(sp[ok])
    r, n, s = np.concatenate(res), np.concatenate(near), np.concatenate(speed)
    p95 = np.percentile(r, 95)
    tail = r > p95
    print(f"residual samples {len(r)}; median {np.median(r):.2f} px, p95 {p95:.2f} px (headroom script: 2.15 / 26.91)")
    print(f"share of ALL samples near a switch: {100 * n.mean():.2f}%")
    # Only boolean INDEXING below: `tail & ~n` on named locals is silently done in place by numpy 1.26 under
    # Python 3.14 (see verify_numpy_elision_hazard.py), which corrupted the first version of this script.
    n_t, s_t = n[tail], s[tail]
    q = np.percentile(s, 90)
    print(f"share of ABOVE-p95 samples near a switch: {100 * n_t.mean():.2f}%")
    print(f"above-p95 samples NOT near a switch: {100 * (1 - n_t.mean()):.1f}% of the tail")
    print(f"above-p95 samples whose frame-to-frame hand step is in the top decile (> {q:.1f} px): "
          f"{100 * (s_t > q).mean():.1f}% (vs 10% by construction overall)")
    print(f"above-p95 samples NOT near a switch but in the top speed decile: "
          f"{100 * (s_t[~n_t] > q).sum() / len(s_t):.1f}% of the tail")
    assert abs(n_t.mean() + (~n_t).mean() - 1) < 1e-12 and tail.sum() == len(n_t)


if __name__ == "__main__":
    self_check()
    main()
