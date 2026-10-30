"""How much would each literature pose-cleaning step actually touch in THIS project's DWPose input?

Descriptive only (no model, no BLEU). Clips: official VALIDATION DWPose (800) + the local train-side DWPose
experiment clips (592 after removing the 29 test and 36 val clips that folder also holds). Test clips not read.

Measures, per body part: confidence scale (DWPose/rtmlib SimCC scores are not probabilities; min > 0, can exceed 1),
share of keypoints under thresholds, "low-confidence hand" frames and the length of their runs (what gap
interpolation could fill), all-zero frames (what the loader already drops), and jitter in the loader's own
frame-normalised features (via slt.data.normalize_clip) with the residual a Savitzky-Golay filter would remove.
Run: ~/.pyenv/versions/3.14.3/bin/python pose_cleaning_headroom.py
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

REPO = "."
sys.path.insert(0, os.path.join(REPO, "src"))
from config import Config                      # noqa: E402
from slt.data import normalize_clip, load_pose  # noqa: E402

PARTS = {"body": range(0, 17), "feet": range(17, 23), "face": range(23, 91),
         "lhand": range(91, 112), "rhand": range(112, 133)}
THR = (0.2, 0.3, 0.5)


def runs(mask):
    """Lengths of consecutive True runs in a 1-D bool array."""
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def self_check(cfg):
    # 1) Savitzky-Golay (window 5, order 2) is exact on a quadratic, so its residual is a pure jitter measure.
    t = np.arange(20, dtype=float)
    q = 0.3 * t ** 2 - 2 * t + 1
    assert np.allclose(savgol_filter(q, 5, 2), q)
    # 2) normalize_clip lays features out as [x, y, conf] per keypoint: conf dims must equal the raw scores.
    kps, bbox = load_pose(sorted(glob.glob(f"{REPO}/results/val_clips_dwpose/*.npz"))[0])
    feats, _ = normalize_clip(kps, bbox, cfg)
    ok = np.abs(kps).reshape(len(kps), -1).sum(1) > 1e-6
    assert feats.shape[1] == 399 and np.allclose(feats[:, 2::3], kps[ok][..., 2][: len(feats)], atol=1e-6)
    print("self-check passed")


def main():
    cfg = Config()
    assert cfg.coord_norm == "frame" and cfg.num_keypoints == 133 and cfg.conf_threshold == 0.0
    self_check(cfg)
    val = sorted(glob.glob(f"{REPO}/results/val_clips_dwpose/*.npz"))
    # the experiment folder holds 29 official-test and 36 validation clips too: keep only the train-side remainder
    skip = {os.path.basename(p) for p in val + glob.glob(f"{REPO}/results/test_clips_dwpose/*.npz")}
    exp = [p for p in sorted(glob.glob(f"{REPO}/results/dwpose_experiment/dwpose_clips/*.npz"))
           if os.path.basename(p) not in skip]
    assert len(val) == 800 and 500 < len(exp) < 657, (len(val), len(exp))
    paths = val + exp
    conf = {p: [] for p in PARTS}
    frames = zero_frames = 0
    hand_low = {t: [0, 0] for t in THR}      # [low-hand frames, hand-frames]
    hand_runs = {t: [] for t in THR}
    clips_any_low = {t: 0 for t in THR}
    acc_hi, acc_lo, step, sg_res = [], [], [], []
    for p in paths:
        kps, bbox = load_pose(p)
        T = len(kps)
        frames += T
        live = np.abs(kps).reshape(T, -1).sum(1) > 1e-6
        zero_frames += int((~live).sum())
        k = kps[live]
        for name, idx in PARTS.items():
            conf[name].append(k[:, list(idx), 2].ravel())
        any_low = {t: False for t in THR}
        for h in ("lhand", "rhand"):
            mc = k[:, list(PARTS[h]), 2].mean(1)
            for t in THR:
                low = mc < t
                hand_low[t][0] += int(low.sum()); hand_low[t][1] += len(mc)
                hand_runs[t] += runs(low)
                any_low[t] |= bool(low.any())
        for t in THR:
            clips_any_low[t] += any_low[t]
        # jitter in the loader's own features (frame-normalised x, y)
        f, _ = normalize_clip(kps, bbox, cfg)
        if len(f) < 7:
            continue
        xy = f.reshape(len(f), 133, 3)
        hand = list(PARTS["lhand"]) + list(PARTS["rhand"])
        H = xy[:, hand, :2]
        c = xy[:, hand, 2]
        a = np.linalg.norm(H[2:] - 2 * H[1:-1] + H[:-2], axis=-1)       # (T-2, 42) second difference
        cm = c[1:-1]
        acc_hi.append(a[cm >= 0.5]); acc_lo.append(a[cm < 0.3])
        step.append(np.linalg.norm(H[1:] - H[:-1], axis=-1).ravel())
        sg_res.append(np.linalg.norm(H - savgol_filter(H, 5, 2, axis=0), axis=-1).ravel())
    n = len(paths)
    print(f"clips {n}; frames {frames}; all-zero frames {zero_frames} ({100 * zero_frames / frames:.3f}%)")
    print("confidence by part: min / median / p99 / max; % below 0.2 / 0.3 / 0.5; % exactly 0")
    for name, v in conf.items():
        v = np.concatenate(v)
        below = " / ".join(f"{100 * (v < t).mean():.1f}" for t in THR)
        print(f"  {name:5s} {v.min():.3f} / {np.median(v):.3f} / {np.percentile(v, 99):.3f} / {v.max():.3f};"
              f" {below}; {100 * (v == 0).mean():.2f}")
    print("hand frames with mean hand confidence below t (per hand), clips with any, run lengths of such frames")
    for t in THR:
        r = np.array(hand_runs[t]) if hand_runs[t] else np.array([0])
        print(f"  t={t}: {100 * hand_low[t][0] / hand_low[t][1]:.2f}% of hand-frames; clips with any "
              f"{100 * clips_any_low[t] / n:.1f}%; runs n={len(hand_runs[t])}, <=2 frames "
              f"{100 * (r <= 2).mean():.1f}%, <=3 {100 * (r <= 3).mean():.1f}%, median {np.median(r):.0f}, "
              f"max {r.max()}")
    hi, lo = np.concatenate(acc_hi), np.concatenate(acc_lo)
    st, sg = np.concatenate(step), np.concatenate(sg_res)
    print("hand keypoints, frame-normalised units (loader features):")
    print(f"  |2nd difference| median: conf>=0.5 {np.median(hi):.4f} (n={len(hi)}), "
          f"conf<0.3 {np.median(lo):.4f} (n={len(lo)}), ratio {np.median(lo) / np.median(hi):.2f}")
    print(f"  frame-to-frame step median {np.median(st):.4f}; Savitzky-Golay(5,2) residual median "
          f"{np.median(sg):.4f} = {100 * np.median(sg) / np.median(st):.1f}% of the median step")
    norm_stats(paths)


def norm_stats(paths, t=0.3):
    """coord_norm='frame' pools ALL 133 points (incl. out-of-frame legs/feet) into one per-frame mean/std. How much
    do those statistics jitter frame to frame, vs the same statistics over confident points only (conf >= t)?
    Units: pixels for the mean, relative change for the std."""
    dm_all, dm_conf, ds_all, ds_conf = [], [], [], []
    for p in paths:
        k = np.load(p)["KPS"].astype(np.float64)
        xy, c = k[..., :2], k[..., 2]
        m_all = xy.reshape(len(k), -1).mean(1)
        s_all = xy.reshape(len(k), -1).std(1)
        w = np.repeat((c >= t)[..., None], 2, -1).reshape(len(k), -1)
        v = xy.reshape(len(k), -1)
        m_c = (v * w).sum(1) / np.maximum(w.sum(1), 1)
        s_c = np.sqrt((((v - m_c[:, None]) ** 2) * w).sum(1) / np.maximum(w.sum(1), 1))
        dm_all.append(np.abs(np.diff(m_all))); dm_conf.append(np.abs(np.diff(m_c)))
        ds_all.append(np.abs(np.diff(s_all)) / s_all[1:]); ds_conf.append(np.abs(np.diff(s_c)) / s_c[1:])
    f = lambda a: np.nanmedian(np.concatenate(a))                        # noqa: E731
    g = lambda a: np.nanpercentile(np.concatenate(a), 95)                # noqa: E731
    print(f"frame-norm statistics, frame-to-frame change (median / p95): mean all-points {f(dm_all):.2f} / "
          f"{g(dm_all):.2f} px vs conf>={t} {f(dm_conf):.2f} / {g(dm_conf):.2f} px; std all-points "
          f"{100 * f(ds_all):.2f}% / {100 * g(ds_all):.2f}% vs conf>={t} {100 * f(ds_conf):.2f}% / "
          f"{100 * g(ds_conf):.2f}%")


def pixel_jitter(paths):
    """Hand jitter in RAW pixels (confident points, conf >= 0.5) vs the signer's shoulder width, so it can be set
    against literature noise-sensitivity curves that are stated in input units."""
    res, sw = [], []
    for p in paths:
        k = np.load(p)["KPS"].astype(np.float64)
        if len(k) < 7:
            continue
        h = k[:, 91:133]
        r = np.linalg.norm(h[..., :2] - savgol_filter(h[..., :2], 5, 2, axis=0), axis=-1)
        res.append(r[h[..., 2] >= 0.5])
        s = k[:, 5, :2] - k[:, 6, :2]
        sw.append(np.linalg.norm(s, axis=-1)[(k[:, 5, 2] >= 0.5) & (k[:, 6, 2] >= 0.5)])
    r, s = np.concatenate(res), np.median(np.concatenate(sw))
    print(f"raw-pixel hand SG(5,2) residual, conf>=0.5: median {np.median(r):.2f} px, p95 {np.percentile(r, 95):.2f} px; "
          f"median shoulder width {s:.1f} px (residual = {100 * np.median(r) / s:.2f}% of shoulder width)")


GT_NPZ = "/private/tmp/claude-501/-Users-preet-Honours/c69cfc0c-6e97-4020-b91d-72bcf24d0e82/scratchpad/gt/npz"


def v2_masked_runs():
    """Tracker v2 (TUNED) on the 60 labelled all-detections clips: how long are the gaps it masks, and where are they?
    These clips were drawn from the three switch bins, so this is NOT a corpus rate."""
    sys.path.insert(0, os.path.join(REPO, "results", "tracker_v2"))
    import tracker_v2 as tv2
    paths = sorted(glob.glob(f"{GT_NPZ}/*.npz"))
    if not paths:
        print("v2 masked runs: GT all-detections npz not present, skipped"); return
    T = masked = 0
    interior, edge = [], []
    for p in paths:
        _, ch = tv2.choose(np.load(p), tv2.TUNED)
        m = ch < 0
        T += len(m); masked += int(m.sum())
        i = 0
        while i < len(m):                                 # classify each masked run as interior or at a clip edge
            if not m[i]:
                i += 1; continue
            j = i
            while j < len(m) and m[j]:
                j += 1
            (edge if i == 0 or j == len(m) else interior).append(j - i)
            i = j
    a = np.array(interior) if interior else np.array([0])
    print(f"tracker v2 on {len(paths)} GT clips: masked {masked}/{T} frames ({100 * masked / T:.1f}%); runs interior "
          f"{len(interior)} (frames {int(a.sum())}; <=2 frames {100 * (a <= 2).mean():.0f}% of runs, <=3 "
          f"{100 * (a <= 3).mean():.0f}%; median {np.median(a):.0f}, max {a.max()}), at a clip edge {len(edge)} "
          f"(frames {sum(edge)})")


def self_check_runs():
    assert runs([True, True, False, True, False, False, True, True, True]) == [2, 1, 3]
    assert runs([False, False]) == []


if __name__ == "__main__":
    self_check_runs()
    main()
    val = sorted(glob.glob(f"{REPO}/results/val_clips_dwpose/*.npz"))
    pixel_jitter(val)
    v2_masked_runs()
