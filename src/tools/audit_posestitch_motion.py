"""Validation-safe motion-fidelity audit for duration-corrected PoseStitch data.

This is a diagnostic, not a model trainer.  It compares synthetic pose clips for which
``build_posestitch_corpus_v2.py --audit_jsonl`` stored segment provenance against real
continuous Auslan-Daily clips from a *training-only* manifest.  It deliberately omits
clip duration from the real-vs-synthetic diagnostic: duration was already corrected and
must not make a motion-quality conclusion trivially true.

The tool reports:
  * matched-length root-relative velocity, acceleration, jerk and pause density;
  * discontinuities at recorded synthetic joins versus interior real frame pairs; and
  * a shallow held-out logistic diagnostic using only those dynamics statistics.

It never reads a model checkpoint, target text, or official validation/test manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import warnings

import numpy as np


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_jsonl", required=True,
                    help="PoseStitch provenance sidecar generated with --audit_jsonl")
    ap.add_argument("--real_train_manifest", required=True,
                    help="training-only real Auslan-Daily TSV; first column is the pose path")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_conf", type=float, default=0.10)
    ap.add_argument("--root_indices", default="11,12",
                    help="COCO-WholeBody hip indices used as the pose root (default: 11,12)")
    ap.add_argument("--length_tolerance", type=float, default=0.15,
                    help="maximum relative length mismatch for a real/synthetic pair")
    ap.add_argument("--max_pairs", type=int, default=1000)
    ap.add_argument("--pause_threshold", type=float, default=0.0,
                    help="normalised median keypoint speed below which a frame counts as a pause. "
                         "0 (default) = CALIBRATE from the real clips at --pause_quantile, which is "
                         "the only defensible choice: the previous hard-coded 0.0125 was never "
                         "measured against real data and saturated the feature at 1.0")
    ap.add_argument("--pause_quantile", type=float, default=0.15,
                    help="when --pause_threshold 0, set the threshold to this quantile of the REAL "
                         "per-frame speed distribution, so 'pause' means 'slow for this corpus'")
    ap.add_argument("--coord_norm", default="auto", choices=["auto", "bbox", "frame"],
                    help="pose scale normalisation. Real Auslan-Daily DWPose npz files carry an "
                         "ALL-ZERO BBOX (657/657 verified), so bbox normalisation yields zero usable "
                         "frames; 'frame' matches slt/data.py coord_norm=frame, the space the encoder "
                         "actually consumes. 'auto' picks bbox only when a real box is present.")
    return ap.parse_args()


def _paths_from_manifest(path: str) -> list[str]:
    paths = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and "\t" in line:
            paths.append(line.split("\t", 1)[0])
    return paths


def _load_sidecar(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("clip_path") and row.get("segments"):
            rows.append(row)
    return rows


def _load_pose(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["KPS"].astype(np.float32), z["BBOX"].astype(np.float32)


def _normalised_root_relative(kps: np.ndarray, bbox: np.ndarray, roots: list[int],
                              min_conf: float,
                              coord_norm: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Return scale-normalised root-relative xy, a valid-frame mask, and the mode used.

    DWPose output is COCO-WholeBody-compatible: 11 and 12 are hips.  If a hip is
    unavailable for a frame, use the median of valid body points 0..16 rather than
    silently interpreting a face or hand landmark as the torso root.

    NORMALISATION (fixed 2026-09-09).  The original version divided by the BBOX, but
    **every** real Auslan-Daily DWPose npz stores an all-zero BBOX (verified on 657/657
    local clips): the DWPose extraction path never wrote one, which is exactly why the
    project's own recipe is `coord_norm=frame`.  Under bbox normalisation every frame
    failed `frame_ok`, so this audit produced ZERO usable frames on the real corpus and
    could only ever have "passed" on synthetic data with fabricated boxes.
    `frame` reproduces `slt/data.py::coord_norm=="frame"` (per-frame standardisation over
    all keypoints), i.e. it measures motion in the space the encoder actually consumes.
    "auto" uses the bbox when it is genuinely present and falls back to `frame` otherwise.
    """
    if kps.ndim != 3 or kps.shape[2] < 3 or bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"unexpected pose layout KPS={kps.shape}, BBOX={bbox.shape}")
    n = min(len(kps), len(bbox))
    kps, bbox = kps[:n], bbox[:n]
    x1, y1, w, h = (bbox[:, i] for i in range(4))
    nonzero = np.abs(kps).reshape(n, -1).sum(1) > 1e-6
    box_ok = (w > 1e-3) & (h > 1e-3)
    mode = coord_norm
    if coord_norm == "auto":
        mode = "bbox" if box_ok.mean() >= 0.5 else "frame"
    if mode not in ("bbox", "frame"):
        raise ValueError(f"coord_norm must be auto|bbox|frame, got {coord_norm!r}")

    xy = kps[..., :2].astype(np.float64).copy()
    if mode == "bbox":
        frame_ok = box_ok & nonzero
        xy[..., 0] = (xy[..., 0] - x1[:, None]) / np.where(frame_ok, w, 1.0)[:, None]
        xy[..., 1] = (xy[..., 1] - y1[:, None]) / np.where(frame_ok, h, 1.0)[:, None]
    else:
        frame_ok = nonzero
        flat = xy.reshape(n, -1)
        m = flat.mean(axis=1)[:, None, None]
        sd = flat.std(axis=1)[:, None, None] + 1e-6
        xy = (xy - m) / sd
    conf_ok = kps[..., 2] >= min_conf
    xy[~conf_ok] = np.nan

    usable_roots = [i for i in roots if 0 <= i < xy.shape[1]]
    with warnings.catch_warnings():   # a frame with no confident root/body point is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        root = np.nanmean(xy[:, usable_roots], axis=1) if usable_roots else np.full((n, 2), np.nan)
        fallback = np.nanmedian(xy[:, :min(17, xy.shape[1])], axis=1)
    root[~np.isfinite(root).all(1)] = fallback[~np.isfinite(root).all(1)]
    frame_ok &= np.isfinite(root).all(1)
    rel = xy - root[:, None, :]
    rel[~frame_ok] = np.nan
    return rel, frame_ok, mode


def _frame_speeds(rel: np.ndarray, frame_ok: np.ndarray) -> np.ndarray | None:
    """Per-frame-transition median keypoint speed, or None if the clip is unusable.

    Split out of _clip_features so the pause threshold can be calibrated on the real
    corpus before any feature is computed with it.
    """
    # Require successive real frames; invalid pose detections are not treated as zero motion.
    if len(rel) < 4:
        return None
    point_ok = np.isfinite(rel).all(2)
    dv = np.linalg.norm(rel[1:] - rel[:-1], axis=2)
    good_v = point_ok[1:] & point_ok[:-1] & frame_ok[1:, None] & frame_ok[:-1, None]
    with warnings.catch_warnings():        # an all-invalid frame pair is expected, not an error
        warnings.simplefilter("ignore", RuntimeWarning)
        speeds = np.nanmedian(np.where(good_v, dv, np.nan), axis=1)
    speeds = speeds[np.isfinite(speeds)]
    return speeds if len(speeds) >= 3 else None


def _clip_features(rel: np.ndarray, frame_ok: np.ndarray, pause_threshold: float) -> dict | None:
    speeds = _frame_speeds(rel, frame_ok)
    if speeds is None:
        return None
    accel = np.abs(np.diff(speeds))
    jerk = np.abs(np.diff(accel))
    return {
        "velocity_median": float(np.median(speeds)),
        "acceleration_median": float(np.median(accel)) if len(accel) else float("nan"),
        "jerk_median": float(np.median(jerk)) if len(jerk) else float("nan"),
        "pause_density": float(np.mean(speeds < pause_threshold)),
        "valid_speed_frames": int(len(speeds)),
    }


def _frame_gap(rel: np.ndarray, frame_ok: np.ndarray, a: int, b: int) -> float | None:
    if not (0 <= a < len(rel) and 0 <= b < len(rel)) or not (frame_ok[a] and frame_ok[b]):
        return None
    good = np.isfinite(rel[a]).all(1) & np.isfinite(rel[b]).all(1)
    if not np.any(good):
        return None
    return float(np.median(np.linalg.norm(rel[b, good] - rel[a, good], axis=1)))


def _join_gaps(rel: np.ndarray, frame_ok: np.ndarray, segments: list[dict]) -> list[float]:
    out = []
    for seg in segments[1:]:
        start = int(seg.get("start_frame", -1))
        transition = int(seg.get("transition_frames_before", 0))
        # Compare the last true frame of the prior citation clip with the first true
        # frame of this citation clip.  Interpolated frames are deliberately skipped:
        # an L2 blend could hide this statistic without improving sign motion.
        gap = _frame_gap(rel, frame_ok, start - transition - 1, start)
        if gap is not None:
            out.append(gap)
    return out


def _summary(values: list[float]) -> dict:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(a):
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None}
    return {"n": int(len(a)), "median": float(np.median(a)), "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
                  features: list[str] | None = None) -> dict:
    """Small dependency-free diagnostic; it is not a learned feature extractor."""
    mu, sd = train_x.mean(0), train_x.std(0)
    sd = np.where(sd > 1e-7, sd, 1.0)
    x = (train_x - mu) / sd
    xt = (test_x - mu) / sd
    xb = np.c_[np.ones(len(x)), x]
    xtb = np.c_[np.ones(len(xt)), xt]
    w = np.zeros(xb.shape[1], dtype=float)
    for _ in range(800):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
        grad = xb.T @ (p - train_y) / len(xb) + 1e-3 * np.r_[0.0, w[1:]]
        w -= 0.08 * grad
    prob = 1.0 / (1.0 + np.exp(-np.clip(xtb @ w, -30, 30)))
    pred = (prob >= 0.5).astype(int)
    return {"heldout_accuracy": float(np.mean(pred == test_y)),
            "heldout_balanced_accuracy": float(0.5 * (
                np.mean(pred[test_y == 1] == 1) + np.mean(pred[test_y == 0] == 0))),
            "features": features or ["velocity_median", "acceleration_median",
                                     "jerk_median", "pause_density"],
            "duration_included": False}


def _diagnostic_with_controls(train_x, train_y, test_x, test_y,
                              train_len, test_len, feature_names, rng, n_perm: int = 50) -> dict:
    """Main dynamics diagnostic PLUS the two controls the E0 spec requires.

    Reporting a separability number alone is not interpretable here:
      * LENGTH-ONLY BASELINE -- pairs are matched only to within --length_tolerance, so residual
        duration differences survive. If length alone separates the classes about as well as the
        dynamics features do, the "motion mismatch" is really a length mismatch and the whole
        conclusion is trivially true, which is precisely the failure the E0 spec warns about.
      * SHUFFLED-LABEL CONTROL -- the empirical chance level for this sample size and feature
        count. With few matched pairs a logistic fit can look confident on noise; comparing
        against permuted labels is the honest floor, not the nominal 0.5.
    """
    main = _fit_logistic(train_x, train_y, test_x, test_y, feature_names)
    length_only = _fit_logistic(train_len.reshape(-1, 1), train_y,
                                test_len.reshape(-1, 1), test_y, ["clip_length_only"])
    length_only["duration_included"] = True
    accs = []
    for _ in range(n_perm):
        perm = rng.permutation(train_y)
        accs.append(_fit_logistic(train_x, perm, test_x, test_y,
                                  feature_names)["heldout_balanced_accuracy"])
    main["length_only_baseline"] = length_only
    main["shuffled_label_control"] = {
        "n_permutations": n_perm,
        "balanced_accuracy_mean": float(np.mean(accs)),
        "balanced_accuracy_p95": float(np.percentile(accs, 95)),
    }
    main["verdict"] = (
        "dynamics separate real from synthetic beyond both controls"
        if (main["heldout_balanced_accuracy"] > main["shuffled_label_control"]["balanced_accuracy_p95"]
            and main["heldout_balanced_accuracy"] > length_only["heldout_balanced_accuracy"])
        else "NOT separable beyond controls -- E0 falsification gate says do NOT add a boundary module")
    return main


def main() -> None:
    a = parse()
    rng = np.random.default_rng(a.seed)
    roots = [int(v) for v in a.root_indices.split(",") if v.strip()]
    modes_used: set[str] = set()
    synth_rows = _load_sidecar(a.audit_jsonl)
    real_paths = [p for p in _paths_from_manifest(a.real_train_manifest) if os.path.exists(p)]
    if not synth_rows or not real_paths:
        raise SystemExit("need a non-empty sidecar and a training-only real manifest with existing paths")

    # PASS 1 -- normalise the real clips and, if asked, CALIBRATE the pause threshold on
    # their own speed distribution.  A fixed threshold cannot be right across normalisation
    # modes: bbox units and per-frame-standardised units differ by orders of magnitude.
    real_norm: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    real_speeds: list[np.ndarray] = []
    for path in real_paths:
        try:
            kps, bbox = _load_pose(path)
            rel, ok, _mode = _normalised_root_relative(kps, bbox, roots, a.min_conf, a.coord_norm)
            modes_used.add(_mode)
        except (OSError, KeyError, ValueError):
            continue
        sp = _frame_speeds(rel, ok)
        if sp is not None:
            real_norm.append((path, rel, ok, kps))
            real_speeds.append(sp)
    if not real_speeds:
        raise SystemExit("no compatible real pose clips after validity checks")

    # The normalisation mode must be IDENTICAL on both sides.  Auslan-Daily DWPose npz files
    # carry a zero BBOX while stitched MM-WLAuslan clips may carry a real one, so a per-clip
    # "auto" would silently normalise real clips by frame and synthetic clips by bbox and then
    # compare them -- a guaranteed false "mismatch".  Resolve on the real corpus and force it.
    if len(modes_used) != 1:
        raise SystemExit(f"internal: real corpus resolved to mixed modes {sorted(modes_used)}")
    resolved_norm = next(iter(modes_used))

    pause_threshold = a.pause_threshold
    calibration = {"mode": "explicit", "value": pause_threshold}
    if pause_threshold <= 0:
        pooled = np.concatenate(real_speeds)
        pause_threshold = float(np.quantile(pooled, a.pause_quantile))
        calibration = {"mode": "calibrated_on_real", "quantile": a.pause_quantile,
                       "value": pause_threshold, "real_frame_transitions": int(pooled.size),
                       "real_speed_median": float(np.median(pooled))}

    # PASS 2 -- features, now with a threshold that means something for this corpus.
    real_cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]] = []
    for path, rel, ok, kps in real_norm:
        feat = _clip_features(rel, ok, pause_threshold)
        if feat:
            real_cache.append((path, rel, ok, kps, feat))
    if not real_cache:
        raise SystemExit("no compatible real pose clips after validity checks")

    real_lengths = np.asarray([len(rel) for _, rel, _, _, _ in real_cache], dtype=int)
    pairs, synth_join, real_internal = [], [], []
    skipped = {"missing": 0, "layout": 0, "no_feature": 0, "length": 0}
    for row in synth_rows:
        if len(pairs) >= a.max_pairs:
            break
        path = row["clip_path"]
        if not os.path.exists(path):
            skipped["missing"] += 1
            continue
        try:
            kps, bbox = _load_pose(path)
            rel, ok, _mode = _normalised_root_relative(kps, bbox, roots, a.min_conf, resolved_norm)
            feat = _clip_features(rel, ok, pause_threshold)
        except (OSError, KeyError, ValueError):
            skipped["layout"] += 1
            continue
        if not feat:
            skipped["no_feature"] += 1
            continue
        n = len(rel)
        candidates = np.flatnonzero(np.abs(real_lengths - n) / max(n, 1) <= a.length_tolerance)
        if not len(candidates):
            skipped["length"] += 1
            continue
        # Avoid using the same closest real clip for every synthetic example: within the
        # pre-specified duration band, choose a real continuous sequence at random.
        idx = int(candidates[int(rng.integers(0, len(candidates)))])
        real_path, real_rel, real_ok, _, real_feat = real_cache[idx]
        pairs.append({"synthetic_path": path, "real_path": real_path,
                      "synthetic_frames": n, "real_frames": len(real_rel),
                      "synthetic": feat, "real": real_feat})
        synth_join.extend(_join_gaps(rel, ok, row["segments"]))
        # One interior real gap per recorded synthetic join; never use an end frame.
        for _ in row["segments"][1:]:
            if len(real_rel) < 3:
                continue
            point = int(rng.integers(1, len(real_rel) - 1))
            gap = _frame_gap(real_rel, real_ok, point - 1, point)
            if gap is not None:
                real_internal.append(gap)

    if len(pairs) < 20:
        raise SystemExit(f"only {len(pairs)} duration-matched pairs; audit cannot be interpreted")
    feature_names = ["velocity_median", "acceleration_median", "jerk_median", "pause_density"]
    sx = np.array([[p["synthetic"][f] for f in feature_names] for p in pairs])
    rx = np.array([[p["real"][f] for f in feature_names] for p in pairs])
    sl = np.array([float(p["synthetic_frames"]) for p in pairs])
    rl = np.array([float(p["real_frames"]) for p in pairs])
    finite = np.isfinite(sx).all(1) & np.isfinite(rx).all(1)
    sx, rx, sl, rl = sx[finite], rx[finite], sl[finite], rl[finite]
    order = rng.permutation(len(sx))
    cut = max(1, int(0.7 * len(order)))
    train, test = order[:cut], order[cut:]
    if not len(test):
        raise SystemExit("not enough pairs for held-out diagnostic")
    train_x = np.r_[sx[train], rx[train]]
    train_y = np.r_[np.ones(len(train)), np.zeros(len(train))]
    test_x = np.r_[sx[test], rx[test]]
    test_y = np.r_[np.ones(len(test)), np.zeros(len(test))]
    train_len = np.r_[sl[train], rl[train]]
    test_len = np.r_[sl[test], rl[test]]
    report = {
        "purpose": "motion-fidelity diagnostic only; no translation model or official evaluation used",
        "inputs": {"audit_jsonl": os.path.abspath(a.audit_jsonl),
                   "real_train_manifest": os.path.abspath(a.real_train_manifest),
                   "seed": a.seed, "root_indices": roots, "length_tolerance": a.length_tolerance,
                   "coord_norm_requested": a.coord_norm,
                   "coord_norm_used": resolved_norm,
                   "pause_threshold": pause_threshold,
                   "pause_threshold_calibration": calibration},
        "matched_pairs": int(len(sx)), "skipped": skipped,
        "synthetic_dynamics": {f: _summary(sx[:, i].tolist()) for i, f in enumerate(feature_names)},
        "real_dynamics": {f: _summary(rx[:, i].tolist()) for i, f in enumerate(feature_names)},
        "clip_frames_synthetic": _summary(sl.tolist()),
        "clip_frames_real": _summary(rl.tolist()),
        "join_gap_synthetic": _summary(synth_join),
        "interior_gap_real": _summary(real_internal),
        "real_vs_synthetic_diagnostic": _diagnostic_with_controls(
            train_x, train_y, test_x, test_y, train_len, test_len, feature_names, rng),
        "interpretation_guard": (
            "A separable diagnostic or a larger synthetic join gap is evidence of a residual motion "
            "mismatch, not evidence that smoothing will improve SLT. Any future intervention must be "
            "selected by a matched validation-only translation run and must not optimise an L2 proxy alone."
        ),
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"matched_pairs": report["matched_pairs"], "out_json": a.out_json,
                      "diagnostic": report["real_vs_synthetic_diagnostic"],
                      "join_gap_synthetic": report["join_gap_synthetic"],
                      "interior_gap_real": report["interior_gap_real"]}, indent=2))


if __name__ == "__main__":
    main()
