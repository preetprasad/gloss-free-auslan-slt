#!/usr/bin/env python3
"""Audit DWPose clip health before changing the translation input pipeline.

RTMW/DWPose emits COCO-WholeBody-133 in this order: body (17), feet (6),
face (68), left hand (21), right hand (21).  The audit only describes the
already-extracted tensors; it does not alter a training manifest or pose file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "body_and_feet": slice(0, 23),
    "face": slice(23, 91),
    "left_hand": slice(91, 112),
    "right_hand": slice(112, 133),
}
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.clip_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ clips in {args.clip_dir}")

    summary = {
        "clip_dir": str(args.clip_dir.resolve()),
        "clips": len(paths),
        "layout": "COCO-WholeBody-133: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand",
        "groups": {},
        "frame_counts": [],
        "invalid_or_zero_frame_count": 0,
        "all_zero_clips": [],
        "malformed_clips": [],
    }
    values = {name: [] for name in GROUPS}
    low_counts = {name: {str(t): 0 for t in THRESHOLDS} for name in GROUPS}
    total_counts = {name: 0 for name in GROUPS}

    for index, path in enumerate(paths, start=1):
        try:
            with np.load(path) as payload:
                kps = np.asarray(payload["KPS"], dtype=np.float32)
        except Exception as error:
            summary["malformed_clips"].append({"clip": path.name, "error": str(error)})
            continue
        if kps.ndim != 3 or kps.shape[1:] != (133, 3) or not np.isfinite(kps).all():
            summary["malformed_clips"].append({"clip": path.name, "shape": list(kps.shape)})
            continue
        summary["frame_counts"].append(int(kps.shape[0]))
        zero_frames = np.abs(kps).sum(axis=(1, 2)) <= 1e-6
        summary["invalid_or_zero_frame_count"] += int(zero_frames.sum())
        if bool(zero_frames.all()):
            summary["all_zero_clips"].append(path.name)
        for name, group in GROUPS.items():
            confidence = kps[:, group, 2].reshape(-1)
            values[name].append(confidence)
            total_counts[name] += confidence.size
            for threshold in THRESHOLDS:
                low_counts[name][str(threshold)] += int((confidence < threshold).sum())
        if index % 1000 == 0:
            print(f"[dwpose audit] clips={index}/{len(paths)}", flush=True)

    frames = np.asarray(summary.pop("frame_counts"), dtype=np.int64)
    summary["frame_count"] = {
        "min": int(frames.min()), "median": float(np.median(frames)),
        "mean": float(frames.mean()), "max": int(frames.max()), "total": int(frames.sum()),
    }
    for name in GROUPS:
        confidence = np.concatenate(values[name]) if values[name] else np.empty(0, dtype=np.float32)
        summary["groups"][name] = {
            "points_per_frame": GROUPS[name].stop - GROUPS[name].start,
            "mean_confidence": float(confidence.mean()),
            "median_confidence": float(np.median(confidence)),
            "fraction_below": {
                threshold: low_counts[name][threshold] / max(1, total_counts[name])
                for threshold in low_counts[name]
            },
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_json)
    print(json.dumps({
        "clips": summary["clips"],
        "all_zero_clips": len(summary["all_zero_clips"]),
        "malformed_clips": len(summary["malformed_clips"]),
        "hand_confidence": {
            name: round(summary["groups"][name]["mean_confidence"], 4)
            for name in ("left_hand", "right_hand")
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
