"""Infer whether a pose manifest stores boxes as xywh or xyxy.

Scores each interpretation by the fraction of confident body+hand AlphaPose
points contained by its signer box.  Auslan-Daily should strongly favour xywh.
This is a read-only preflight check; it never changes the dataset.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.data import load_pose, read_manifest


BODY_HANDS = list(range(26)) + list(range(94, 136))


def containment(rows, limit: int, seed: int, confidence: float):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    scores = {"xywh": [], "xyxy": []}
    valid_boxes = {"xywh": 0, "xyxy": 0}
    frames = 0
    for path, _ in rows[:limit]:
        keypoints, boxes = load_pose(path)
        keypoints = np.asarray(keypoints)
        boxes = np.asarray(boxes).reshape(len(keypoints), -1)[:, :4]
        xy = keypoints[:, BODY_HANDS, :2]
        conf = keypoints[:, BODY_HANDS, 2]
        frames += len(boxes)
        for mode in scores:
            x1, y1 = boxes[:, 0, None], boxes[:, 1, None]
            if mode == "xywh":
                x2, y2 = x1 + boxes[:, 2, None], y1 + boxes[:, 3, None]
            else:
                x2, y2 = boxes[:, 2, None], boxes[:, 3, None]
            box_ok = (x2 > x1) & (y2 > y1)
            valid_boxes[mode] += int(box_ok.sum())
            selected = (conf > confidence) & box_ok
            inside = ((xy[..., 0] >= x1) & (xy[..., 0] <= x2)
                      & (xy[..., 1] >= y1) & (xy[..., 1] <= y2) & selected)
            if selected.sum():
                scores[mode].append(float(inside.sum() / selected.sum()))
    means = {mode: float(np.mean(values)) for mode, values in scores.items()}
    return means, valid_boxes, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--expect", choices=("xywh", "xyxy"))
    args = parser.parse_args()
    means, valid, frames = containment(read_manifest(args.manifest), args.limit,
                                       args.seed, args.confidence)
    winner = max(means, key=means.get)
    print(f"sampled_frames={frames}")
    for mode in ("xywh", "xyxy"):
        print(f"{mode}: confident_joint_containment={100 * means[mode]:.1f}% "
              f"valid_boxes={valid[mode]}/{frames}")
    print(f"inferred_bbox_format={winner}")
    if args.expect and args.expect != winner:
        raise SystemExit(f"ERROR: expected {args.expect}, but audit strongly favours {winner}")


if __name__ == "__main__":
    main()
