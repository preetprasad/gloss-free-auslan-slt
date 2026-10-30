#!/usr/bin/env python3
"""Run one reproducible cross-architecture MMPose whole-body check.

This is intentionally a *probe*, not a pose-estimator benchmark.  It runs an
MMPose HRNet-W48 COCO-WholeBody model on one raw Auslan-Daily frame and compares
its 133 output joints to the already-extracted DWPose frame in the same image
coordinate system.  It avoids a detector confound by supplying the full frame as
the top-down person box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


GROUPS = {
    "body_and_feet": slice(0, 23),
    "face": slice(23, 91),
    "left_hand": slice(91, 112),
    "right_hand": slice(112, 133),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--dwpose_npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--min_confidence", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {frame_index} from {video}")
    return frame


def normalized_group_disagreement(
    reference_xy: np.ndarray,
    reference_conf: np.ndarray,
    candidate_xy: np.ndarray,
    candidate_conf: np.ndarray,
    confidence: float,
) -> dict:
    """Return same-image disagreement normalized by the reference group span."""
    joint_ok = (reference_conf >= confidence) & (candidate_conf >= confidence)
    if not joint_ok.any():
        return {"joint_count": 0, "median_pixels": None, "p90_pixels": None,
                "reference_span_pixels": None, "median_over_span": None}
    ref = reference_xy[joint_ok]
    cand = candidate_xy[joint_ok]
    errors = np.linalg.norm(ref - cand, axis=-1)
    # A per-group bounding-box diagonal has interpretable units for body, face and
    # hands, while avoiding an arbitrary fixed image-size normaliser.
    span = float(np.linalg.norm(ref.max(0) - ref.min(0)))
    return {
        "joint_count": int(joint_ok.sum()),
        "median_pixels": float(np.median(errors)),
        "p90_pixels": float(np.quantile(errors, 0.90)),
        "reference_span_pixels": span,
        "median_over_span": float(np.median(errors) / span) if span > 1e-6 else None,
    }


def main() -> None:
    args = parse_args()
    if args.frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    for path in (args.video, args.dwpose_npz, args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Import inside main so --help remains usable without the MMPose environment.
    import mmcv
    import mmpose
    from mmpose.apis import inference_topdown, init_model

    frame = read_frame(args.video, args.frame_index)
    height, width = frame.shape[:2]
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    # The DWPose crop was signer-only.  Giving MMPose the full frame removes a
    # detector choice from this first cross-architecture compatibility check.
    full_frame_bbox = np.asarray([[0.0, 0.0, float(width - 1), float(height - 1)]],
                                 dtype=np.float32)
    predictions = inference_topdown(
        model, frame, bboxes=full_frame_bbox, bbox_format="xyxy")
    if len(predictions) != 1:
        raise RuntimeError(f"expected one top-down prediction, got {len(predictions)}")
    instances = predictions[0].pred_instances
    keypoints = np.asarray(instances.keypoints, dtype=np.float32)
    scores = np.asarray(instances.keypoint_scores, dtype=np.float32)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]
    if keypoints.shape != (133, 2) or scores.shape != (133,):
        raise RuntimeError(
            f"expected COCO-WholeBody (133,2)/(133,), got {keypoints.shape}/{scores.shape}")

    with np.load(args.dwpose_npz) as payload:
        dwpose = np.asarray(payload["KPS"], dtype=np.float32)
    if dwpose.ndim != 3 or dwpose.shape[1:] != (133, 3):
        raise RuntimeError(f"expected DWPose (T,133,3), got {dwpose.shape}")
    if args.frame_index >= dwpose.shape[0]:
        raise IndexError(
            f"frame {args.frame_index} is outside DWPose length {dwpose.shape[0]}")
    dw = dwpose[args.frame_index]

    groups = {}
    for name, group in GROUPS.items():
        groups[name] = {
            "keypoints": group.stop - group.start,
            "dwpose_mean_confidence": float(dw[group, 2].mean()),
            "mmpose_mean_confidence": float(scores[group].mean()),
            "same_image_disagreement": normalized_group_disagreement(
                dw[group, :2], dw[group, 2], keypoints[group], scores[group],
                args.min_confidence),
        }

    summary = {
        "purpose": "one-frame cross-architecture compatibility probe; not a benchmark",
        "model": "MMPose HRNet-W48 COCO-WholeBody top-down heatmap",
        "layout": "COCO-WholeBody-133: body+feet 0:23, face 23:91, left hand 91:112, right hand 112:133",
        "source": {
            "video": str(args.video), "dwpose_npz": str(args.dwpose_npz),
            "frame_index": args.frame_index, "width": int(width), "height": int(height),
            "person_box": "full image (detector bypassed)",
        },
        "versions": {
            "mmpose": mmpose.__version__, "mmcv": mmcv.__version__,
            "numpy": np.__version__, "torch": torch.__version__,
        },
        "min_confidence": args.min_confidence,
        "groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_json)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
