"""Face-ROI DINO feature extraction for a controlled Auslan-Daily feasibility gate.

The established RGB stream stores full-signer, left-hand and right-hand DINO
vectors.  This script adds *only* a face crop, localised from DWPose's
COCO-WholeBody face landmarks (23:91), so its crop-validity statistics can be
audited before a full multi-stream translation experiment is considered.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from transformers import AutoModel

os.environ.setdefault("OMP_NUM_THREADS", "4")

DINO = "facebook/dinov2-small"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FACE = slice(23, 91)  # COCO-WholeBody-133 DWPose layout


def crop_box(xy: np.ndarray, confidence: np.ndarray, threshold: float = 0.3,
             padding: float = 1.4) -> tuple[int, int, int, int] | None:
    keep = confidence > threshold
    if int(keep.sum()) < 3:
        return None
    points = xy[keep]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    radius = max(x1 - x0, y1 - y0) * padding / 2 + 1
    return (int(centre_x - radius), int(centre_y - radius),
            int(centre_x + radius), int(centre_y + radius))


def crop(frame: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray | None:
    if box is None:
        return None
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="per-shard clip cap; 0 means all")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="TSV manifest(s) whose pose paths define the clips to extract; repeat to combine splits",
    )
    args = parser.parse_args()
    rgb_dir = os.environ.get("RGB_DIR", "/scratch/pmc099/pprasad/dwpose/rgb_clips")
    pose_dir = os.environ.get("DWPOSE_DIR", "/scratch/pmc099/pprasad/dwpose/dwpose_clips")
    out_dir = os.environ.get("OUT_DIR", "/scratch/pmc099/pprasad/dwpose/face_feats")
    os.makedirs(out_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(rgb_dir, "**", "*.mp4"), recursive=True))
    # Restrict validation-only data preparation to the requested split(s), so an
    # official-test example is never staged merely by traversing the RGB folder.
    if args.manifest:
        requested: set[str] = set()
        for manifest in args.manifest:
            with open(manifest, "r", encoding="utf-8") as handle:
                for line in handle:
                    pose_path = line.rstrip("\n").split("\t", 1)[0]
                    requested.add(os.path.basename(pose_path).removesuffix(".npz"))
        videos = [
            video for video in videos
            if os.path.basename(video).removesuffix("_signer.mp4") in requested
        ]
    mine = [video for index, video in enumerate(videos) if index % args.nshards == args.shard]
    if args.limit:
        mine = mine[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    clips_written = frames_total = frames_valid = skipped = 0

    for video in mine:
        clip_id = os.path.basename(video).replace("_signer.mp4", "")
        output = os.path.join(out_dir, f"face_{clip_id}.npz")
        if os.path.exists(output):
            skipped += 1
            continue
        pose_path = os.path.join(pose_dir, f"{clip_id}.npz")
        if not os.path.exists(pose_path):
            skipped += 1
            continue
        keypoints = np.load(pose_path)["KPS"].astype(np.float32)
        capture = cv2.VideoCapture(video)
        video_frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            video_frames.append(frame)
        capture.release()
        length = min(len(video_frames), len(keypoints))
        if length < 1:
            skipped += 1
            continue
        indices = np.linspace(0, length - 1, min(args.frames, length)).round().astype(int)
        crops, valid = [], []
        for index in indices:
            frame = cv2.cvtColor(video_frames[index], cv2.COLOR_BGR2RGB)
            face = keypoints[index, FACE]
            current = crop(frame, crop_box(face[:, :2], face[:, 2]))
            valid.append(current is not None)
            crops.append(current if current is not None else np.zeros((224, 224, 3), np.uint8))
        tensor = torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        tensor = (tensor - mean) / std
        parts = []
        with torch.no_grad():
            for start in range(0, tensor.shape[0], 128):
                parts.append(model(pixel_values=tensor[start:start + 128]).pooler_output.cpu())
        features = torch.cat(parts).numpy().astype(np.float32)
        validity = np.asarray(valid, dtype=np.uint8)
        features[~validity.astype(bool)] = 0.0
        temporary = output + ".writing"
        np.savez_compressed(temporary, FEATURES=features, VALID=validity)
        os.replace(temporary + ".npz", output)
        clips_written += 1
        frames_total += len(validity)
        frames_valid += int(validity.sum())
        if clips_written % 20 == 0:
            print(f"[face shard {args.shard}] clips={clips_written} skipped={skipped} "
                  f"valid={frames_valid}/{frames_total}", flush=True)

    report = {
        "shard": args.shard, "clips_requested": len(mine), "clips_written": clips_written,
        "clips_skipped": skipped, "sampled_frames": frames_total, "valid_face_frames": frames_valid,
        "valid_face_fraction": frames_valid / max(1, frames_total), "feature_dim": 384,
        "dino": DINO, "device": device, "manifests": args.manifest,
    }
    with open(os.path.join(out_dir, f"face_extract_audit_shard{args.shard}.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[face shard DONE] " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
