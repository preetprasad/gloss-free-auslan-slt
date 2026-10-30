#!/usr/bin/env python3
"""Export temporal features from a pretrained Auslan pose MAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASELINE_DIR))

from config import Config  # noqa: E402
from slt.data import load_pose, normalize_clip, read_manifest  # noqa: E402
from slt.engine import resolve_device  # noqa: E402
from slt.pose_mae import MultiStreamPoseMAE  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PoseExtractionDataset(Dataset):
    def __init__(self, manifest: Path, cfg: Config, limit: int = 0):
        self.samples = read_manifest(str(manifest))
        if limit > 0:
            self.samples = self.samples[:limit]
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        source_path, target = self.samples[index]
        keypoints, bbox = load_pose(source_path)
        features, _ = normalize_clip(keypoints, bbox, self.cfg)
        return index, source_path, target, torch.from_numpy(features)


def collate(batch):
    longest = max(item[3].size(0) for item in batch)
    width = batch[0][3].size(1)
    source = torch.zeros(len(batch), longest, width, dtype=torch.float32)
    source_pad = torch.ones(len(batch), longest, dtype=torch.bool)
    records = []
    for row, (index, path, target, tensor) in enumerate(batch):
        source[row, : tensor.size(0)] = tensor
        source_pad[row, : tensor.size(0)] = False
        records.append((index, path, target))
    return source, source_pad, records


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for feature_path, target in rows:
            handle.write(f"{feature_path}\t{target}\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "cfg" not in checkpoint:
        raise ValueError("checkpoint lacks pose-MAE model/config provenance")
    cfg_values = checkpoint["cfg"]
    allowed = Config.__dataclass_fields__
    cfg = Config(**{key: value for key, value in cfg_values.items() if key in allowed})
    if cfg.keypoint_subset != "body_hands" or not cfg.use_confidence:
        raise ValueError("checkpoint input is not body_hands coordinates+confidence")

    model = MultiStreamPoseMAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()

    dataset = PoseExtractionDataset(args.manifest, cfg, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[tuple[str, str]] = []
    total_frames = 0
    with torch.inference_mode():
        for batch_number, (source, source_pad, records) in enumerate(loader, start=1):
            source = source.to(device)
            source_pad = source_pad.to(device)
            source_lengths = (~source_pad).sum(dim=1).cpu().tolist()
            features, feature_pad = model.extract_features(source, source_pad)
            features = features.float().cpu().numpy()
            feature_pad = feature_pad.cpu().numpy()
            for row, (index, source_path, target) in enumerate(records):
                length = int((~feature_pad[row]).sum())
                array = features[row, :length]
                if array.ndim != 2 or array.shape[1] != model.output_dim:
                    raise RuntimeError(f"unexpected feature shape {array.shape} for {source_path}")
                if not np.isfinite(array).all():
                    raise RuntimeError(f"non-finite feature values for {source_path}")
                destination = args.output_dir / f"{index:06d}_{Path(source_path).stem}.npz"
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        FEATURES=array.astype(np.float32),
                        SOURCE_PATH=np.asarray(source_path),
                        SOURCE_FRAMES=np.asarray(int(source_lengths[row])),
                        FEATURE_FRAMES=np.asarray(length),
                    )
                os.replace(temporary, destination)
                output_rows.append((str(destination.resolve()), target))
                total_frames += length
            if batch_number == 1 or batch_number % 100 == 0:
                print(
                    f"[pose-mae extract] batches={batch_number}/{len(loader)} "
                    f"clips={len(output_rows)}/{len(dataset)}",
                    flush=True,
                )

    write_manifest(args.output_manifest, output_rows)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_manifest": str(args.manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "clips": len(output_rows),
        "feature_frames": total_frames,
        "feature_dim": model.output_dim,
        "device": str(device),
        "model_config": checkpoint["model_config"],
        "input_config": cfg.to_dict(),
        "test_pose_exposure": False,
    }
    provenance_path = args.output_manifest.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        f"[pose-mae extract done] clips={len(output_rows)} "
        f"frames={total_frames} dim={model.output_dim} manifest={args.output_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
