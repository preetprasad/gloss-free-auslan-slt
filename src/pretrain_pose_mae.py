"""Pretrain an Auslan-specific multi-stream masked pose encoder."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from slt.data import LengthBucketBatchSampler, load_pose, normalize_clip, read_manifest
from slt.engine import WarmupCosine, resolve_device
from slt.pose_mae import MultiStreamPoseMAE


def atomic_json(payload, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(payload, path):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


class PoseOnlyDataset(Dataset):
    def __init__(self, manifest: str, cfg: Config, limit: int = 0):
        self.samples = read_manifest(manifest)
        if limit > 0:
            self.samples = self.samples[:limit]
        self.cfg = cfg
        self.root = ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        keypoints, bbox = load_pose(path)
        features, _ = normalize_clip(keypoints, bbox, self.cfg)
        return torch.from_numpy(features)


def pose_collate(batch):
    longest = max(item.size(0) for item in batch)
    width = batch[0].size(1)
    source = torch.zeros(len(batch), longest, width, dtype=torch.float32)
    source_pad = torch.ones(len(batch), longest, dtype=torch.bool)
    for row, item in enumerate(batch):
        source[row, :item.size(0)] = item
        source_pad[row, :item.size(0)] = False
    return source, source_pad


def parse():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Same-detector multi-stream masked pose pretraining")
    for name, value in defaults.to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=value)
        elif value is None:
            parser.add_argument(f"--{name}", type=str, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    parser.add_argument("--mae_hidden", type=int, default=256)
    parser.add_argument("--mae_encoder_layers", type=int, default=4)
    parser.add_argument("--mae_heads", type=int, default=8)
    parser.add_argument("--mae_ff", type=int, default=1024)
    parser.add_argument("--mae_decoder_hidden", type=int, default=256)
    parser.add_argument("--mae_decoder_layers", type=int, default=2)
    parser.add_argument("--mae_patch_size", type=int, default=4)
    parser.add_argument("--mae_mask_ratio", type=float, default=0.5)
    parser.add_argument("--mae_output_dim", type=int, default=1024)
    parser.add_argument("--mae_min_confidence", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    args = parser.parse_args()
    extra_names = {name for name in vars(args) if name.startswith("mae_")}
    extra_names |= {"max_train_samples", "max_val_samples"}
    cfg = Config(**{key: value for key, value in vars(args).items()
                    if key not in extra_names})
    extras = {key: value for key, value in vars(args).items() if key in extra_names}
    return cfg, extras


def main():
    cfg, extra = parse()
    if cfg.keypoint_subset != "body_hands" or not cfg.use_confidence:
        raise ValueError("pose MAE currently requires body_hands with confidence")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_ds = PoseOnlyDataset(
        cfg.train_manifest, cfg, int(extra["max_train_samples"]))
    val_ds = PoseOnlyDataset(
        cfg.val_manifest, cfg, int(extra["max_val_samples"]))
    if cfg.length_bucket_batches:
        sampler = LengthBucketBatchSampler(
            train_ds, cfg.batch_size, cfg.max_src_frames, cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, num_workers=cfg.num_workers,
            collate_fn=pose_collate)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=pose_collate)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=pose_collate)

    model = MultiStreamPoseMAE(
        hidden=extra["mae_hidden"], encoder_layers=extra["mae_encoder_layers"],
        heads=extra["mae_heads"], ff_size=extra["mae_ff"],
        decoder_hidden=extra["mae_decoder_hidden"],
        decoder_layers=extra["mae_decoder_layers"],
        patch_size=extra["mae_patch_size"], dropout=cfg.dropout,
        output_dim=extra["mae_output_dim"],
        min_confidence=extra["mae_min_confidence"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay)
    updates = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    scheduler = WarmupCosine(
        optimizer, cfg.lr, cfg.warmup_steps, cfg.epochs * max(1, updates))

    model_config = {
        "hidden": extra["mae_hidden"],
        "encoder_layers": extra["mae_encoder_layers"],
        "heads": extra["mae_heads"], "ff_size": extra["mae_ff"],
        "decoder_hidden": extra["mae_decoder_hidden"],
        "decoder_layers": extra["mae_decoder_layers"],
        "patch_size": extra["mae_patch_size"], "dropout": cfg.dropout,
        "output_dim": extra["mae_output_dim"],
        "min_confidence": extra["mae_min_confidence"],
    }
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[pose-mae setup] device={device} params={parameters/1e6:.2f}M "
        f"train={len(train_ds)} val={len(val_ds)} batch={cfg.batch_size} "
        f"accum={cfg.grad_accum_steps} effective_batch="
        f"{cfg.batch_size * cfg.grad_accum_steps} patch={extra['mae_patch_size']} "
        f"mask={extra['mae_mask_ratio']}", flush=True)

    start_epoch, best = 1, float("inf")
    history = []
    if cfg.resume_checkpoint:
        checkpoint = torch.load(
            cfg.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.step_num = int(checkpoint.get("scheduler_step_num", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_val_loss", best))
        history = list(checkpoint.get("history", []))
        print(f"[pose-mae resume] epoch={start_epoch} best_val={best:.6f}", flush=True)

    def validate(epoch):
        model.eval()
        total = count = 0
        region = {name: 0.0 for name in model.names}
        with torch.no_grad():
            for batch_index, (source, source_pad) in enumerate(val_loader):
                source, source_pad = source.to(device), source_pad.to(device)
                result = model(
                    source, source_pad, extra["mae_mask_ratio"],
                    mask_seed=cfg.seed + batch_index)
                size = source.size(0)
                total += float(result.loss) * size
                count += size
                for name in model.names:
                    region[name] += result.region_rmse[name] * size
        return total / max(1, count), {
            name: value / max(1, count) for name, value in region.items()}

    progress_path = output / "progress.json"
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        scheduler.zero_grad()
        total = count = 0
        for step, (source, source_pad) in enumerate(train_loader, start=1):
            source, source_pad = source.to(device), source_pad.to(device)
            result = model(source, source_pad, extra["mae_mask_ratio"])
            (result.loss / max(1, cfg.grad_accum_steps)).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scheduler.step()
                scheduler.zero_grad()
            size = source.size(0)
            total += float(result.loss.detach()) * size
            count += size
        train_loss = total / max(1, count)
        val_loss, region_rmse = validate(epoch)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": val_loss, "val_region_rmse": region_rmse}
        history.append(row)
        print(
            f"[pose-mae ep {epoch:03d}] train={train_loss:.6f} "
            f"val={val_loss:.6f} body={region_rmse['body']:.4f} "
            f"left={region_rmse['left_hand']:.4f} "
            f"right={region_rmse['right_hand']:.4f}", flush=True)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler_step_num": scheduler.step_num, "epoch": epoch,
            "best_val_loss": min(best, val_loss), "history": history,
            "cfg": cfg.to_dict(), "model_config": model_config,
            "research_provenance": {
                "method": "local scaled MS-MAE DiffMask adaptation",
                "training_data": "Auslan-Daily train split only (Communication + News)",
                "natural_missing_targets_ignored": True,
                "test_pose_exposure": False,
            },
        }
        atomic_torch_save(payload, output / "latest.pt")
        if val_loss < best:
            best = val_loss
            payload["best_val_loss"] = best
            atomic_torch_save(payload, output / "best.pt")
        atomic_json({
            "status": "running", "epoch": epoch, "epochs": cfg.epochs,
            "latest": row, "best_val_loss": best, "device": str(device),
            "train_samples": len(train_ds), "val_samples": len(val_ds),
            "mask_ratio": extra["mae_mask_ratio"], "patch_size": extra["mae_patch_size"],
        }, progress_path)

    atomic_json({
        "status": "complete", "epoch": cfg.epochs, "epochs": cfg.epochs,
        "best_val_loss": best, "device": str(device),
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "mask_ratio": extra["mae_mask_ratio"], "patch_size": extra["mae_patch_size"],
    }, progress_path)
    print(f"[pose-mae done] best validation loss={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
