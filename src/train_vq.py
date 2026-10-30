"""Stage A: Train VQ tokeniser over pose/RGB features.

Usage (Kaya/Colab):
    python train_vq.py \
        --train_manifest /scratch/pmc099/pprasad/dwpose/manifests/train.tsv \
        --val_manifest   /scratch/pmc099/pprasad/dwpose/manifests/val.tsv \
        --vq_modality pose \
        --vq_codebook_size 1024 --vq_code_dim 256 \
        --vq_num_codebooks 1 --vq_ema_decay 0.99 \
        --vq_commitment_cost 0.25 \
        --batch_size 16 --grad_accum_steps 1 \
        --epochs 50 --out_dir runs/vq_pose
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from slt.data import (AuslanDailyDataset, LengthBucketBatchSampler,
                      collate, read_manifest, load_pose, normalize_clip)
from slt.engine import WarmupCosine, resolve_device
from slt.sign_vq import VQTokeniser, build_vq_tokeniser
from slt.metrics import bleu_signature, decision_gate


W, H = 1920, 1080
SUBJECTS = ["the man", "the woman", "the boy", "the girl", "the teacher"]
VERBS = ["sees", "helps", "greets", "teaches", "calls"]
OBJECTS = ["the child", "the doctor", "a friend", "the family", "the class"]


def make_clip(seed: int):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(24, 80))
    x1 = rng.uniform(200, 600); y1 = rng.uniform(100, 300)
    bw = rng.uniform(500, 800); bh = rng.uniform(600, 760)
    bbox = np.zeros((T, 4), np.float32)
    kps = np.zeros((T, 136, 3), np.float32)
    for t in range(T):
        cx = x1 + 5 * np.sin(t / 5.0)
        cy = y1 + 5 * np.cos(t / 5.0)
        bbox[t] = [cx, cy, bw, bh]
        if t == 0:
            continue
        pts = rng.uniform(0, 1, size=(136, 2)) * [bw, bh] + [cx, cy]
        conf = rng.uniform(0.3, 1.0, size=(136, 1))
        kps[t, :, :2] = pts
        kps[t, :, 2:] = conf
    return kps, bbox


def sentence(seed: int) -> str:
    r = random.Random(seed)
    return f"{r.choice(SUBJECTS)} {r.choice(VERBS)} {r.choice(OBJECTS)} ."


def ensure_synthetic_data(cfg: Config, limit_samples: int = 0):
    train_path = cfg.train_manifest
    val_path = cfg.val_manifest
    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path

    tmp = tempfile.mkdtemp(prefix="vq_smoke_")
    clips = os.path.join(tmp, "clips")
    man = os.path.join(tmp, "manifests")
    os.makedirs(clips, exist_ok=True)
    os.makedirs(man, exist_ok=True)

    n_train = limit_samples if limit_samples else 48
    n_val = limit_samples if limit_samples else 16

    def write_split(name: str, n: int, base: int):
        rows = []
        for i in range(n):
            s = base + i
            kps, bbox = make_clip(s)
            path = os.path.join(clips, f"{name}_{i:04d}.pt")
            torch.save({"KPS": torch.from_numpy(kps), "BBOX": torch.from_numpy(bbox)}, path)
            rows.append(f"{path}\t{sentence(s)}")
        out_path = os.path.join(man, f"{name}.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        return out_path

    train_m = write_split("train", n_train, 1000)
    val_m = write_split("val", n_val, 5000)
    return train_m, val_m


class VQDataset(torch.utils.data.Dataset):
    """Dataset for VQ tokeniser training: loads pose/RGB features."""

    def __init__(self, manifest: str, cfg: Config, root: str = "", limit_samples: int = 0):
        self.samples = read_manifest(manifest)
        if limit_samples:
            self.samples = self.samples[:limit_samples]
        self.cfg = cfg
        self.root = root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        clip, _text = self.samples[i]
        path = os.path.join(self.root, clip) if self.root else clip

        if self.cfg.vq_modality == "pose":
            kps, bbox = load_pose(path)
            feats, _ = normalize_clip(kps, bbox, self.cfg)
        elif self.cfg.vq_modality == "rgb":
            feats = np.load(path)["FEATURES"].astype(np.float32)
            if feats.shape[0] > self.cfg.max_src_frames:
                sel = np.linspace(0, feats.shape[0] - 1, self.cfg.max_src_frames).round().astype(int)
                feats = feats[sel]
        elif self.cfg.vq_modality == "pose_rgb":
            # Load both and concatenate
            kps, bbox = load_pose(path)
            pose_feats, _ = normalize_clip(kps, bbox, self.cfg)
            rgb_path = path.replace("dwpose_clips", "rgb_feats").replace(".npz", "_rgb.npz")
            if os.path.exists(rgb_path):
                rgb_feats = np.load(rgb_path)["FEATURES"].astype(np.float32)
            else:
                rgb_feats = np.zeros((pose_feats.shape[0], 1152), dtype=np.float32)
            min_t = min(pose_feats.shape[0], rgb_feats.shape[0])
            feats = np.concatenate([pose_feats[:min_t], rgb_feats[:min_t]], axis=-1)
        else:
            raise ValueError(f"Unknown vq_modality: {self.cfg.vq_modality}")

        return torch.from_numpy(feats), clip


def collate_vq(batch, pad_id: int = 0):
    feats, clips = zip(*batch)
    B = len(feats)
    Tmax = max(f.shape[0] for f in feats)
    Fdim = feats[0].shape[1]
    src = torch.zeros(B, Tmax, Fdim, dtype=torch.float32)
    src_pad = torch.ones(B, Tmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        t = f.shape[0]
        src[i, :t] = f
        src_pad[i, :t] = False
    return src, src_pad, list(clips)


def atomic_json(payload: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(payload: dict, path: str):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg: Config, root: str = "", limit_samples: int = 0) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_m, val_m = ensure_synthetic_data(cfg, limit_samples)

    train_ds = VQDataset(train_m, cfg, root, limit_samples)
    val_ds = VQDataset(val_m, cfg, root, limit_samples)

    if cfg.length_bucket_batches:
        batch_sampler = LengthBucketBatchSampler(
            train_ds, cfg.batch_size, cfg.max_src_frames, cfg.seed)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler,
                              num_workers=cfg.num_workers, collate_fn=collate_vq)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_vq)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_vq)

    model = build_vq_tokeniser(cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] device={device} | vq_modality={cfg.vq_modality} "
          f"| params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
          f"| codebook={cfg.vq_codebook_size}x{cfg.vq_code_dim} K={cfg.vq_num_codebooks} "
          f"| batch={cfg.batch_size} accum={cfg.grad_accum_steps} "
          f"eff_batch={cfg.batch_size * cfg.grad_accum_steps} "
          f"| train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    updates_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
    total_steps = cfg.epochs * max(1, updates_per_epoch)
    sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, total_steps)

    best = {"recon_loss": float("inf")}
    history = []
    start_epoch = 1

    if cfg.resume_checkpoint:
        resume = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        best = resume.get("best", best)
        history = resume.get("history", history)
        start_epoch = int(resume["epoch"]) + 1
        if hasattr(sched, "step_num") and resume.get("scheduler_step_num") is not None:
            sched.step_num = int(resume["scheduler_step_num"])
        print(f"[resume] {cfg.resume_checkpoint} -> epoch {start_epoch}")

    progress_path = os.path.join(cfg.out_dir, "progress.json")

    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        total_vq, total_recon, total_perp = 0.0, 0.0, 0.0
        sched.zero_grad()

        for step, (src, src_pad, _clips) in enumerate(train_dl, start=1):
            src, src_pad = src.to(device), src_pad.to(device)

            out = model(src, src_pad, return_reconstruction=True)
            vq_loss = out["vq_loss"]
            recon_loss = out["recon_loss"]
            loss = vq_loss + cfg.vq_reconstruction_weight * recon_loss

            (loss / cfg.grad_accum_steps).backward()
            update_now = step % cfg.grad_accum_steps == 0 or step == len(train_dl)
            if update_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                sched.step()
                sched.zero_grad()

            total += loss.item() * src.size(0)
            total_vq += vq_loss.item() * src.size(0)
            total_recon += recon_loss.item() * src.size(0)
            total_perp += out["perplexity"].item() * src.size(0)
            n += src.size(0)

        train_loss = total / max(1, n)
        avg_vq = total_vq / max(1, n)
        avg_recon = total_recon / max(1, n)
        avg_perp = total_perp / max(1, n)

        # Validation
        model.eval()
        val_vq, val_recon, val_perp = 0.0, 0.0, 0.0
        val_n = 0
        with torch.no_grad():
            for src, src_pad, _clips in val_dl:
                src, src_pad = src.to(device), src_pad.to(device)
                out = model(src, src_pad, return_reconstruction=True)
                val_vq += out["vq_loss"].item() * src.size(0)
                val_recon += out["recon_loss"].item() * src.size(0)
                val_perp += out["perplexity"].item() * src.size(0)
                val_n += src.size(0)

        val_vq /= max(1, val_n)
        val_recon /= max(1, val_n)
        val_perp /= max(1, val_n)

        history.append({
            "epoch": ep, "loss": round(train_loss, 4),
            "vq_loss": round(avg_vq, 4), "recon_loss": round(avg_recon, 4),
            "perplexity": round(avg_perp, 2),
            "val_vq_loss": round(val_vq, 4), "val_recon_loss": round(val_recon, 4),
            "val_perplexity": round(val_perp, 2),
        })
        print(f"[ep {ep:03d}] loss={train_loss:.4f} (vq={avg_vq:.4f} recon={avg_recon:.4f}) "
              f"perp={avg_perp:.1f} | val_vq={val_vq:.4f} val_recon={val_recon:.4f} val_perp={val_perp:.1f}")

        # Save best by validation reconstruction loss
        if val_recon < best["recon_loss"]:
            best = {"recon_loss": val_recon, "vq_loss": val_vq, "perplexity": val_perp, "epoch": ep}
            atomic_torch_save({
                "model": model.state_dict(),
                "cfg": cfg.to_dict(),
            }, os.path.join(cfg.out_dir, "best.pt"))

        latest = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": cfg.to_dict(),
            "epoch": ep,
            "best": best,
            "history": history,
            "scheduler_step_num": getattr(sched, "step_num", None),
        }
        atomic_torch_save(latest, os.path.join(cfg.out_dir, "latest.pt"))

        atomic_json({
            "status": "running", "epoch": ep, "epochs": cfg.epochs,
            "latest": history[-1], "best": best,
            "device": str(device), "architecture": f"vq_tokeniser_{cfg.vq_modality}",
        }, progress_path)

    # Final evaluation on val
    result = {"best": best, "history": history, "config": cfg.to_dict()}
    atomic_json(result, os.path.join(cfg.out_dir, "metrics.json"))

    atomic_json({
        "status": "complete", "epoch": cfg.epochs, "epochs": cfg.epochs,
        "best": best, "device": str(device),
        "architecture": f"vq_tokeniser_{cfg.vq_modality}",
    }, progress_path)

    print(f"[done] Best val recon_loss: {best['recon_loss']:.4f} at epoch {best['epoch']}")
    return result


def parse_args() -> tuple[Config, str, int]:
    cfg = Config()
    p = argparse.ArgumentParser(description="VQ tokeniser for sign features")
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--root", type=str, default="", help="prefix dir for clip paths")
    p.add_argument("--limit_samples", type=int, default=0, help="limit samples for smoke test (0=all)")
    a = p.parse_args()
    root = a.root
    limit_samples = a.limit_samples
    kwargs = {k: v for k, v in vars(a).items() if k not in ("root", "limit_samples")}
    return Config(**kwargs), root, limit_samples


if __name__ == "__main__":
    cfg, root, limit_samples = parse_args()
    run(cfg, root, limit_samples)