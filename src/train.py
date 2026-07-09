"""Train the gloss-free Auslan-Daily SLT baseline and evaluate the BLEU-4 gate.

Usage (on Colab/Kaya, once manifests point at the real .pt clips):
    python train.py --train_manifest manifests/train.tsv \
                    --val_manifest   manifests/val.tsv \
                    --epochs 40 --out_dir runs/baseline

A manifest is a TSV of `clip_path<TAB>english_translation`. Build it from the
Auslan-Daily split tables + translation .xlsx (see tools/build_manifest.py stub
in the README) so this trainer stays decoupled from the Drive layout.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, read_manifest
from slt.engine import WarmupInverseSqrt, evaluate, resolve_device, train_one_epoch
from slt.metrics import decision_gate
from slt.model import PoseSLT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg: Config, vocab: Vocab, root: str = ""):
    train_ds = AuslanDailyDataset(cfg.train_manifest, vocab, cfg, root)
    val_ds = AuslanDailyDataset(cfg.val_manifest, vocab, cfg, root)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate)
    return train_dl, val_dl


def run(cfg: Config, root: str = "") -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_rows = read_manifest(cfg.train_manifest)
    vocab = Vocab.build([t for _, t in train_rows], min_freq=cfg.min_freq)
    train_dl, val_dl = build_loaders(cfg, vocab, root)

    model = PoseSLT(cfg.feature_dim(), len(vocab), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay)
    sched = WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] device={device} | vocab={len(vocab)} | params={n_params/1e6:.2f}M "
          f"| feat_dim={cfg.feature_dim()} | train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    best = {"bleu4": -1.0}
    history = []
    for ep in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_dl, sched, device,
                               cfg.label_smoothing, cfg.grad_clip)
        bleu4, wer, hyps, refs = evaluate(model, val_dl, vocab, device, cfg.max_tgt_tokens)
        history.append({"epoch": ep, "loss": round(loss, 4),
                        "bleu4": round(bleu4, 3), "wer": round(wer, 4)})
        print(f"[ep {ep:03d}] loss={loss:.4f}  BLEU-4={bleu4:.2f}  WER={wer:.3f}")
        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": ep}
            torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                        "cfg": cfg.to_dict()}, os.path.join(cfg.out_dir, "best.pt"))

    gate = decision_gate(best["bleu4"], cfg.bleu4_gate)
    result = {"best": best, "gate": gate, "history": history, "config": cfg.to_dict()}
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gate] {gate['verdict']}")
    return result


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Gloss-free Auslan-Daily SLT baseline")
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--root", type=str, default="", help="prefix dir for clip paths")
    a = p.parse_args()
    root = a.root
    kwargs = {k: v for k, v in vars(a).items() if k != "root"}
    return Config(**kwargs), root


if __name__ == "__main__":
    cfg, root = parse_args()
    run(cfg, root)
