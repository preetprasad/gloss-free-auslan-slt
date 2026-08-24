"""Paired validation-only temporal-motion ablation for DWPose + frozen RGB.

The two arms differ only by ``--temporal_adapter``.  Both are direct
sign-to-English models with translation cross-entropy only: no CTC, pseudo
glosses, or official-test input.  This isolates whether feature-level temporal
motion modelling improves the existing DWPose+RGB representation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import PAD, Vocab
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device
from slt.rgb_fusion import RGBDataset, collate_rgb, evaluate_fusion
from slt.temporal_rgb_fusion import TemporalRGBFusionSLT


def parse() -> tuple[Config, argparse.Namespace]:
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in cfg.to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=value)
        elif value is None:
            parser.add_argument(f"--{name}", type=str, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    parser.add_argument("--temporal_adapter", choices=("none", "pyramid"), default="pyramid")
    parser.add_argument("--limit_samples", type=int, default=0)
    args = parser.parse_args()
    extra = {"temporal_adapter", "limit_samples"}
    return Config(**{key: value for key, value in vars(args).items() if key not in extra}), args


def read_texts(manifest: str) -> list[str]:
    with open(manifest, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[1] for line in handle if line.strip()]


def warm_start(model: TemporalRGBFusionSLT, cfg: Config) -> int:
    """Match the original RGB+CTC base warm start exactly."""
    if not (cfg.init_encoder and os.path.exists(cfg.init_encoder)):
        return 0
    checkpoint = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    reference = model.base.state_dict()
    keep = {key: value for key, value in state.items() if key in reference and value.shape == reference[key].shape}
    model.base.load_state_dict(keep, strict=False)
    print(f"[init] warm-started base with {len(keep)}/{len(reference)} tensors from {cfg.init_encoder}", flush=True)
    return len(keep)


def main() -> None:
    cfg, args = parse()
    if not cfg.rgb_feat_dir:
        raise ValueError("train_temporal_pyramid.py requires --rgb_feat_dir")
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    train_texts = read_texts(cfg.train_manifest)
    vocab = Vocab.build(train_texts, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)

    def make_loader(manifest: str, shuffle: bool, limit: int = 0) -> DataLoader:
        dataset = RGBDataset(manifest, vocab, cfg, "")
        if limit:
            dataset.samples = dataset.samples[:limit]
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate_rgb)

    train_loader = make_loader(cfg.train_manifest, True, args.limit_samples)
    val_loader = make_loader(cfg.val_manifest, False)
    model = TemporalRGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg,
                                 args.temporal_adapter).to(device)
    loaded = warm_start(model, cfg)
    params = list(model.parameters())
    optimiser = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = (WarmupCosine(optimiser, cfg.lr, cfg.warmup_steps, total_steps)
                 if cfg.lr_schedule == "cosine"
                 else WarmupInverseSqrt(optimiser, cfg.d_model, cfg.warmup_steps))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    max_length = cfg.decode_max_len or cfg.max_tgt_tokens
    parameter_count = sum(parameter.numel() for parameter in params)
    metadata = {
        "experiment": "translation_only_temporal_motion_ablation",
        "temporal_adapter": args.temporal_adapter,
        "warm_started_base_tensors": loaded,
        "pseudo_labels_used": False,
        "ctc_used": False,
        "test_set_used": False,
        "parameters": parameter_count,
    }
    with open(os.path.join(cfg.out_dir, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[temporal] arm={args.temporal_adapter} device={device} params={parameter_count / 1e6:.2f}M "
          f"warm_tensors={loaded} test_set_used=False", flush=True)

    best = {"bleu4": -1.0, "wer": float("inf"), "epoch": 0}
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0; item_count = 0
        for src, src_pad, rgb, rgb_pad, target, _texts in train_loader:
            src, src_pad = src.to(device), src_pad.to(device)
            rgb, rgb_pad, target = rgb.to(device), rgb_pad.to(device), target.to(device)
            target_in, target_out = target[:, :-1], target[:, 1:]
            logits = model(src, src_pad, rgb, rgb_pad, target_in, target_in.eq(PAD))
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            scheduler.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); scheduler.step()
            batch_size = src.size(0)
            total_loss += float(loss.detach()) * batch_size
            item_count += batch_size
        bleu4, wer, hypotheses, _ = evaluate_fusion(
            model, val_loader, vocab, device, max_len=max_length,
            bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize
        )
        record = {"epoch": epoch, "loss": total_loss / max(1, item_count), "bleu4": bleu4,
                  "wer": wer, "uniq": len(set(hypotheses))}
        history.append(record)
        print("[temporal {temporal_adapter} ep {epoch:03d}] loss={loss:.3f} val_bleu4={bleu4:.3f} "
              "wer={wer:.3f} uniq={uniq}".format(temporal_adapter=args.temporal_adapter, **record), flush=True)
        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": epoch}
            torch.save({"model": model.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict(),
                        "temporal_adapter": args.temporal_adapter, "experiment": metadata},
                       os.path.join(cfg.out_dir, "best.pt"))
        with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": "running", "epoch": epoch, "epochs": cfg.epochs, "best": best,
                       "history": history, **metadata}, handle, indent=2)

    with open(os.path.join(cfg.out_dir, "progress.json"), encoding="utf-8") as handle:
        progress = json.load(handle)
    progress["status"] = "done"
    with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)
    print(f"[temporal DONE] arm={args.temporal_adapter} best_val_bleu4={best['bleu4']:.3f} "
          f"epoch={best['epoch']} test_set_used=False", flush=True)


if __name__ == "__main__":
    main()
