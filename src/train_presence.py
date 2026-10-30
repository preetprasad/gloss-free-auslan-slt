"""Order-invariant pseudo-label supervision for the RGB+DWPose translator.

This experiment deliberately avoids CTC: English-derived pseudo-content labels
need not occur in Auslan temporal order.  A temporally max-pooled multi-label
head asks the *fused* visual representation only whether the video supports a
content token.  Translation stays direct sign-to-English at inference.

It is standalone so the historical CTC and RGB-fusion training paths remain
reproducible byte-for-byte in behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import PAD, Vocab
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device
from slt.pseudo_labels import (
    PSEUDO_LABEL_VERSION,
    build_label_vocab,
    label_statistics,
    multi_hot,
)
from slt.rgb_fusion import RGBDataset, RGBFusionSLT, collate_rgb, evaluate_fusion


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
    parser.add_argument("--presence_weight", type=float, default=0.30,
                        help="weight on order-invariant pseudo-label BCE")
    parser.add_argument("--presence_smooth_weight", type=float, default=0.01,
                        help="L1 temporal smoothness on pseudo-label probabilities")
    parser.add_argument("--limit_samples", type=int, default=0)
    args = parser.parse_args()
    extra = {"presence_weight", "presence_smooth_weight", "limit_samples"}
    return Config(**{k: v for k, v in vars(args).items() if k not in extra}), args


def read_texts(manifest: str) -> list[str]:
    with open(manifest, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[1] for line in handle if line.strip()]


def make_targets(texts: list[str], label_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    target = torch.zeros((len(texts), len(label_to_id)), dtype=torch.float32, device=device)
    for row, ids in enumerate(multi_hot(texts, label_to_id)):
        target[row, ids] = 1.0
    return target


def balanced_presence_bce(pooled_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Balance every clip's few positive pseudo labels against its negatives."""
    losses = F.binary_cross_entropy_with_logits(pooled_logits, targets, reduction="none")
    label_count = targets.sum(dim=1, keepdim=True).clamp_min(1.0)
    negative_count = (targets.size(1) - label_count).clamp_min(1.0)
    weights = torch.where(targets.bool(), negative_count / label_count, torch.ones_like(targets))
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def presence_loss(
    frame_logits: torch.Tensor, pad: torch.Tensor, targets: torch.Tensor, smooth_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set-level BCE plus valid-frame temporal smoothing, with no order loss."""
    valid = ~pad
    pooled = frame_logits.masked_fill(~valid.unsqueeze(-1), torch.finfo(frame_logits.dtype).min).amax(dim=1)
    set_loss = balanced_presence_bce(pooled, targets)
    probs = torch.sigmoid(frame_logits)
    if probs.size(1) < 2:
        smooth = probs.new_zeros(())
    else:
        pair_valid = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
        delta = (probs[:, 1:] - probs[:, :-1]).abs()
        smooth = (delta * pair_valid).sum() / pair_valid.sum().clamp_min(1)
    return set_loss + smooth_weight * smooth, set_loss.detach(), smooth.detach()


@torch.no_grad()
def set_diagnostics(
    model: RGBFusionSLT,
    head: nn.Module,
    loader: DataLoader,
    label_to_id: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    """Top-k set F1, where k is the known pseudo-label count for each sample."""
    model.eval(); head.eval()
    precision = recall = f1 = 0.0
    examples = 0
    for src, src_pad, rgb, rgb_pad, _tgt, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        rgb, rgb_pad = rgb.to(device), rgb_pad.to(device)
        mem_pose = model.base.encode(src, src_pad)
        mem_rgb = model.encode_rgb(rgb, rgb_pad)
        memory, pad = model._fuse(mem_pose, src_pad, mem_rgb, rgb_pad)
        scores = head(memory).masked_fill(pad.unsqueeze(-1), torch.finfo(memory.dtype).min).amax(dim=1)
        for row, true_ids in enumerate(multi_hot(texts, label_to_id)):
            k = max(1, len(true_ids))
            predicted = set(scores[row].topk(k).indices.tolist())
            truth = set(true_ids)
            overlap = len(predicted & truth)
            p = overlap / len(predicted)
            r = overlap / len(truth)
            precision += p; recall += r; f1 += (2 * p * r / (p + r)) if p + r else 0.0
            examples += 1
    divisor = max(1, examples)
    return {"set_precision_at_k": precision / divisor, "set_recall_at_k": recall / divisor,
            "set_f1_at_k": f1 / divisor}


def warm_start(model: RGBFusionSLT, cfg: Config) -> None:
    """Load only compatible pose/decoder parameters, matching the original RGB run."""
    if not (cfg.init_encoder and os.path.exists(cfg.init_encoder)):
        return
    checkpoint = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    reference = model.base.state_dict()
    keep = {key: value for key, value in state.items() if key in reference and value.shape == reference[key].shape}
    model.base.load_state_dict(keep, strict=False)
    print(f"[init] warm-started base with {len(keep)}/{len(reference)} tensors from {cfg.init_encoder}", flush=True)


def main() -> None:
    cfg, args = parse()
    if not cfg.rgb_feat_dir:
        raise ValueError("train_presence.py requires --rgb_feat_dir; this is the fused DWPose+RGB experiment")
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    train_texts = read_texts(cfg.train_manifest)
    vocab = Vocab.build(train_texts, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    label_to_id = build_label_vocab(train_texts)
    print(f"[presence] pseudo_labels={PSEUDO_LABEL_VERSION} vocab={len(vocab)} label_vocab={len(label_to_id)} "
          f"device={device} pose_feat={cfg.feature_dim()} rgb_feat={cfg.rgb_feature_dim}", flush=True)

    def make_loader(manifest: str, shuffle: bool, limit: int = 0) -> DataLoader:
        dataset = RGBDataset(manifest, vocab, cfg, "")
        if limit:
            dataset.samples = dataset.samples[:limit]
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate_rgb)

    train_loader = make_loader(cfg.train_manifest, True, args.limit_samples)
    val_loader = make_loader(cfg.val_manifest, False)
    model = RGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg).to(device)
    warm_start(model, cfg)
    head = nn.Linear(cfg.d_model, len(label_to_id)).to(device)
    parameters = list(model.parameters()) + list(head.parameters())
    optimiser = torch.optim.Adam(parameters, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = (WarmupCosine(optimiser, cfg.lr, cfg.warmup_steps, total_steps)
                 if cfg.lr_schedule == "cosine"
                 else WarmupInverseSqrt(optimiser, cfg.d_model, cfg.warmup_steps))
    translation_loss = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    max_length = cfg.decode_max_len or cfg.max_tgt_tokens
    audit = {"train": label_statistics(train_texts), "presence_weight": args.presence_weight,
             "presence_smooth_weight": args.presence_smooth_weight}
    with open(os.path.join(cfg.out_dir, "pseudo_label_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    best = {"bleu4": -1.0, "wer": float("inf"), "epoch": 0}
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train(); head.train()
        totals = {"loss": 0.0, "translation": 0.0, "presence": 0.0, "smooth": 0.0, "items": 0}
        for src, src_pad, rgb, rgb_pad, target, texts in train_loader:
            src, src_pad = src.to(device), src_pad.to(device)
            rgb, rgb_pad, target = rgb.to(device), rgb_pad.to(device), target.to(device)
            target_in, target_out = target[:, :-1], target[:, 1:]
            target_pad = target_in.eq(PAD)
            pose_memory = model.base.encode(src, src_pad)
            rgb_memory = model.encode_rgb(rgb, rgb_pad)
            fused_memory, fused_pad = model._fuse(pose_memory, src_pad, rgb_memory, rgb_pad)
            logits = model.base.decode_step(target_in, fused_memory, fused_pad, target_pad)
            loss_translation = translation_loss(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            loss_presence, raw_presence, smooth = presence_loss(
                head(fused_memory), fused_pad, make_targets(texts, label_to_id, device), args.presence_smooth_weight
            )
            loss = loss_translation + args.presence_weight * loss_presence
            scheduler.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip); scheduler.step()
            batch = src.size(0)
            totals["loss"] += float(loss.detach()) * batch
            totals["translation"] += float(loss_translation.detach()) * batch
            totals["presence"] += float(raw_presence) * batch
            totals["smooth"] += float(smooth) * batch
            totals["items"] += batch

        bleu4, wer, hypotheses, _ = evaluate_fusion(
            model, val_loader, vocab, device, max_len=max_length,
            bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize
        )
        diagnostics = set_diagnostics(model, head, val_loader, label_to_id, device)
        count = max(1, totals["items"])
        record = {
            "epoch": epoch, "bleu4": bleu4, "wer": wer, "uniq": len(set(hypotheses)),
            "loss": totals["loss"] / count, "translation_loss": totals["translation"] / count,
            "presence_loss": totals["presence"] / count, "smooth": totals["smooth"] / count,
            **diagnostics,
        }
        history.append(record)
        print("[presence ep {epoch:03d}] loss={loss:.3f} ce={translation_loss:.3f} "
              "set={presence_loss:.3f} val_bleu4={bleu4:.3f} wer={wer:.3f} uniq={uniq} "
              "set_f1={set_f1_at_k:.3f}".format(**record), flush=True)
        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": epoch, **diagnostics}
            torch.save({"model": model.state_dict(), "presence_head": head.state_dict(),
                        "vocab": vocab.itos, "cfg": cfg.to_dict(), "pseudo_label_version": PSEUDO_LABEL_VERSION,
                        "label_to_id": label_to_id}, os.path.join(cfg.out_dir, "best.pt"))
        with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": "running", "epoch": epoch, "epochs": cfg.epochs,
                       "best": best, "history": history, "test_set_used": False}, handle, indent=2)

    with open(os.path.join(cfg.out_dir, "progress.json"), encoding="utf-8") as handle:
        progress = json.load(handle)
    progress["status"] = "done"
    with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)
    print(f"[presence DONE] best_val_bleu4={best['bleu4']:.3f} epoch={best['epoch']} test_set_used=False", flush=True)


if __name__ == "__main__":
    main()
