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
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from slt.data import (AuslanDailyDataset, LengthBucketBatchSampler, Vocab,
                      collate, read_manifest)
from slt.engine import WarmupCosine, WarmupInverseSqrt, evaluate, resolve_device, train_one_epoch
from slt.metrics import bleu_signature, decision_gate
from slt.reference_model import build_model


def atomic_json(payload: dict, path: str):
    """Write progress without exposing a partially-written JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(payload: dict, path: str):
    """Checkpoint atomically so interruption leaves the previous epoch intact."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg: Config, vocab: Vocab, root: str = ""):
    train_ds = AuslanDailyDataset(cfg.train_manifest, vocab, cfg, root)
    val_ds = AuslanDailyDataset(cfg.val_manifest, vocab, cfg, root)
    if cfg.length_bucket_batches:
        batch_sampler = LengthBucketBatchSampler(
            train_ds, cfg.batch_size, cfg.max_src_frames, cfg.seed)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler,
                              num_workers=cfg.num_workers, collate_fn=collate)
    else:
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
    vocab = Vocab.build([t for _, t in train_rows], min_freq=cfg.min_freq,
                        tokenization=cfg.text_tokenization,
                        sentencepiece_model=cfg.sentencepiece_model)
    train_dl, val_dl = build_loaders(cfg, vocab, root)

    model = build_model(cfg.feature_dim(), len(vocab), cfg)
    if cfg.pretrained_target_embeddings:
        if cfg.model_arch != "gaslt":
            raise ValueError("pretrained_target_embeddings is currently GASLT-only")
        from slt.pretrained import initialize_target_embeddings
        loaded, total = initialize_target_embeddings(
            model, vocab, cfg.pretrained_target_embeddings)
        print(f"[target-init] loaded BPEmb vectors for {loaded}/{total} target tokens")
    model = model.to(device)
    loaded_encoder = False
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        ref = model.state_dict()
        encoder_prefixes = (
            # Legacy generic PoseSLT.
            "input_proj", "pos_enc", "transformer.encoder",
            # Source-compatible SignJoey/GASLT encoders.
            "pose_adapter", "spatial_embedding", "source_position",
            "encoder_layers", "encoder_norm",
        )
        enc = {k: v for k, v in sd.items()
               if k.startswith(encoder_prefixes)
               and k in ref and v.shape == ref[k].shape}
        model.load_state_dict(enc, strict=False)
        loaded_encoder = True
        print(f"[init] loaded {len(enc)} VLP-pretrained encoder tensors from {cfg.init_encoder}")
    if cfg.freeze_encoder:
        if not loaded_encoder:
            raise ValueError("--freeze_encoder true requires a valid --init_encoder checkpoint")
        frozen_prefixes = (
            "input_proj", "transformer.encoder", "pose_adapter",
            "spatial_embedding", "encoder_layers", "encoder_norm")
        for name, param in model.named_parameters():
            if name.startswith(frozen_prefixes):
                param.requires_grad_(False)
        print("[freeze] VLP pose encoder fixed; training decoder, token embeddings, and generator")
    trainable = [p for p in model.parameters() if p.requires_grad]
    # The public SignJoey configs use beta2=.998 and eps=1e-8.  Legacy runs keep
    # their original .98/1e-9 settings unless plateau/reference mode is selected.
    reference_optim = cfg.model_arch in ("signjoey_slt", "gaslt")
    betas = (0.9, 0.998) if reference_optim else (0.9, 0.98)
    eps = 1e-8 if reference_optim else 1e-9
    if cfg.encoder_lr_scale != 1.0:
        transfer_encoder_prefixes = (
            "input_proj", "pos_enc", "transformer.encoder", "pose_adapter",
            "spatial_embedding", "source_position", "encoder_layers", "encoder_norm")
        encoder_params, other_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (encoder_params if name.startswith(transfer_encoder_prefixes)
                      else other_params)
            target.append(parameter)
        param_groups = [
            {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_scale,
             "lr_scale": cfg.encoder_lr_scale},
            {"params": other_params, "lr": cfg.lr, "lr_scale": 1.0},
        ]
    else:
        param_groups = [{"params": trainable, "lr": cfg.lr, "lr_scale": 1.0}]
    opt = torch.optim.Adam(param_groups, lr=cfg.lr, betas=betas, eps=eps,
                           weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        updates_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
        total_steps = cfg.epochs * max(1, updates_per_epoch)
        sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, total_steps)
    elif cfg.lr_schedule == "noam":
        sched = WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)
    elif cfg.lr_schedule == "plateau":
        sched = opt
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, min_lr=cfg.lr_min)
    else:
        raise ValueError(f"unknown lr_schedule={cfg.lr_schedule!r}")

    semantic_bank = None
    if cfg.gaslt_similarity_weight > 0:
        if cfg.model_arch != "gaslt":
            raise ValueError("gaslt_similarity_weight > 0 requires --model_arch gaslt")
        from slt.semantic import SemanticSimilarityBank
        semantic_bank = SemanticSimilarityBank(
            [text for _, text in train_rows], cfg.semantic_model, cfg.semantic_cache)

    vlp_aux = None
    if cfg.vlp_aux_weight > 0:
        if not cfg.vlp_aux_checkpoint or not os.path.exists(cfg.vlp_aux_checkpoint):
            raise ValueError("vlp_aux_weight > 0 requires a valid --vlp_aux_checkpoint")
        from slt.vlp import FrozenVLPAuxiliary
        auxiliary_checkpoint = torch.load(
            cfg.vlp_aux_checkpoint, map_location="cpu", weights_only=False)
        vlp_aux = FrozenVLPAuxiliary(
            auxiliary_checkpoint, len(vocab), cfg).to(device)
        print(f"[vlp-aux] loaded frozen pose/text heads from "
              f"{cfg.vlp_aux_checkpoint} (weight={cfg.vlp_aux_weight:g})")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[setup] device={device} | arch={cfg.model_arch} | vocab={len(vocab)} "
          f"| params={n_params/1e6:.2f}M "
          f"trainable={n_trainable/1e6:.2f}M "
          f"| raw_feat_dim={cfg.feature_dim()} "
          f"| visual_feature_dim={cfg.visual_feature_dim or cfg.feature_dim()} "
          f"| batch={cfg.batch_size} accum={cfg.grad_accum_steps} "
          f"effective_batch={cfg.batch_size * cfg.grad_accum_steps} "
          f"length_buckets={cfg.length_bucket_batches} "
          f"encoder_lr_scale={cfg.encoder_lr_scale:g} "
          f"| train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    best = {"bleu4": -1.0}
    history = []
    start_epoch = 1
    if cfg.resume_checkpoint:
        resume = torch.load(cfg.resume_checkpoint, map_location=device,
                            weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        best = resume.get("best", best)
        history = resume.get("history", history)
        start_epoch = int(resume["epoch"]) + 1
        if cfg.lr_schedule == "plateau" and resume.get("plateau"):
            plateau.load_state_dict(resume["plateau"])
        elif resume.get("scheduler_step_num") is not None:
            sched.step_num = int(resume["scheduler_step_num"])
        print(f"[resume] {cfg.resume_checkpoint} -> epoch {start_epoch} "
              f"(best BLEU-4={best['bleu4']:.2f})")

    progress_path = os.path.join(cfg.out_dir, "progress.json")
    for ep in range(start_epoch, cfg.epochs + 1):
        loss = train_one_epoch(model, train_dl, sched, device,
                               cfg.label_smoothing, cfg.grad_clip,
                               cfg.signcl_weight, cfg.signcl_margin, cfg.signcl_zipf,
                               cfg.grad_accum_steps, semantic_bank,
                               cfg.gaslt_similarity_weight,
                               cfg.loss_normalization,
                               vlp_aux, cfg.vlp_aux_weight)
        decode_len = cfg.decode_max_len or cfg.max_tgt_tokens
        bleu4, wer, hyps, refs = evaluate(model, val_dl, vocab, device,
                                          decode_len, cfg.bleu_tokenize,
                                          cfg.detokenize)
        history.append({"epoch": ep, "loss": round(loss, 4),
                        "bleu4": round(bleu4, 3), "wer": round(wer, 4)})
        print(f"[ep {ep:03d}] loss={loss:.4f}  BLEU-4={bleu4:.2f}  WER={wer:.3f}")
        if cfg.lr_schedule == "plateau":
            plateau.step(bleu4)
        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": ep}
            atomic_torch_save({"model": model.state_dict(), "vocab": vocab.itos,
                               "cfg": cfg.to_dict()},
                              os.path.join(cfg.out_dir, "best.pt"))
        latest = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "vocab": vocab.itos, "cfg": cfg.to_dict(), "epoch": ep,
            "best": best, "history": history,
            "scheduler_step_num": getattr(sched, "step_num", None),
            "plateau": plateau.state_dict() if cfg.lr_schedule == "plateau" else None,
        }
        atomic_torch_save(latest, os.path.join(cfg.out_dir, "latest.pt"))
        atomic_json({
            "status": "running", "epoch": ep, "epochs": cfg.epochs,
            "latest": history[-1], "best": best,
            "device": str(device), "architecture": cfg.model_arch,
            "bbox_format": cfg.bbox_format,
        }, progress_path)

    # ── official test split: the reportable number ──────────────────────────
    # Model selection used val; the gate is applied to TEST when available so we
    # never report a number chosen on the same data it was selected on.
    test = None
    if cfg.test_manifest and os.path.exists(cfg.test_manifest):
        ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_dl = DataLoader(AuslanDailyDataset(cfg.test_manifest, vocab, cfg, root),
                             batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)
        decode_len = cfg.decode_max_len or cfg.max_tgt_tokens
        t_bleu, t_wer, _, _ = evaluate(model, test_dl, vocab, device,
                                       decode_len, cfg.bleu_tokenize,
                                       cfg.detokenize)
        test = {"bleu4": t_bleu, "wer": t_wer, "n": len(test_dl.dataset),
                "from_epoch": best.get("epoch")}
        print(f"[TEST] BLEU-4={t_bleu:.2f}  WER={t_wer:.3f}  (n={test['n']}, "
              f"ckpt from ep{best.get('epoch')})")

    # Reference-relative gate (Atif): threshold = gate_fraction x reference BLEU-4,
    # when a reference number is supplied; otherwise the fixed fallback.
    ref_bleu = float(cfg.reference_bleu4) if cfg.reference_bleu4 else None
    if ref_bleu:
        gate_thresh = cfg.gate_fraction * ref_bleu
        basis = (f"{cfg.gate_fraction:.0%} of reference BLEU-4 {ref_bleu} "
                 f"(same split)")
    else:
        gate_thresh = cfg.bleu4_gate
        basis = f"fixed placeholder {cfg.bleu4_gate}"
    gate_value = test["bleu4"] if test else best["bleu4"]
    gate = decision_gate(gate_value, gate_thresh, basis)
    gate["split"] = "test" if test else "val"
    result = {"best": best, "test": test, "gate": gate,
              "bleu_signature": bleu_signature(cfg.bleu_tokenize, cfg.detokenize),
              "history": history, "config": cfg.to_dict()}
    atomic_json(result, os.path.join(cfg.out_dir, "metrics.json"))
    atomic_json({
        "status": "complete", "epoch": cfg.epochs, "epochs": cfg.epochs,
        "best": best, "test": test, "gate": gate,
        "device": str(device), "architecture": cfg.model_arch,
        "bbox_format": cfg.bbox_format,
    }, progress_path)
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
