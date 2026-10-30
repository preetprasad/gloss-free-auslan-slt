"""VLP pretraining (Stage 1). Trains the pose encoder to align with text via a
CLIP-style contrastive loss on the Auslan-Daily train split, then saves a
PoseSLT-compatible checkpoint. Fine-tune with:  train.py --init_encoder <ckpt>

Usage:
    python pretrain_vlp.py --train_manifest manifests/train.tsv \
        --keypoint_subset body_hands --d_model 256 --batch_size 32 \
        --epochs 40 --lr 5e-4 --lr_schedule cosine --device auto \
        --out_dir runs/vlp
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
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device
from slt.vlp import VLPModel


def parse():
    cfg = Config()
    p = argparse.ArgumentParser(description="VLP pretraining (pose<->text contrastive)")
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--vlp_proj", type=int, default=256)
    p.add_argument("--vlp_temp", type=float, default=0.07)
    a = p.parse_args()
    extra = {"vlp_proj", "vlp_temp"}
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in extra})
    return cfg, a.vlp_proj, a.vlp_temp


def main():
    cfg, proj, temp = parse()
    device = resolve_device(cfg.device)
    # Seed every generator. Until 9 Oct 2026 this program set no seed, so --seed changed nothing here and
    # pretraining was not repeatable; the loader's shuffle and the model initialisation now follow cfg.seed.
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    print(f"[vlp-seed] seed={cfg.seed}", flush=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    rows = read_manifest(cfg.train_manifest)
    vocab = Vocab.build([t for _, t in rows], min_freq=cfg.min_freq,
                        tokenization=cfg.text_tokenization,
                        sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(cfg.train_manifest, vocab, cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, collate_fn=collate)
    val_rows = []
    val_dl = None
    if cfg.val_manifest and os.path.exists(cfg.val_manifest):
        val_rows = read_manifest(cfg.val_manifest)
        val_ds = AuslanDailyDataset(cfg.val_manifest, vocab, cfg)
        val_dl = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers, collate_fn=collate)

    model = VLPModel(cfg.feature_dim(), len(vocab), cfg, proj=proj, temp=temp).to(device)
    # Optional encoder warm-start (compound pretraining, e.g. ISLR -> VLP). Loads the
    # pose-encoder tensors of a prior checkpoint into VLPModel.slt before contrastive training.
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck); ref = model.slt.state_dict()
        pref = ("input_proj", "pos_enc", "transformer.encoder", "pose_adapter",
                "spatial_embedding", "source_position", "encoder_layers", "encoder_norm")
        enc = {k: v for k, v in sd.items() if k.startswith(pref) and k in ref and v.shape == ref[k].shape}
        model.slt.load_state_dict(enc, strict=False)
        print(f"[vlp init] warm-started encoder with {len(enc)} tensors from {cfg.init_encoder}", flush=True)
    reference_optim = cfg.model_arch in ("signjoey_slt", "gaslt")
    betas = (0.9, 0.998) if reference_optim else (0.9, 0.98)
    eps = 1e-8 if reference_optim else 1e-9
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=betas,
                           eps=eps, weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, cfg.epochs * max(1, len(dl)))
    else:
        sched = WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[vlp-setup] device={device} vocab={len(vocab)} params={n_params/1e6:.2f}M "
          f"feat_dim={cfg.feature_dim()} proj={proj} batch={cfg.batch_size} steps/ep={len(dl)}",
          flush=True)

    def atomic_torch_save(payload, path):
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def atomic_json(payload, path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)

    def save(tag):
        atomic_torch_save(
            {"model": model.slt.state_dict(), "cfg": cfg.to_dict(), "vlp_proj": proj},
            os.path.join(cfg.out_dir, f"vlp_encoder_{tag}.pt"))

    @torch.no_grad()
    def evaluate_retrieval():
        """Global held-out retrieval, with duplicate captions accepted at R@1."""
        if val_dl is None:
            return None
        model.eval()
        pose_embeddings, text_embeddings = [], []
        for src, src_pad, tgt, _ in val_dl:
            src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
            pose_embeddings.append(model.pose_embed(src, src_pad).cpu())
            text_embeddings.append(model.text(tgt).cpu())
        pose = torch.cat(pose_embeddings)
        text = torch.cat(text_embeddings)
        similarity = pose @ text.t()
        top1 = similarity.argmax(1)
        labels = torch.arange(similarity.size(0))
        strict_r1 = top1.eq(labels).float().mean().item()
        top5 = similarity.topk(min(5, similarity.size(1)), dim=1).indices
        strict_r5 = top5.eq(labels[:, None]).any(1).float().mean().item()
        normalized_text = [" ".join(text.lower().split()) for _, text in val_rows]
        caption_r1 = sum(
            normalized_text[int(pred)] == normalized_text[index]
            for index, pred in enumerate(top1)
        ) / max(1, len(normalized_text))
        return {
            "strict_r1": strict_r1,
            "strict_r5": strict_r5,
            "caption_aware_r1": caption_r1,
        }

    best = -1.0
    best_val_caption_r1 = -1.0
    start_epoch = 1
    if cfg.resume_checkpoint:
        resume = torch.load(cfg.resume_checkpoint, map_location=device,
                            weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.step_num = int(resume.get("scheduler_step_num", 0))
        # If a finished cosine run is extended, retain its original horizon.
        # Recomputing the curve for the larger epoch count would jump the LR
        # from its floor back toward the peak at the resume boundary.
        if cfg.lr_schedule == "cosine":
            original_epochs = int(resume.get("cfg", {}).get("epochs", cfg.epochs))
            sched.total = max(original_epochs * max(1, len(dl)), sched.warmup + 1)
        best = float(resume.get("best", best))
        best_val_caption_r1 = float(
            resume.get("best_val_caption_r1", best_val_caption_r1))
        start_epoch = int(resume["epoch"]) + 1
        print(f"[vlp-resume] {cfg.resume_checkpoint} -> epoch {start_epoch} "
              f"(best train retrieval={best:.3f})", flush=True)

        # Older checkpoints predate held-out retrieval. Establish their baseline
        # before the first additional update so model selection stays honest.
        if val_dl is not None and best_val_caption_r1 < 0:
            baseline_retrieval = evaluate_retrieval()
            best_val_caption_r1 = baseline_retrieval["caption_aware_r1"]
            save("best")
            print(
                "[vlp val baseline] "
                f"strict-R@1={baseline_retrieval['strict_r1']:.3f} "
                f"strict-R@5={baseline_retrieval['strict_r5']:.3f} "
                f"caption-R@1={baseline_retrieval['caption_aware_r1']:.3f}",
                flush=True)

    progress_path = os.path.join(cfg.out_dir, "progress.json")
    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        tl = ta = n = 0
        for src, src_pad, tgt, _ in dl:
            src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
            loss, acc = model(src, src_pad, tgt)
            sched.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            sched.step()
            b = src.size(0)
            tl += loss.item() * b; ta += acc.item() * b; n += b
        acc = ta / max(1, n)
        retrieval = evaluate_retrieval()
        validation_summary = ""
        if retrieval is not None:
            validation_summary = (
                f"  val_strict_R1={retrieval['strict_r1']:.3f}"
                f"  val_R5={retrieval['strict_r5']:.3f}"
                f"  val_caption_R1={retrieval['caption_aware_r1']:.3f}")
        print(f"[vlp ep {ep:03d}] loss={tl/max(1,n):.4f}  "
              f"train_retr_acc={acc:.3f}{validation_summary}", flush=True)
        best = max(best, acc)
        if retrieval is not None:
            if retrieval["caption_aware_r1"] > best_val_caption_r1:
                best_val_caption_r1 = retrieval["caption_aware_r1"]
                save("best")
        elif acc >= best:
            save("best")
        atomic_torch_save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": ep, "best": best, "scheduler_step_num": sched.step_num,
            "best_val_caption_r1": best_val_caption_r1,
            "cfg": cfg.to_dict(), "vlp_proj": proj, "vlp_temp": temp,
        }, os.path.join(cfg.out_dir, "vlp_latest.pt"))
        atomic_json({
            "status": "running", "epoch": ep, "epochs": cfg.epochs,
            "latest": {"loss": tl / max(1, n),
                       "training_in_batch_retrieval_accuracy": acc,
                       "held_out_retrieval": retrieval},
            "best_training_in_batch_retrieval_accuracy": best,
            "best_held_out_caption_r1": best_val_caption_r1,
            "device": str(device),
            "architecture": cfg.model_arch, "bbox_format": cfg.bbox_format,
        }, progress_path)
    save("final")
    atomic_json({
        "status": "complete", "epoch": cfg.epochs, "epochs": cfg.epochs,
        "best_training_in_batch_retrieval_accuracy": best,
        "best_held_out_caption_r1": best_val_caption_r1, "device": str(device),
        "architecture": cfg.model_arch, "bbox_format": cfg.bbox_format,
    }, progress_path)
    print(f"[vlp done] best train in-batch retrieval={best:.3f} "
          f"best held-out caption-R@1={best_val_caption_r1:.3f} "
          f"-> {cfg.out_dir}/vlp_encoder_best.pt", flush=True)


if __name__ == "__main__":
    main()
