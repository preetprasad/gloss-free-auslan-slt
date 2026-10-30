"""Multi-GPU (DistributedDataParallel) training entrypoint for Kaya.

Reuses the single-GPU model / data / engine unchanged; adds DDP, a
DistributedSampler, and rank-0-only logging + checkpointing. Data-parallel:
each GPU holds a replica and processes a shard of the batch, gradients are
all-reduced automatically on backward (NCCL on CUDA, Gloo otherwise).

Launch — single node, N GPUs (the common Kaya case):
    torchrun --standalone --nproc_per_node=N train_ddp.py \
        --train_manifest manifests/train.tsv --val_manifest manifests/val.tsv \
        --test_manifest manifests/test.tsv --batch_size 64 --lr 5e-4 \
        --lr_schedule cosine --keypoint_subset body_hands --use_velocity True \
        --epochs 60 --num_workers 8 --out_dir runs/ddp_comm

Multi-node: launch under srun with --nnodes / --node_rank / --rdzv_endpoint
(see jobs/slurm/ddp_train.sbatch).

If torchrun env vars are absent it runs as a single process (world_size=1), so
it can be smoke-tested on a laptop/CPU/MPS. Effective batch = batch_size x
world_size, so scale --lr with world size (roughly linear).

⚠ Untested on real multi-GPU here — do a 1-2 epoch, 2-GPU smoke run on Kaya
($MYSCRATCH data, NCCL) before committing to a full job.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, read_manifest
from slt.engine import WarmupCosine, WarmupInverseSqrt, evaluate, train_one_epoch
from slt.metrics import bleu_signature, decision_gate
from slt.reference_model import build_model
from train import parse_args  # identical CLI -> Config (safe import; no side effects)


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        mps = getattr(torch.backends, "mps", None)
        device = torch.device("mps") if (mps and mps.is_available()) else torch.device("cpu")
    return rank, world, local, distributed, device


def main():
    rank, world, local, distributed, device = setup_distributed()
    main_proc = rank == 0
    cfg, _root = parse_args()

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # Vocab is built from the train manifest -> deterministic + identical on every rank.
    train_rows = read_manifest(cfg.train_manifest)
    vocab = Vocab.build([t for _, t in train_rows], min_freq=cfg.min_freq,
                        tokenization=cfg.text_tokenization,
                        sentencepiece_model=cfg.sentencepiece_model)

    train_ds = AuslanDailyDataset(cfg.train_manifest, vocab, cfg)
    val_ds = AuslanDailyDataset(cfg.val_manifest, vocab, cfg)
    train_sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
                     if distributed else None)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                          shuffle=(train_sampler is None), num_workers=cfg.num_workers,
                          collate_fn=collate, pin_memory=torch.cuda.is_available(),
                          persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate)

    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local] if torch.cuda.is_available() else None)
    core = model.module if distributed else model  # unwrapped module for eval/save

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98),
                           eps=1e-9, weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, cfg.epochs * max(1, len(train_dl)))
    else:
        sched = WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)

    if main_proc:
        os.makedirs(cfg.out_dir, exist_ok=True)
        n_params = sum(p.numel() for p in core.parameters())
        print(f"[ddp] world={world} device={device} | vocab={len(vocab)} params={n_params/1e6:.2f}M "
              f"| bs/gpu={cfg.batch_size} eff_bs={cfg.batch_size*world} | feat_dim={cfg.feature_dim()}",
              flush=True)

    best = {"bleu4": -1.0}
    history = []
    for ep in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        loss = train_one_epoch(model, train_dl, sched, device, cfg.label_smoothing, cfg.grad_clip,
                               cfg.signcl_weight, cfg.signcl_margin, cfg.signcl_zipf)
        if main_proc:
            bleu4, wer, _, _ = evaluate(core, val_dl, vocab, device, cfg.max_tgt_tokens,
                                        cfg.bleu_tokenize, cfg.detokenize)
            history.append({"epoch": ep, "loss": round(loss, 4),
                            "bleu4": round(bleu4, 3), "wer": round(wer, 4)})
            print(f"[ep {ep:03d}] loss={loss:.4f}  BLEU-4={bleu4:.2f}  WER={wer:.3f}", flush=True)
            if bleu4 > best["bleu4"]:
                best = {"bleu4": bleu4, "wer": wer, "epoch": ep}
                torch.save({"model": core.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
                           os.path.join(cfg.out_dir, "best.pt"))
        if distributed:
            dist.barrier()  # keep ranks in lockstep while rank-0 evaluates/saves

    # Official test split + gate — rank 0 only.
    if main_proc:
        test = None
        if cfg.test_manifest and os.path.exists(cfg.test_manifest):
            ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device,
                              weights_only=False)
            core.load_state_dict(ckpt["model"])
            test_dl = DataLoader(AuslanDailyDataset(cfg.test_manifest, vocab, cfg),
                                 batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, collate_fn=collate)
            t_bleu, t_wer, _, _ = evaluate(core, test_dl, vocab, device, cfg.max_tgt_tokens,
                                           cfg.bleu_tokenize, cfg.detokenize)
            test = {"bleu4": t_bleu, "wer": t_wer, "n": len(test_dl.dataset),
                    "from_epoch": best.get("epoch")}
            print(f"[TEST] BLEU-4={t_bleu:.2f}  WER={t_wer:.3f}  (n={test['n']})", flush=True)
        ref = float(cfg.reference_bleu4) if cfg.reference_bleu4 else None
        thresh = cfg.gate_fraction * ref if ref else cfg.bleu4_gate
        basis = (f"{cfg.gate_fraction:.0%} of reference BLEU-4 {ref} (same split)"
                 if ref else f"fixed placeholder {cfg.bleu4_gate}")
        gate = decision_gate(test["bleu4"] if test else best["bleu4"], thresh, basis)
        gate["split"] = "test" if test else "val"
        result = {"best": best, "test": test, "gate": gate,
                  "bleu_signature": bleu_signature(cfg.bleu_tokenize, cfg.detokenize),
                  "history": history, "config": cfg.to_dict(), "world_size": world}
        with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"[gate] {gate['verdict']}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
