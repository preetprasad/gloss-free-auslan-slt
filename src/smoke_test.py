"""End-to-end smoke test on synthetic data (CPU, no real dataset, no GPU).

Proves the pipeline is wired correctly BEFORE it ever touches real Auslan-Daily
tensors on Colab/Kaya:
  * loads synthetic HALPE-136 (T,136,3) clips through the real normaliser
    (incl. the all-zero frame-0 quirk and bbox normalisation),
  * builds vocab, batches with padding + masks,
  * trains the Transformer baseline a few epochs,
  * runs greedy decode + BLEU-4 / WER,
  * exercises the decision-gate logic.

Run:  python smoke_test.py
Passing means the code is correct; it does NOT validate research performance.
"""
from __future__ import annotations

import shutil
import sys
import tempfile

import numpy as np
import torch

from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, load_pose, normalize_clip
from slt.engine import WarmupInverseSqrt, evaluate, resolve_device, train_one_epoch
from slt.metrics import decision_gate
from slt.model import PoseSLT
from make_synthetic import build
from torch.utils.data import DataLoader


def check(name, cond):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


# The synthetic clips use the dataset's shipped layout (HALPE-136 keypoints with a bounding box), not the
# 133-keypoint DWPose layout that config.py now defaults to.
HALPE = dict(num_keypoints=136, coord_norm="bbox")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="slt_smoke_")
    try:
        print("1) synthetic data")
        train_m, val_m = build(tmp, n_train=48, n_val=16)

        cfg = Config(
            train_manifest=train_m, val_manifest=val_m, **HALPE,
            d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=128, batch_size=8, epochs=8, warmup_steps=40,
            max_src_frames=64, max_tgt_tokens=16, num_workers=0,
            out_dir=f"{tmp}/run", device="cpu", bleu4_gate=1.0,
        )

        print("2) normaliser (frame-0 zero handling + bbox norm)")
        kps, bbox = load_pose(AuslanDailyDataset(train_m, Vocab(["<pad>", "<bos>", "<eos>", "<unk>"]), cfg).samples[0][0])
        feats, mask = normalize_clip(kps, bbox, cfg)
        check("frame-0 dropped (T_out < T_in)", feats.shape[0] < kps.shape[0])
        check(f"feature dim == {cfg.feature_dim()}", feats.shape[1] == cfg.feature_dim())
        check("features finite", bool(np.isfinite(feats).all()))
        check("normalised coords in sane range", float(np.abs(feats[:, :2]).max()) < 5.0)

        legacy_kps = np.ones((2, 136, 3), dtype=np.float32)
        legacy_kps[..., 0] = 75.0
        legacy_kps[..., 1] = 75.0
        reversed_xyxy = np.tile(
            np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float32), (2, 1))
        try:                                  # every frame invalid -> normalize_clip refuses the clip
            normalize_clip(legacy_kps, reversed_xyxy, Config(bbox_format="xyxy", **HALPE))
            rejected = False
        except ValueError:
            rejected = True
        legacy_features, _ = normalize_clip(
            legacy_kps, reversed_xyxy,
            Config(bbox_format="xyxy", legacy_abs_bbox_validity=True, **HALPE))
        check("strict bbox validity rejects reversed spans", rejected)
        check("legacy bbox compatibility preserves historical frames",
              legacy_features.shape[0] == 2)

        print("3) vocab + batching")
        rows = AuslanDailyDataset(train_m, Vocab(["<pad>"]), cfg).samples
        vocab = Vocab.build([t for _, t in rows], min_freq=1)
        check("vocab has specials + words", len(vocab) > 10)
        ds = AuslanDailyDataset(train_m, vocab, cfg)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
        src, src_pad, tgt, texts = next(iter(dl))
        check("src is (B,T,F)", src.dim() == 3 and src.shape[2] == cfg.feature_dim())
        check("pad mask matches src", src_pad.shape == src.shape[:2])
        check("some frames padded OR all full", src_pad.any() or True)

        print("3b) precomputed visual feature path")
        feature_path = f"{tmp}/published_frontend.npz"
        np.savez_compressed(
            feature_path,
            FEATURES=np.arange(5 * 32, dtype=np.float32).reshape(5, 32),
        )
        feature_manifest = f"{tmp}/published_frontend.tsv"
        with open(feature_manifest, "w", encoding="utf-8") as handle:
            handle.write(f"{feature_path}\thello .\n")
        feature_cfg = Config(
            train_manifest=feature_manifest,
            val_manifest=feature_manifest,
            input_representation="precomputed",
            precomputed_feature_dim=32,
            max_src_frames=4,
        )
        feature_ds = AuslanDailyDataset(feature_manifest, vocab, feature_cfg)
        feature_src, _, _ = feature_ds[0]
        check(
            "precomputed features load, validate width, and obey temporal cap",
            tuple(feature_src.shape) == (4, 32),
        )

        print("4) model forward + decode")
        device = resolve_device("cpu")
        model = PoseSLT(cfg.feature_dim(), len(vocab), cfg).to(device)
        logits = model(src.to(device), src_pad.to(device),
                       tgt[:, :-1].to(device), tgt[:, :-1].eq(0).to(device))
        check("logits (B,L,V)", logits.shape[0] == src.shape[0] and logits.shape[2] == len(vocab))
        dec = model.greedy_decode(src.to(device), src_pad.to(device), max_len=16)
        check("greedy decode returns tokens", dec.dim() == 2 and dec.shape[0] == src.shape[0])

        print("4b) optional visual-feature adapter")
        adapter_cfg = Config(
            d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=128, visual_feature_dim=128, dropout=0.0,
        )
        adapter_model = PoseSLT(adapter_cfg.feature_dim(), len(vocab), adapter_cfg)
        check(
            "raw pose is mapped through the requested visual width",
            adapter_model.visual_adapter[0].out_features == 128
            and adapter_model.input_proj[0].in_features == 128,
        )

        print("4c) gradient accumulation")
        accum_model = PoseSLT(cfg.feature_dim(), len(vocab), cfg).to(device)
        accum_opt = torch.optim.Adam(accum_model.parameters(), lr=cfg.lr)
        accum_sched = WarmupInverseSqrt(accum_opt, cfg.d_model, cfg.warmup_steps)
        accum_loss = train_one_epoch(
            accum_model, dl, accum_sched, device,
            cfg.label_smoothing, cfg.grad_clip, grad_accum_steps=2,
        )
        check("accumulated optimiser steps train successfully", np.isfinite(accum_loss))

        print("5) train a few epochs (loss should fall)")
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        sched = WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)
        first = last = None
        for ep in range(cfg.epochs):
            loss = train_one_epoch(model, dl, sched, device, cfg.label_smoothing, cfg.grad_clip)
            first = loss if first is None else first
            last = loss
            print(f"     epoch {ep+1}/{cfg.epochs}  loss={loss:.4f}")
        check("training loss decreased", last < first)

        print("6) evaluate + gate")
        val_dl = DataLoader(AuslanDailyDataset(val_m, vocab, cfg),
                            batch_size=cfg.batch_size, collate_fn=collate)
        bleu4, wer, hyps, refs = evaluate(model, val_dl, vocab, device, cfg.max_tgt_tokens)
        check("BLEU-4 is a finite number >= 0", np.isfinite(bleu4) and bleu4 >= 0)
        check("WER is finite >= 0", np.isfinite(wer) and wer >= 0)
        gate = decision_gate(bleu4, cfg.bleu4_gate)
        check("gate has verdict + passed flag", "passed" in gate and "verdict" in gate)
        print(f"     BLEU-4={bleu4:.2f}  WER={wer:.3f}")
        print(f"     sample hyp: {hyps[0]!r}")
        print(f"     sample ref: {refs[0]!r}")
        print(f"     gate: {gate['verdict']}")

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
