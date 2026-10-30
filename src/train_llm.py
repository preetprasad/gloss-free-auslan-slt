"""Train the mT5+LoRA decoder for gloss-free Auslan-Daily SLT.

Usage (on Kaya/Colab):
    python train_llm.py \
        --train_manifest /scratch/pmc099/pprasad/dwpose/manifests/train.tsv \
        --val_manifest   /scratch/pmc099/pprasad/dwpose/manifests/val.tsv \
        --test_manifest  /scratch/pmc099/pprasad/dwpose/manifests/test.tsv \
        --llm_name google/mt5-base \
        --lora_r 16 --lora_alpha 32 \
        --llm_lr 1e-4 --pose_encoder_lr_scale 1.0 \
        --batch_size 8 --grad_accum_steps 2 \
        --epochs 20 --out_dir runs/mt5_base_lora

Smoke test (CPU, synthetic data):
    python train_llm.py --device cpu --limit_samples 64 --epochs 1 --batch_size 2 \
        --llm_name google/mt5-base --lora_r 16 --lora_alpha 32 --out_dir runs/smoke_test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, MT5Config, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from slt.data import (AuslanDailyDataset, LengthBucketBatchSampler,
                      collate, read_manifest, load_pose, normalize_clip)
from slt.engine import WarmupCosine, evaluate, resolve_device
from slt.llm_decoder import PoseLLM, build_pose_llm
from slt.metrics import bleu_signature, corpus_bleu4, corpus_wer, decision_gate, detokenize


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
    """Create synthetic data if manifest files don't exist."""
    train_path = cfg.train_manifest
    val_path = cfg.val_manifest
    test_path = cfg.test_manifest

    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path, test_path

    tmp = tempfile.mkdtemp(prefix="slt_llm_smoke_")
    clips = os.path.join(tmp, "clips")
    man = os.path.join(tmp, "manifests")
    os.makedirs(clips, exist_ok=True)
    os.makedirs(man, exist_ok=True)

    n_train = limit_samples if limit_samples else 48
    n_val = limit_samples if limit_samples else 16
    n_test = limit_samples if limit_samples else 16

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
    test_m = write_split("test", n_test, 9000)

    return train_m, val_m, test_m


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


def build_loaders(cfg: Config, tokenizer, root: str = "", limit_samples: int = 0):
    train_m, val_m, test_m = ensure_synthetic_data(cfg, limit_samples)

    train_rows = read_manifest(train_m)
    if limit_samples:
        train_rows = train_rows[:limit_samples]

    val_rows = read_manifest(val_m)
    if limit_samples:
        val_rows = val_rows[:limit_samples]

    train_ds = AuslanDailyDatasetLLM(train_m, tokenizer, cfg, root, limit_samples)
    val_ds = AuslanDailyDatasetLLM(val_m, tokenizer, cfg, root, limit_samples)

    if cfg.length_bucket_batches:
        batch_sampler = LengthBucketBatchSampler(
            train_ds, cfg.batch_size, cfg.max_src_frames, cfg.seed)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler,
                              num_workers=cfg.num_workers, collate_fn=collate_llm)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_llm)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_llm)
    return train_dl, val_dl, train_rows, test_m


class AuslanDailyDatasetLLM(AuslanDailyDataset):
    """Dataset that tokenizes targets with mT5 tokenizer instead of project Vocab."""

    def __init__(self, manifest: str, tokenizer, cfg: Config, root: str = "", limit_samples: int = 0):
        self.samples = read_manifest(manifest)
        if limit_samples:
            self.samples = self.samples[:limit_samples]
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.root = root

    def __getitem__(self, i: int):
        clip, text = self.samples[i]
        path = os.path.join(self.root, clip) if self.root else clip

        if self.cfg.input_representation == "precomputed":
            from slt.data import load_precomputed_features
            feats = load_precomputed_features(path, self.cfg.precomputed_feature_key)
            if feats.shape[1] != self.cfg.feature_dim():
                raise ValueError(
                    f"precomputed feature width {feats.shape[1]} != configured "
                    f"{self.cfg.feature_dim()} in {path}")
            if feats.shape[0] > self.cfg.max_src_frames:
                selected = np.linspace(0, feats.shape[0] - 1, self.cfg.max_src_frames).round().astype(int)
                feats = feats[selected]
        else:
            from slt.data import load_pose, normalize_clip
            kps, bbox = load_pose(path)
            feats, _ = normalize_clip(kps, bbox, self.cfg)

        tgt_enc = self.tokenizer(
            text,
            max_length=self.cfg.max_tgt_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_ids = tgt_enc.input_ids.squeeze(0)
        tgt_mask = tgt_enc.attention_mask.squeeze(0)

        return (
            torch.from_numpy(feats),
            tgt_ids,
            tgt_mask,
            text,
        )


def collate_llm(batch, pad_id: int = 0):
    feats, tgt_ids, tgt_masks, texts = zip(*batch)
    B = len(feats)
    Tmax = max(f.shape[0] for f in feats)
    Fdim = feats[0].shape[1]
    src = torch.zeros(B, Tmax, Fdim, dtype=torch.float32)
    src_pad = torch.ones(B, Tmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        t = f.shape[0]
        src[i, :t] = f
        src_pad[i, :t] = False

    tgt_ids = torch.stack(tgt_ids, dim=0)
    tgt_masks = torch.stack(tgt_masks, dim=0)

    return src, src_pad, tgt_ids, tgt_masks, list(texts)


@torch.no_grad()
def evaluate_llm(model, loader, tokenizer, device, cfg: Config
                 ) -> tuple[float, float, list[str], list[str]]:
    model.eval()
    hyps, refs = [], []
    for src, src_pad, _tgt_ids, _tgt_mask, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        generated = model.generate(
            src, src_pad,
            max_length=cfg.max_gen_len,
            num_beams=cfg.num_beams,
            length_penalty=cfg.length_penalty,
        )
        for row, ref in zip(generated.tolist(), texts):
            hyp = tokenizer.decode(row, skip_special_tokens=True)
            hyps.append(hyp)
            refs.append(ref)

    h_eval = [detokenize(h) for h in hyps]
    r_eval = [detokenize(r) for r in refs]
    bleu4 = corpus_bleu4(h_eval, r_eval, tokenize=cfg.bleu_tokenize)
    wer = corpus_wer(h_eval, r_eval)
    return bleu4, wer, hyps, refs


def run(cfg: Config, root: str = "", limit_samples: int = 0) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)

    train_dl, val_dl, train_rows, test_m = build_loaders(cfg, tokenizer, root, limit_samples)

    model = build_pose_llm(cfg.feature_dim(), cfg, tokenizer)
    model = model.to(device)

    trainable_params, frozen_params = model.get_trainable_params()
    pose_encoder_params = []
    lora_params = []
    for name, param in trainable_params:
        if name.startswith("pose_encoder") or name.startswith("bridge"):
            pose_encoder_params.append(param)
        else:
            lora_params.append(param)

    param_groups = [
        {"params": pose_encoder_params, "lr": cfg.llm_lr * cfg.pose_encoder_lr_scale,
         "lr_scale": cfg.pose_encoder_lr_scale, "name": "pose_encoder"},
        {"params": lora_params, "lr": cfg.llm_lr, "lr_scale": 1.0, "name": "lora"},
    ]

    opt = torch.optim.AdamW(param_groups, lr=cfg.llm_lr, weight_decay=cfg.weight_decay)

    updates_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
    total_steps = cfg.epochs * max(1, updates_per_epoch)
    sched = WarmupCosine(opt, cfg.llm_lr, cfg.warmup_steps, total_steps)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for _, p in trainable_params)
    print(f"[setup] device={device} | llm={cfg.llm_name} "
          f"| params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
          f"| feat_dim={cfg.feature_dim()} | llm_d={model.llm_d_model} "
          f"| batch={cfg.batch_size} accum={cfg.grad_accum_steps} "
          f"eff_batch={cfg.batch_size * cfg.grad_accum_steps} "
          f"| train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    best = {"bleu4": -1.0}
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
        print(f"[resume] {cfg.resume_checkpoint} -> epoch {start_epoch} "
              f"(best BLEU-4={best['bleu4']:.2f})")

    progress_path = os.path.join(cfg.out_dir, "progress.json")

    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        reduction = "sum" if cfg.loss_normalization == "batch" else "mean"
        total, n = 0.0, 0
        sched.zero_grad()

        for step, (src, src_pad, tgt_ids, tgt_mask, _texts) in enumerate(train_dl, start=1):
            src = src.to(device)
            src_pad = src_pad.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_mask = tgt_mask.to(device)

            outputs = model(src, src_pad, tgt_ids, tgt_mask)
            loss = outputs.loss

            if cfg.loss_normalization == "batch":
                loss = loss / src.size(0)

            (loss / cfg.grad_accum_steps).backward()
            update_now = step % cfg.grad_accum_steps == 0 or step == len(train_dl)
            if update_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                sched.step()
                sched.zero_grad()
            total += loss.item() * src.size(0)
            n += src.size(0)

        train_loss = total / max(1, n)

        bleu4, wer, hyps, refs = evaluate_llm(model, val_dl, tokenizer, device, cfg)
        history.append({"epoch": ep, "loss": round(train_loss, 4),
                        "bleu4": round(bleu4, 3), "wer": round(wer, 4)})
        print(f"[ep {ep:03d}] loss={train_loss:.4f}  BLEU-4={bleu4:.2f}  WER={wer:.3f}")

        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": ep}
            atomic_torch_save({
                "model": model.state_dict(),
                "tokenizer_name": cfg.llm_name,
                "cfg": cfg.to_dict(),
            }, os.path.join(cfg.out_dir, "best.pt"))

        latest = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "tokenizer_name": cfg.llm_name,
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
            "device": str(device), "architecture": f"mt5_lora_{cfg.llm_name.split('/')[-1]}",
        }, progress_path)

    test = None
    if test_m and os.path.exists(test_m):
        from slt.data import read_manifest
        test_ds = AuslanDailyDatasetLLM(test_m, tokenizer, cfg, root, limit_samples)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_llm)

        ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        t_bleu, t_wer, _, _ = evaluate_llm(model, test_dl, tokenizer, device, cfg)
        test = {"bleu4": t_bleu, "wer": t_wer, "n": len(test_dl.dataset),
                "from_epoch": best.get("epoch")}
        print(f"[TEST] BLEU-4={t_bleu:.2f}  WER={t_wer:.3f}  (n={test['n']}, "
              f"ckpt from ep{best.get('epoch')})")

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
        "device": str(device), "architecture": f"mt5_lora_{cfg.llm_name.split('/')[-1]}",
    }, progress_path)

    print(f"[gate] {gate['verdict']}")
    return result


def parse_args() -> tuple[Config, str, int]:
    cfg = Config()
    p = argparse.ArgumentParser(description="mT5+LoRA decoder for Auslan-Daily SLT")
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