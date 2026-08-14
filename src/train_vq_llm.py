"""Stage B: Train LLM decoder over VQ sign tokens.

Usage (Kaya/Colab):
    python train_vq_llm.py \
        --train_manifest /scratch/pmc099/pprasad/dwpose/manifests/train.tsv \
        --val_manifest   /scratch/pmc099/pprasad/dwpose/manifests/val.tsv \
        --test_manifest  /scratch/pmc099/pprasad/dwpose/manifests/test.tsv \
        --vq_encoder_ckpt runs/vq_pose/best.pt \
        --vq_modality pose \
        --vq_llm_name google/mt5-base \
        --vq_lora_r 16 --vq_lora_alpha 32 \
        --vq_llm_lr 1e-4 \
        --batch_size 8 --grad_accum_steps 2 \
        --epochs 20 --out_dir runs/vq_llm
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
from slt.llm_vq_decoder import VQSignLLM, build_vq_sign_llm
from slt.sign_vq import VQTokeniser, build_vq_tokeniser
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
    train_path = cfg.train_manifest
    val_path = cfg.val_manifest
    test_path = cfg.test_manifest
    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path, test_path

    tmp = tempfile.mkdtemp(prefix="vq_llm_smoke_")
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


class VQSignDataset(torch.utils.data.Dataset):
    """Dataset for VQ-LLM training: loads pose/RGB features + text targets."""

    def __init__(self, manifest: str, tokenizer, cfg: Config, root: str = "", limit_samples: int = 0):
        self.samples = read_manifest(manifest)
        if limit_samples:
            self.samples = self.samples[:limit_samples]
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.root = root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        clip, text = self.samples[i]
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

        tgt_enc = self.tokenizer(
            text,
            max_length=self.cfg.max_tgt_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_ids = tgt_enc.input_ids.squeeze(0)
        tgt_mask = tgt_enc.attention_mask.squeeze(0)

        return torch.from_numpy(feats), tgt_ids, tgt_mask, text


def collate_vq_llm(batch, pad_id: int = 0):
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
def evaluate_vq_llm(model, loader, tokenizer, device, cfg: Config
                    ) -> tuple[float, float, list[str], list[str]]:
    model.eval()
    hyps, refs = [], []
    for src, src_pad, _tgt_ids, _tgt_mask, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        generated = model.generate(
            src, src_pad,
            max_length=cfg.vq_max_gen_len,
            num_beams=cfg.vq_num_beams,
            length_penalty=cfg.vq_length_penalty,
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.vq_llm_name)

    train_m, val_m, test_m = ensure_synthetic_data(cfg, limit_samples)

    train_ds = VQSignDataset(train_m, tokenizer, cfg, root, limit_samples)
    val_ds = VQSignDataset(val_m, tokenizer, cfg, root, limit_samples)

    if cfg.length_bucket_batches:
        batch_sampler = LengthBucketBatchSampler(
            train_ds, cfg.batch_size, cfg.max_src_frames, cfg.seed)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler,
                              num_workers=cfg.num_workers, collate_fn=collate_vq_llm)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_vq_llm)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_vq_llm)

    # Build model
    feat_dim = 399 if cfg.vq_modality == "pose" else (1152 if cfg.vq_modality == "rgb" else 399+1152)
    model = build_vq_sign_llm(feat_dim, cfg, tokenizer)
    model = model.to(device)

    # Load VQ encoder from Stage A checkpoint
    if cfg.vq_encoder_ckpt and os.path.exists(cfg.vq_encoder_ckpt):
        model.load_vq_encoder(cfg.vq_encoder_ckpt, device)
    else:
        print("[warn] No VQ encoder checkpoint provided; using random VQ encoder")

    # Freeze VQ encoder, train only code embedder + LoRA
    for name, param in model.named_parameters():
        if "vq_encoder" in name:
            param.requires_grad_(False)

    trainable_params, _ = model.get_trainable_params()
    code_embedder_params = []
    lora_params = []
    for name, param in trainable_params:
        if "code_embedder" in name:
            code_embedder_params.append(param)
        else:
            lora_params.append(param)

    param_groups = [
        {"params": code_embedder_params, "lr": cfg.vq_llm_lr, "lr_scale": 1.0, "name": "code_embedder"},
        {"params": lora_params, "lr": cfg.vq_llm_lr, "lr_scale": 1.0, "name": "lora"},
    ]

    opt = torch.optim.AdamW(param_groups, lr=cfg.vq_llm_lr, weight_decay=cfg.weight_decay)

    updates_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
    total_steps = cfg.epochs * max(1, updates_per_epoch)
    sched = WarmupCosine(opt, cfg.vq_llm_lr, cfg.warmup_steps, total_steps)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for _, p in trainable_params)
    print(f"[setup] device={device} | vq_llm={cfg.vq_llm_name} "
          f"| params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
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
        print(f"[resume] {cfg.resume_checkpoint} -> epoch {start_epoch}")

    progress_path = os.path.join(cfg.out_dir, "progress.json")

    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
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

        bleu4, wer, hyps, refs = evaluate_vq_llm(model, val_dl, tokenizer, device, cfg)
        history.append({"epoch": ep, "loss": round(train_loss, 4),
                        "bleu4": round(bleu4, 3), "wer": round(wer, 4)})
        print(f"[ep {ep:03d}] loss={train_loss:.4f}  BLEU-4={bleu4:.2f}  WER={wer:.3f}")

        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": ep}
            atomic_torch_save({
                "model": model.state_dict(),
                "tokenizer_name": cfg.vq_llm_name,
                "cfg": cfg.to_dict(),
            }, os.path.join(cfg.out_dir, "best.pt"))

        latest = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "tokenizer_name": cfg.vq_llm_name,
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
            "device": str(device), "architecture": f"vq_llm_{cfg.vq_llm_name.split('/')[-1]}",
        }, progress_path)

    test = None
    if test_m and os.path.exists(test_m):
        test_ds = VQSignDataset(test_m, tokenizer, cfg, root, limit_samples)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_vq_llm)

        ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        t_bleu, t_wer, _, _ = evaluate_vq_llm(model, test_dl, tokenizer, device, cfg)
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
        "device": str(device), "architecture": f"vq_llm_{cfg.vq_llm_name.split('/')[-1]}",
    }, progress_path)

    print(f"[gate] {gate['verdict']}")
    return result


def parse_args() -> tuple[Config, str, int]:
    cfg = Config()
    p = argparse.ArgumentParser(description="VQ-LLM decoder for sign language translation")
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