"""P5c: face-augmented temporal RGB fusion with the P3 training protocol.

This is intentionally a *channel-addition baseline*, not a reproduction of a
raw-video face/hand architecture: a 384-D frozen DINO face vector is appended
to the existing 1,152-D full-signer/hand vector at each sampled frame.  The
pose encoder, temporal adapter, decoder, pseudo-content CTC target, seed and
schedule match P3.  It trains and selects on train/validation only.
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
from slt.rgb_fusion import RGBDataset, build_gloss_vocab, collate_rgb, evaluate_fusion, to_gloss
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
    parser.add_argument("--ctc_weight", type=float, default=0.30)
    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0,
                        help="smoke-only cap; never use its metric for model selection")
    args = parser.parse_args()
    extra = {"ctc_weight", "limit_samples", "val_limit"}
    return Config(**{key: value for key, value in vars(args).items() if key not in extra}), args


def read_texts(manifest: str) -> list[str]:
    with open(manifest, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[1] for line in handle if line.strip()]


def warm_start(model: TemporalRGBFusionSLT, cfg: Config) -> int:
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
    if not (cfg.rgb_feat_dir and cfg.face_feat_dir and cfg.face_feature_dim > 0):
        raise ValueError("P5c requires --rgb_feat_dir, --face_feat_dir and --face_feature_dim")
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    train_texts = read_texts(cfg.train_manifest)
    vocab = Vocab.build(train_texts, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    gloss_to_id = build_gloss_vocab(train_texts)

    def make_loader(manifest: str, shuffle: bool, limit: int = 0) -> DataLoader:
        dataset = RGBDataset(manifest, vocab, cfg, "")
        if limit:
            dataset.samples = dataset.samples[:limit]
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate_rgb)

    def ctc_targets(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        identifiers: list[int] = []; lengths: list[int] = []
        for text in texts:
            tokens = [gloss_to_id.get(token, gloss_to_id["<g>"]) for token in to_gloss(text)]
            identifiers.extend(tokens); lengths.append(len(tokens))
        return torch.tensor(identifiers, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    train_loader = make_loader(cfg.train_manifest, True, args.limit_samples)
    val_loader = make_loader(cfg.val_manifest, False, args.val_limit)
    rgb_input_dim = int(cfg.rgb_feature_dim) + int(cfg.face_feature_dim)
    model = TemporalRGBFusionSLT(cfg.feature_dim(), rgb_input_dim, len(vocab), cfg, "pyramid").to(device)
    loaded = warm_start(model, cfg)
    ctc_head = nn.Linear(cfg.d_model, len(gloss_to_id) + 1).to(device)
    parameters = list(model.parameters()) + list(ctc_head.parameters())
    optimiser = torch.optim.Adam(parameters, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = (WarmupCosine(optimiser, cfg.lr, cfg.warmup_steps, total_steps)
                 if cfg.lr_schedule == "cosine" else WarmupInverseSqrt(optimiser, cfg.d_model, cfg.warmup_steps))
    translation_loss = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    max_length = cfg.decode_max_len or cfg.max_tgt_tokens
    metadata = {
        "experiment": "p5c_face_channel_plus_temporal_pyramid_and_legacy_single_ctc",
        "rgb_feature_dim": int(cfg.rgb_feature_dim), "face_feature_dim": int(cfg.face_feature_dim),
        "face_input": "DWPose-localised frozen-DINO face ROI", "warm_started_base_tensors": loaded,
        "ctc_weight": args.ctc_weight, "ctc_memory": "pose_only_exact_legacy_placement",
        "test_set_used": False, "validation_smoke_only": bool(args.val_limit),
    }
    with open(os.path.join(cfg.out_dir, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[p5c] device={device} rgb_input={rgb_input_dim} params={sum(p.numel() for p in parameters)/1e6:.2f}M "
          f"test_set_used=False val_smoke_only={bool(args.val_limit)}", flush=True)

    best = {"bleu4": -1.0, "wer": float("inf"), "epoch": 0}; history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train(); ctc_head.train()
        total = ce_total = ctc_total = count = 0.0
        for src, src_pad, rgb, rgb_pad, target, texts in train_loader:
            src, src_pad, rgb, rgb_pad, target = (src.to(device), src_pad.to(device), rgb.to(device),
                                                   rgb_pad.to(device), target.to(device))
            target_in, target_out = target[:, :-1], target[:, 1:]
            logits, pose_memory = model(src, src_pad, rgb, rgb_pad, target_in, target_in.eq(PAD), return_memory=True)
            loss_translation = translation_loss(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            log_probs = torch.log_softmax(ctc_head(pose_memory), dim=-1).transpose(0, 1).float().cpu()
            token_ids, token_lengths = ctc_targets(texts)
            loss_ctc = ctc(log_probs, token_ids, (~src_pad).sum(1).cpu(), token_lengths)
            loss = loss_translation + args.ctc_weight * loss_ctc.to(device)
            scheduler.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip); scheduler.step()
            batch = src.size(0)
            total += float(loss.detach()) * batch; ce_total += float(loss_translation.detach()) * batch
            ctc_total += float(loss_ctc.detach()) * batch; count += batch
        bleu4, wer, hypotheses, _ = evaluate_fusion(
            model, val_loader, vocab, device, max_len=max_length,
            bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        record = {"epoch": epoch, "loss": total / count, "translation_loss": ce_total / count,
                  "ctc_loss": ctc_total / count, "bleu4": bleu4, "wer": wer, "uniq": len(set(hypotheses))}
        history.append(record)
        print("[p5c ep {epoch:03d}] loss={loss:.3f} ce={translation_loss:.3f} ctc={ctc_loss:.3f} "
              "val_bleu4={bleu4:.3f} wer={wer:.3f} uniq={uniq}".format(**record), flush=True)
        if bleu4 > best["bleu4"]:
            best = {"bleu4": bleu4, "wer": wer, "epoch": epoch}
            torch.save({"model": model.state_dict(), "ctc_head": ctc_head.state_dict(),
                        "vocab": vocab.itos, "cfg": cfg.to_dict(), "gloss2idx": gloss_to_id,
                        "experiment": metadata}, os.path.join(cfg.out_dir, "best.pt"))
        with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": "running", "epoch": epoch, "epochs": cfg.epochs, "best": best,
                       "history": history, **metadata}, handle, indent=2)

    with open(os.path.join(cfg.out_dir, "progress.json"), encoding="utf-8") as handle:
        progress = json.load(handle)
    progress["status"] = "done"
    with open(os.path.join(cfg.out_dir, "progress.json"), "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)
    print(f"[p5c DONE] best_val_bleu4={best['bleu4']:.3f} epoch={best['epoch']} test_set_used=False", flush=True)


if __name__ == "__main__":
    main()
