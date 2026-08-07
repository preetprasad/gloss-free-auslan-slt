"""Training / evaluation loops for the gloss-free SLT baseline."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .data import PAD, Vocab
from .metrics import corpus_bleu4, corpus_wer, detokenize


def resolve_device(pref: str = "auto") -> torch.device:
    """Pick the compute device.

    "auto" prefers CUDA (Colab/Kaya), then Apple-Silicon MPS (local Mac), then
    CPU. MPS lets the baseline train on this Mac's GPU during the Kaya outage.
    """
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


class WarmupInverseSqrt:
    """Standard Transformer warmup → inverse-sqrt decay LR schedule."""

    def __init__(self, optimizer, d_model: int, warmup: int):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = max(1, warmup)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        s = self.step_num
        scale = (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = scale * float(g.get("lr_scale", 1.0))
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


class WarmupCosine:
    """Linear warmup to `base_lr`, then cosine decay to `floor*base_lr`.

    Unlike the Noam schedule (whose absolute LR is fixed by d_model and ignores
    cfg.lr), this respects `lr` and is stable at small step counts — so a tiny
    overfit set actually converges instead of bouncing out after the warmup peak.
    """

    def __init__(self, optimizer, base_lr: float, warmup: int, total_steps: int,
                 floor: float = 0.05):
        self.opt = optimizer
        self.base = base_lr
        self.warmup = max(1, warmup)
        self.total = max(total_steps, self.warmup + 1)
        self.floor = floor
        self.step_num = 0

    def step(self):
        import math
        self.step_num += 1
        s = self.step_num
        if s < self.warmup:
            lr = self.base * s / self.warmup
        else:
            p = (s - self.warmup) / max(1, self.total - self.warmup)
            lr = self.base * (self.floor + (1 - self.floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
        for g in self.opt.param_groups:
            g["lr"] = lr * float(g.get("lr_scale", 1.0))
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


def signcl_loss(memory, src_pad, tgt, margin: float = 1.0, zipf: float = 2.3):
    """SignCL contrastive loss (Ye et al. 2024, Eqns 4-6) on encoder features.

    Within each clip: adjacent valid frames are positive pairs (pulled together);
    frames > 2*frame_margin apart are negatives (pushed >= `margin` on L2-normalised
    features, so distances live in [0, 2]). frame_margin = max(10, T/text_len * zipf).
    """
    feats = torch.nn.functional.normalize(memory, dim=-1)   # (B,T,d) unit-norm
    total = memory.new_zeros(())
    n = 0
    for b in range(memory.size(0)):
        valid = (~src_pad[b]).nonzero(as_tuple=True)[0]
        Tv = valid.numel()
        if Tv < 4:
            continue
        f = feats[b, valid]                                  # (Tv, d)
        text_len = max(1, int((tgt[b] != PAD).sum().item()) - 2)  # minus BOS/EOS
        fmargin = max(10.0, (Tv / text_len) * zipf)
        pos = (f[1:] - f[:-1]).norm(dim=-1).mean()           # adjacent -> pull together
        gap = int(2 * fmargin)
        if Tv > gap + 1:
            i = torch.arange(0, Tv - gap - 1, device=f.device)
            j = i + gap + 1
            if i.numel() > 64:                               # cap negatives for speed
                sel = torch.randperm(i.numel(), device=f.device)[:64]
                i, j = i[sel], j[sel]
            dneg = (f[i] - f[j]).norm(dim=-1)
            neg = torch.clamp(margin - dneg, min=0).mean()   # push >= margin apart
        else:
            neg = memory.new_zeros(())
        total = total + pos + neg
        n += 1
    return total / max(1, n)


def train_one_epoch(model, loader, sched, device, label_smoothing=0.1, grad_clip=1.0,
                    signcl_weight=0.0, signcl_margin=1.0, signcl_zipf=2.3,
                    grad_accum_steps=1, semantic_bank=None,
                    gaslt_similarity_weight=0.0, loss_normalization="tokens",
                    vlp_aux=None, vlp_aux_weight=0.0):
    model.train()
    reduction = "sum" if loss_normalization == "batch" else "mean"
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing,
                               reduction=reduction)
    use_scl = signcl_weight > 0
    use_semantic = semantic_bank is not None and gaslt_similarity_weight > 0
    use_vlp_aux = vlp_aux is not None and vlp_aux_weight > 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    total, n = 0.0, 0
    sched.zero_grad()
    for step, (src, src_pad, tgt, texts) in enumerate(loader, start=1):
        src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_pad = tgt_in.eq(PAD)
        if use_scl or use_semantic or use_vlp_aux:
            logits, memory = model(src, src_pad, tgt_in, tgt_pad, return_memory=True)
            ce = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if loss_normalization == "batch":
                ce = ce / src.size(0)
            loss = ce
            if use_scl:
                loss = loss + signcl_weight * signcl_loss(
                    memory, src_pad, tgt, signcl_margin, signcl_zipf)
            if use_semantic:
                from .reference_model import gaslt_similarity_loss
                target_similarity = semantic_bank.similarity(texts, device)
                loss = loss + gaslt_similarity_loss(
                    memory, src_pad, target_similarity, gaslt_similarity_weight)
            if use_vlp_aux:
                auxiliary, _ = vlp_aux(memory, src_pad, tgt)
                loss = loss + vlp_aux_weight * auxiliary
        else:
            logits = model(src, src_pad, tgt_in, tgt_pad)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if loss_normalization == "batch":
                loss = loss / src.size(0)
        (loss / grad_accum_steps).backward()
        update_now = step % grad_accum_steps == 0 or step == len(loader)
        if update_now:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            sched.step()
            sched.zero_grad()
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, vocab: Vocab, device, max_len=64,
             bleu_tokenize: str = "13a", detok: bool = True
             ) -> Tuple[float, float, List[str], List[str]]:
    model.eval()
    hyps, refs = [], []
    for src, src_pad, _tgt, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        pred = model.greedy_decode(src, src_pad, max_len=max_len)
        for row, ref in zip(pred.tolist(), texts):
            hyps.append(vocab.decode(row))
            refs.append(ref)
    # Literature protocol: detokenise both sides, then sacreBLEU 13a re-splits
    # them consistently (comparable to published BLEU). WER computed on the same
    # detokenised text for consistency.
    h_eval = [detokenize(h) for h in hyps] if detok else hyps
    r_eval = [detokenize(r) for r in refs] if detok else refs
    bleu4 = corpus_bleu4(h_eval, r_eval, tokenize=bleu_tokenize)
    wer = corpus_wer(h_eval, r_eval)
    return bleu4, wer, hyps, refs
