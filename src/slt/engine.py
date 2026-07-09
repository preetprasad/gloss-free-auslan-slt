"""Training / evaluation loops for the gloss-free SLT baseline."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .data import PAD, Vocab
from .metrics import corpus_bleu4, corpus_wer


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            g["lr"] = scale
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


def train_one_epoch(model, loader, sched, device, label_smoothing=0.1, grad_clip=1.0):
    model.train()
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    total, n = 0.0, 0
    for src, src_pad, tgt, _ in loader:
        src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_pad = tgt_in.eq(PAD)
        logits = model(src, src_pad, tgt_in, tgt_pad)
        loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        sched.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        sched.step()
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, vocab: Vocab, device, max_len=64) -> Tuple[float, float, List[str], List[str]]:
    model.eval()
    hyps, refs = [], []
    for src, src_pad, _tgt, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        pred = model.greedy_decode(src, src_pad, max_len=max_len)
        for row, ref in zip(pred.tolist(), texts):
            hyps.append(vocab.decode(row))
            refs.append(ref)
    bleu4 = corpus_bleu4(hyps, refs)
    wer = corpus_wer(hyps, refs)
    return bleu4, wer, hyps, refs
