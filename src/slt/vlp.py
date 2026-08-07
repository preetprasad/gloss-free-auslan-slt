"""Visual-Language Pretraining (GFSLT-VLP style, adapted to pose) — Stage 1.

Aligns the pose encoder to text with a CLIP-style contrastive objective on the
Auslan-Daily data itself (no extra data). A pose clip and its own subtitle are a
positive pair; all other subtitles in the batch are negatives. This gives the
encoder a text-grounded signal *before* translation, so it doesn't start cold and
collapse onto the frequent-phrase prior (the failure we diagnosed).

The pose branch is built by the same model factory as translation, so either the
legacy PoseSLT encoder or the source-compatible SignJoey encoder can be pretrained
and loaded straight into its matching translator via `train.py --init_encoder`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD
from .model import PositionalEncoding
from .reference_model import build_model


def _masked_mean(h, pad):
    """Mean-pool (B,T,d) over the non-pad positions (pad: True=ignore)."""
    m = (~pad).unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class TextEncoder(nn.Module):
    """Small trainable Transformer encoder over target tokens -> unit embedding."""

    def __init__(self, vocab, d, nhead, layers, ff, dropout, proj):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = PositionalEncoding(d, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d, nhead, ff, dropout, batch_first=True)
        # Nested-tensor fast path is optional and its mask check is unavailable
        # on Apple MPS in eval mode. Disabling it preserves identical dense
        # Transformer math and keeps frozen VLP auxiliaries device-native.
        self.enc = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False)
        self.proj = nn.Linear(d, proj)

    def forward(self, tgt):
        pad = tgt.eq(PAD)
        x = self.pos(self.emb(tgt) * math.sqrt(self.d))
        h = self.enc(x, src_key_padding_mask=pad)
        return F.normalize(self.proj(_masked_mean(h, pad)), dim=-1)


class VLPModel(nn.Module):
    """Pose encoder (PoseSLT.encode) + text encoder, trained with a symmetric
    InfoNCE / CLIP loss. Save `self.slt.state_dict()` -> loadable by PoseSLT."""

    def __init__(self, feat_dim, vocab, cfg, proj=256, temp=0.07):
        super().__init__()
        self.slt = build_model(feat_dim, vocab, cfg)          # reuse the selected real encoder
        self.pose_proj = nn.Linear(cfg.d_model, proj)
        self.text = TextEncoder(vocab, cfg.d_model, cfg.nhead,
                                max(2, cfg.num_encoder_layers // 2),
                                cfg.dim_feedforward, cfg.dropout, proj)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def pose_embed(self, src, src_pad):
        mem = self.slt.encode(src, src_pad)                   # (B,T,d)
        return F.normalize(self.pose_proj(_masked_mean(mem, src_pad)), dim=-1)

    def forward(self, src, src_pad, tgt):
        p = self.pose_embed(src, src_pad)                     # (B, proj)
        t = self.text(tgt)                                    # (B, proj)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (p @ t.t())                          # (B, B)
        labels = torch.arange(p.size(0), device=p.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        # retrieval accuracy (pose->text top-1) as a readable training signal
        acc = (logits.argmax(1) == labels).float().mean()
        return loss, acc


class FrozenVLPAuxiliary(nn.Module):
    """Pretrained frozen VLP heads applied to a translator's shared memory.

    The text encoder, pose projection, and temperature remain fixed. Gradients
    flow through the fixed pose projection into the live translation encoder,
    preserving pose/text alignment without maintaining a second pose model.
    """

    def __init__(self, checkpoint: dict, vocab_size: int, cfg):
        super().__init__()
        state = checkpoint["model"]
        proj = int(checkpoint.get("vlp_proj", 256))
        self.pose_proj = nn.Linear(cfg.d_model, proj)
        self.text = TextEncoder(
            vocab_size, cfg.d_model, cfg.nhead,
            max(2, cfg.num_encoder_layers // 2),
            cfg.dim_feedforward, cfg.dropout, proj)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        selected = {}
        for key, value in state.items():
            if key.startswith("pose_proj."):
                selected[key] = value
            elif key.startswith("text."):
                selected[key] = value
            elif key == "logit_scale":
                selected[key] = value
        missing, unexpected = self.load_state_dict(selected, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"incompatible VLP auxiliary checkpoint: missing={missing}, "
                f"unexpected={unexpected}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Keep the frozen text target and BatchNorm/dropout behaviour stable.
        return super().train(False)

    def forward(self, memory, src_pad, tgt):
        pose = F.normalize(self.pose_proj(_masked_mean(memory, src_pad)), dim=-1)
        with torch.no_grad():
            text = self.text(tgt)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (pose @ text.t())
        labels = torch.arange(pose.size(0), device=pose.device)
        loss = 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))
        acc = (logits.argmax(1) == labels).float().mean()
        return loss, acc
