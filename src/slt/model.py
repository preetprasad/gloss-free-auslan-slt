"""Baseline model: pose-sequence Transformer encoder-decoder for gloss-free SLT.

Pipeline:  (T, F) normalised HALPE-136 features
           -> linear projection to d_model  (+ optional confidence gating)
           -> sinusoidal positional encoding
           -> Transformer encoder (signer pose context)
           -> Transformer decoder (autoregressive English tokens)
           -> linear generator over the translation vocabulary.

Deliberately small and standard — this is the baseline to beat, not the novel
Phase 3 system. `batch_first=True` throughout.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import BOS, EOS, PAD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class PoseSLT(nn.Module):
    def __init__(self, feat_dim: int, vocab_size: int, cfg):
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.pos_enc = PositionalEncoding(d, dropout=cfg.dropout)
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(x, src_key_padding_mask=src_pad)

    def decode_step(self, tgt_in, memory, src_pad, tgt_pad):
        L = tgt_in.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt_in.device, dtype=torch.bool), 1)
        y = self.pos_enc(self.tok_emb(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            y, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.generator(out)

    def forward(self, src, src_pad, tgt_in, tgt_pad):
        memory = self.encode(src, src_pad)
        return self.decode_step(tgt_in, memory, src_pad, tgt_pad)

    @torch.no_grad()
    def greedy_decode(self, src, src_pad, max_len: int = 64) -> torch.Tensor:
        self.eval()
        B = src.size(0)
        memory = self.encode(src, src_pad)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.decode_step(ys, memory, src_pad, None)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            nxt[done] = PAD
            ys = torch.cat([ys, nxt], dim=1)
            done |= nxt.squeeze(1).eq(EOS)
            if bool(done.all()):
                break
        return ys
