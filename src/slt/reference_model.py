"""SignJoey-compatible SL-Transformer and GASLT architectures.

This module intentionally keeps the public reference architecture separate from
the project's original ``torch.nn.Transformer`` baseline.  It follows the
official implementations at pinned commits:

* SL-Transformer: neccam/slt@90588825f6229474bc19ac7a6b30ea3116635ba3
* GASLT: YinAoXiong/GASLT@6a36a0e43e29456dc85878f16f64b1c265d3a159

The Auslan-Daily paper does not publish the adapter that turns flattened
AlphaPose coordinates into its model input.  When ``visual_feature_dim`` is
different from the raw pose width, ``pose_adapter`` is therefore an explicit,
learned interface adapter; everything after it mirrors SignJoey.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BOS, EOS, PAD


class SinusoidalPosition(nn.Module):
    """The additive, unscaled positional encoding used by SignJoey."""

    def __init__(self, size: int, max_len: int = 4096):
        super().__init__()
        if size % 2:
            raise ValueError(f"positional encoding requires an even size, got {size}")
        pe = torch.zeros(max_len, size)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, size, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MaskedBatchNorm1d(nn.Module):
    """SignJoey's BatchNorm over valid sequence positions only during training."""

    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.norm = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # valid may be [B,T] or [B,T,1], with True at real positions.
        valid = valid.reshape(-1).bool()
        flat = x.reshape(-1, self.features)
        if self.training:
            selected = flat[valid]
            # Normal training batches contain many valid frames/tokens.  The
            # guard only makes one-sample smoke tests well-defined.
            if selected.size(0) > 1:
                normalised = self.norm(selected)
                flat = flat.clone()
                flat[valid] = normalised
        else:
            flat = self.norm(flat)
        return flat.reshape_as(x)


def _activation(name: str) -> nn.Module:
    if name == "softsign":
        return nn.Softsign()
    if name == "relu":
        return nn.ReLU()
    if name in ("none", ""):
        return nn.Identity()
    raise ValueError(f"unsupported reference embedding activation: {name}")


class SpatialEmbedding(nn.Module):
    """Linear visual projection -> masked BatchNorm -> activation."""

    def __init__(self, input_size: int, output_size: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = MaskedBatchNorm1d(output_size)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(x), valid))


class WordEmbedding(nn.Module):
    """SignJoey word/BPE embedding, including GASLT's optional 300->512 map."""

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 activation: str):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.projection = (nn.Linear(embedding_size, hidden_size)
                           if embedding_size != hidden_size else nn.Identity())
        self.norm = MaskedBatchNorm1d(hidden_size)
        self.activation = _activation(activation)

    def forward(self, ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.projection(self.lookup(ids))
        return self.activation(self.norm(x, valid))


class MultiHeadAttention(nn.Module):
    """The OpenNMT-derived attention implementation used by SignJoey."""

    def __init__(self, heads: int, size: int, dropout: float):
        super().__init__()
        if size % heads:
            raise ValueError(f"model size {size} must be divisible by {heads} heads")
        self.heads = heads
        self.head_size = size // heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = k.size(0)
        k = self.k_layer(k).view(batch, -1, self.heads, self.head_size).transpose(1, 2)
        v = self.v_layer(v).view(batch, -1, self.heads, self.head_size).transpose(1, 2)
        q = self.q_layer(q).view(batch, -1, self.heads, self.head_size).transpose(1, 2)
        scores = torch.matmul(q / math.sqrt(self.head_size), k.transpose(2, 3))
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, -1,
                                                            self.heads * self.head_size)
        return self.output_layer(context)


class GlossAttention(nn.Module):
    """GASLT deformable local attention with N learned sample offsets.

    This preserves the official implementation's sampling geometry, including
    circular (modulo) positions and ``grid_sample(..., align_corners=False)``.
    """

    def __init__(self, heads: int, size: int, dropout: float, num_keys: int = 7):
        super().__init__()
        if size % heads:
            raise ValueError(f"model size {size} must be divisible by {heads} heads")
        self.heads = heads
        self.head_size = size // heads
        self.num_keys = num_keys
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.sample_offsets = nn.Linear(size, heads * num_keys)
        self.output_layer = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        batch, key_len, _ = k.shape
        query_len = q.size(1)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        offsets = self.sample_offsets(q).reshape(
            batch, query_len, self.heads, self.num_keys).transpose(1, 2)
        k = k.view(batch, key_len, self.heads, self.head_size).transpose(1, 2)
        v = v.view(batch, key_len, self.heads, self.head_size).transpose(1, 2)
        q = q.view(batch, query_len, self.heads, self.head_size).transpose(1, 2)

        location = torch.arange(query_len, device=q.device, dtype=q.dtype).view(1, 1, -1, 1)
        left = -self.num_keys // 2
        reference = torch.arange(left, self.num_keys + left,
                                 device=q.device, dtype=q.dtype).view(1, 1, 1, -1)
        # The public code takes (valid length - 1) before modulo.  Clamp only
        # protects degenerate smoke samples; real Auslan clips are longer.
        wrap = (valid_mask.sum(-1).squeeze(1).to(q.dtype) - 1).clamp_min(1)
        samples = (reference + offsets + location) % wrap[:, None, None, None]
        samples = samples / max(1, key_len - 1) * 2 - 1
        sampled_k = self._sample_like_official_grid_sample(k, samples)
        sampled_v = self._sample_like_official_grid_sample(v, samples)
        scores = ((q / math.sqrt(self.head_size)).unsqueeze(3) * sampled_k).sum(-1)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = (attention.unsqueeze(-1) * sampled_v).sum(-2)
        context = context.transpose(1, 2).reshape(batch, query_len, -1)
        return self.output_layer(context)

    @staticmethod
    def _sample_like_official_grid_sample(sequence: torch.Tensor,
                                          x_normalised: torch.Tensor) -> torch.Tensor:
        """Pure-Torch equivalent of the official 1xT 2-D ``grid_sample``.

        GASLT fixes the y coordinate to +1 for a height-one grid and uses
        ``align_corners=False`` with zero padding.  That makes the y interpolation
        weight exactly 0.5.  Expressing the remaining x interpolation with
        ``gather`` is numerically equivalent on CPU and has an MPS backward pass.
        """
        batch, heads, width, depth = sequence.shape
        # align_corners=False unnormalisation from PyTorch GridSampler.cpp.
        x = ((x_normalised + 1) * width - 1) / 2
        lower = torch.floor(x)
        upper = lower + 1
        upper_weight = x - lower
        lower_weight = 1 - upper_weight

        source = sequence.unsqueeze(2).expand(-1, -1, x.size(2), -1, -1)

        def gather(index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            in_bounds = (index >= 0) & (index < width)
            safe = index.clamp(0, width - 1).long()
            safe = safe.unsqueeze(-1).expand(-1, -1, -1, -1, depth)
            values = torch.gather(source, 3, safe)
            return values * (weight * in_bounds).unsqueeze(-1)

        # y=+1 on a height-one input contributes 0.5 under zero padding.
        return 0.5 * (gather(lower, lower_weight) + gather(upper, upper_weight))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size: int, ff_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.layers = nn.Sequential(
            nn.Linear(size, ff_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_size, size), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.norm(x))


class ReferenceEncoderLayer(nn.Module):
    def __init__(self, size: int, ff_size: int, heads: int, dropout: float,
                 attention: str, num_keys: int):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.attention = (GlossAttention(heads, size, dropout, num_keys)
                          if attention == "gloss" else MultiHeadAttention(heads, size, dropout))
        self.ff = PositionwiseFeedForward(size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_valid: torch.Tensor) -> torch.Tensor:
        norm = self.norm(x)
        x = x + self.dropout(self.attention(norm, norm, norm, src_valid))
        return self.ff(x)


class ReferenceDecoderLayer(nn.Module):
    def __init__(self, size: int, ff_size: int, heads: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, size, dropout)
        self.source_attention = MultiHeadAttention(heads, size, dropout)
        self.self_norm = nn.LayerNorm(size, eps=1e-6)
        self.source_norm = nn.LayerNorm(size, eps=1e-6)
        self.ff = PositionwiseFeedForward(size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_valid: torch.Tensor, tgt_valid: torch.Tensor) -> torch.Tensor:
        norm = self.self_norm(x)
        x = x + self.dropout(self.self_attention(norm, norm, norm, tgt_valid))
        norm = self.source_norm(x)
        x = x + self.dropout(self.source_attention(memory, memory, norm, src_valid))
        return self.ff(x)


class ReferencePoseSLT(nn.Module):
    """Exact SignJoey stack with either global SLT or GASLT encoder attention."""

    def __init__(self, feat_dim: int, vocab_size: int, cfg):
        super().__init__()
        if cfg.model_arch not in ("signjoey_slt", "gaslt"):
            raise ValueError(f"ReferencePoseSLT cannot build model_arch={cfg.model_arch!r}")
        d = cfg.d_model
        interface_dim = int(cfg.visual_feature_dim) or feat_dim
        # Not specified by Auslan-Daily: makes the raw 204-D body+hands pose
        # compatible with the requested/public 1,024-D visual input interface.
        self.pose_adapter = (nn.Linear(feat_dim, interface_dim)
                             if feat_dim != interface_dim else nn.Identity())
        self.spatial_embedding = SpatialEmbedding(
            interface_dim, d, cfg.source_embedding_activation)
        self.source_position = SinusoidalPosition(d, cfg.position_max_len)
        self.source_dropout = nn.Dropout(cfg.embedding_dropout)
        attention = "gloss" if cfg.model_arch == "gaslt" else "global"
        self.encoder_layers = nn.ModuleList([
            ReferenceEncoderLayer(d, cfg.dim_feedforward, cfg.nhead, cfg.dropout,
                                  attention, cfg.gaslt_num_keys)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d, eps=1e-6)

        self.word_embedding = WordEmbedding(
            vocab_size, cfg.target_embedding_dim, d, cfg.target_embedding_activation)
        self.target_position = SinusoidalPosition(d, cfg.position_max_len)
        self.target_dropout = nn.Dropout(cfg.embedding_dropout)
        self.decoder_layers = nn.ModuleList([
            ReferenceDecoderLayer(d, cfg.dim_feedforward, cfg.nhead, cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d, eps=1e-6)
        self.generator = nn.Linear(d, vocab_size, bias=False)
        self.d_model = d
        self.architecture = cfg.model_arch
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
        # SignJoey re-zeros this row after Xavier initialisation.
        with torch.no_grad():
            self.word_embedding.lookup.weight[PAD].zero_()

    def encode(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        valid = ~src_pad
        x = self.pose_adapter(src)
        x = self.spatial_embedding(x, valid)
        x = self.source_dropout(self.source_position(x))
        src_valid = valid.unsqueeze(1)
        for layer in self.encoder_layers:
            x = layer(x, src_valid)
        return self.encoder_norm(x)

    def decode_step(self, tgt_in: torch.Tensor, memory: torch.Tensor,
                    src_pad: torch.Tensor, tgt_pad: Optional[torch.Tensor]):
        valid = ~tgt_in.eq(PAD) if tgt_pad is None else ~tgt_pad
        x = self.word_embedding(tgt_in, valid)
        x = self.target_dropout(self.target_position(x))
        length = tgt_in.size(1)
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=tgt_in.device).tril()
        tgt_valid = valid.unsqueeze(1) & causal.unsqueeze(0)
        src_valid = (~src_pad).unsqueeze(1)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_valid, tgt_valid)
        return self.generator(self.decoder_norm(x))

    def forward(self, src, src_pad, tgt_in, tgt_pad, return_memory: bool = False):
        memory = self.encode(src, src_pad)
        logits = self.decode_step(tgt_in, memory, src_pad, tgt_pad)
        return (logits, memory) if return_memory else logits

    @torch.no_grad()
    def greedy_decode(self, src, src_pad, max_len: int = 64) -> torch.Tensor:
        self.eval()
        batch = src.size(0)
        memory = self.encode(src, src_pad)
        output = torch.full((batch, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(batch, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.decode_step(output, memory, src_pad, None)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            next_token[done] = PAD
            output = torch.cat((output, next_token), dim=1)
            done |= next_token.squeeze(1).eq(EOS)
            if bool(done.all()):
                break
        return output

    @torch.no_grad()
    def beam_decode(self, src, src_pad, max_len: int = 64, beam: int = 5,
                    length_penalty: float = 0.6):
        # Keep the existing project's beam implementation/API so decoding
        # comparisons isolate architecture rather than two beam-search programs.
        self.eval()
        device = src.device
        outputs = []
        for b in range(src.size(0)):
            memory = self.encode(src[b:b + 1], src_pad[b:b + 1]).expand(
                beam, -1, -1).contiguous()
            source_pad = src_pad[b:b + 1].expand(beam, -1).contiguous()
            tokens = torch.full((beam, 1), BOS, dtype=torch.long, device=device)
            scores = torch.full((beam,), float("-inf"), device=device)
            scores[0] = 0.0
            finished = []
            for _ in range(max_len - 1):
                logp = torch.log_softmax(
                    self.decode_step(tokens, memory, source_pad, None)[:, -1], dim=-1)
                vocab = logp.size(-1)
                top_scores, indices = (scores[:, None] + logp).reshape(-1).topk(beam)
                parents = torch.div(indices, vocab, rounding_mode="floor")
                next_tokens = indices % vocab
                tokens = torch.cat((tokens[parents], next_tokens[:, None]), dim=1)
                scores = top_scores.clone()
                for i in range(beam):
                    if next_tokens[i].item() == EOS:
                        seq = tokens[i].tolist()
                        finished.append((scores[i].item() /
                                         (max(1, len(seq)) ** length_penalty), seq))
                        scores[i] = float("-inf")
                if len(finished) >= beam:
                    break
            if not finished:
                finished.append((0.0, tokens[int(scores.argmax())].tolist()))
            outputs.append(max(finished, key=lambda pair: pair[0])[1])
        return outputs


def gaslt_similarity_loss(memory: torch.Tensor, src_pad: torch.Tensor,
                          target_similarity: torch.Tensor, weight: float) -> torch.Tensor:
    """GASLT knowledge-transfer loss over upper-triangular batch pairs."""
    valid = (~src_pad).unsqueeze(-1)
    pooled = (memory.masked_fill(~valid, 0).sum(1)
              / valid.sum(1).clamp_min(1))
    pooled = F.normalize(pooled, dim=-1)
    predicted = pooled @ pooled.t()
    pair_mask = torch.ones_like(predicted, dtype=torch.bool).triu(1)
    if not bool(pair_mask.any()):
        return memory.new_zeros(())
    return F.mse_loss(predicted[pair_mask], target_similarity[pair_mask]) * weight


def build_model(feat_dim: int, vocab_size: int, cfg) -> nn.Module:
    """Model factory that preserves all legacy checkpoints by default."""
    if cfg.model_arch == "generic":
        from .model import PoseSLT
        return PoseSLT(feat_dim, vocab_size, cfg)
    return ReferencePoseSLT(feat_dim, vocab_size, cfg)
