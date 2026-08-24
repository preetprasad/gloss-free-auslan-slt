"""A small feature-level temporal-motion adapter for cached RGB sign features.

This is inspired by temporal-pyramid SLT work, but is intentionally *not* a
claim of an exact TSPNet/OLMD reproduction.  It operates on existing frozen
DINO frame vectors, where spatial-map operations from raw-video methods are
not available.  The adapter learns short-, medium-, and long-receptive-field
motion residuals with inexpensive depthwise temporal convolutions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPyramidAdapter(nn.Module):
    """Masked residual adapter with multi-scale temporal motion branches.

    Input and output are ``(batch, time, d_model)``.  A low initial fusion gate
    makes the module close to identity at the start of fine-tuning, which is
    useful when the base pose encoder is warm-started from VLP.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.motion = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation,
                          dilation=dilation, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        width = d_model * (1 + len(dilations))
        self.fuse = nn.Sequential(nn.Linear(width, d_model), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.gate.bias, -2.0)  # begin near the untouched RGB representation

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        valid = (~pad).unsqueeze(-1).to(dtype=x.dtype)
        x = x * valid
        previous = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        # Pair validity stops padding and the sequence's first frame from
        # injecting false motion into the residual branch.
        previous_valid = torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
        delta = (x - previous) * valid * previous_valid
        motion = self.motion(delta)
        base = (x + motion) * valid
        temporal = base.transpose(1, 2)
        branches = [branch(temporal).transpose(1, 2) * valid for branch in self.branches]
        candidate = self.fuse(torch.cat([base, *branches], dim=-1)) * valid
        gate = torch.sigmoid(self.gate(torch.cat([x, candidate], dim=-1))) * valid
        return self.norm(x + gate * candidate) * valid


class TemporalLiftPool(nn.Module):
    """Masked, learnable 2x temporal reduction for cached feature sequences.

    This is a small *TLP-inspired* adapter rather than an implementation claim
    of the raw-video CSLR method. It starts as pairwise averaging, then learns
    whether each local temporal difference should be retained.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.predict_residual = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )
        self.update_residual = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )
        self.detail_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Begin as pairwise averaging; only learn deviations after the warm start.
        for module in (self.predict_residual[-1], self.update_residual[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.detail_gate.bias, -2.0)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ~pad
        if x.size(1) % 2:
            x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
            valid = torch.cat([valid, torch.zeros_like(valid[:, :1])], dim=1)
        even, odd = x[:, 0::2], x[:, 1::2]
        even_valid, odd_valid = valid[:, 0::2], valid[:, 1::2]
        # A final unpaired frame passes through unchanged. For full pairs,
        # identity prediction plus a 1/2 update yields the local average.
        odd_filled = torch.where(odd_valid.unsqueeze(-1), odd, even)
        detail = (odd_filled - (even + self.predict_residual(even))) * odd_valid.unsqueeze(-1)
        coarse = even + 0.5 * detail + self.update_residual(detail)
        gate = torch.sigmoid(self.detail_gate(torch.cat([coarse, detail], dim=-1)))
        pooled = self.norm(coarse + gate * detail) * even_valid.unsqueeze(-1)
        return pooled, ~even_valid
