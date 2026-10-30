"""RGB-fusion model variants with an explicit feature-level temporal adapter."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import Config
from slt.rgb_fusion import RGBFusionSLT
from slt.temporal_pyramid import TemporalLiftPool, TemporalPyramidAdapter


class TemporalRGBFusionSLT(RGBFusionSLT):
    """Existing RGB fusion plus an optional post-RGB temporal adapter."""

    def __init__(self, feat_dim: int, rgb_dim: int, vocab_size: int, cfg: Config, adapter: str):
        super().__init__(feat_dim, rgb_dim, vocab_size, cfg)
        if adapter not in {"none", "pyramid", "pyramid_pool"}:
            raise ValueError(f"unknown temporal adapter {adapter!r}")
        self.temporal_adapter_name = adapter
        self.temporal_adapter = (TemporalPyramidAdapter(cfg.d_model, cfg.dropout)
                                 if adapter in {"pyramid", "pyramid_pool"} else nn.Identity())
        self.pose_pool = TemporalLiftPool(cfg.d_model, cfg.dropout) if adapter == "pyramid_pool" else None
        self.rgb_pool = TemporalLiftPool(cfg.d_model, cfg.dropout) if adapter == "pyramid_pool" else None

    def encode_rgb(self, rgb: torch.Tensor, rgb_pad: torch.Tensor) -> torch.Tensor:
        memory = super().encode_rgb(rgb, rgb_pad)
        return self.temporal_adapter(memory, rgb_pad) if self.temporal_adapter_name != "none" else memory

    def _fuse(self, mem_p, src_pad, mem_r, rgb_pad):
        """Pool only decoder memory; CTC keeps the full pose sequence."""
        if self.temporal_adapter_name == "pyramid_pool":
            mem_p, src_pad = self.pose_pool(mem_p, src_pad)
            mem_r, rgb_pad = self.rgb_pool(mem_r, rgb_pad)
        return super()._fuse(mem_p, src_pad, mem_r, rgb_pad)
