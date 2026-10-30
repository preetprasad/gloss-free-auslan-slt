"""Time-aligned bidirectional gated fusion over the existing pose/RGB memories.

The established fusion simply concatenates pose memory and RGB memory for the
decoder.  This lightweight variant retains the same pose encoder and RGB
temporal-pyramid adapter, but lets the two streams exchange a residual at each
normalised time point before concatenation.  It is an adaptation for cached
vectors, not a claim of reproducing TwoStream's raw-video architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from slt.temporal_rgb_fusion import TemporalRGBFusionSLT


def _align_memory(memory: torch.Tensor, memory_pad: torch.Tensor,
                  destination_pad: torch.Tensor) -> torch.Tensor:
    """Linearly resample each valid sequence onto another stream's valid length."""
    batch, destination_steps = destination_pad.shape
    aligned = memory.new_zeros((batch, destination_steps, memory.size(-1)))
    for index in range(batch):
        source_steps = int((~memory_pad[index]).sum())
        target_steps = int((~destination_pad[index]).sum())
        if source_steps == 0 or target_steps == 0:
            continue
        source = memory[index, :source_steps].transpose(0, 1).unsqueeze(0)
        if source_steps == target_steps:
            current = source
        elif source_steps == 1:
            current = source.expand(-1, -1, target_steps)
        else:
            current = F.interpolate(source, size=target_steps, mode="linear", align_corners=True)
        aligned[index, :target_steps] = current.squeeze(0).transpose(0, 1)
    return aligned


class GatedTemporalRGBFusionSLT(TemporalRGBFusionSLT):
    """Temporal RGB fusion with residual pose↔RGB gated lateral connections."""

    def __init__(self, feat_dim: int, rgb_dim: int, vocab_size: int, cfg: Config):
        super().__init__(feat_dim, rgb_dim, vocab_size, cfg, "pyramid")
        dimension = cfg.d_model
        self.pose_from_rgb = nn.Linear(dimension, dimension, bias=False)
        self.rgb_from_pose = nn.Linear(dimension, dimension, bias=False)
        self.pose_gate = nn.Linear(2 * dimension, dimension)
        self.rgb_gate = nn.Linear(2 * dimension, dimension)
        self.pose_lateral_norm = nn.LayerNorm(dimension)
        self.rgb_lateral_norm = nn.LayerNorm(dimension)
        # Start close to the established late-concatenation baseline; the
        # model must earn use of a cross-modal residual during training.
        nn.init.constant_(self.pose_gate.bias, -2.0)
        nn.init.constant_(self.rgb_gate.bias, -2.0)

    def _fuse(self, mem_p, src_pad, mem_r, rgb_pad):
        rgb_on_pose = _align_memory(mem_r, rgb_pad, src_pad)
        pose_on_rgb = _align_memory(mem_p, src_pad, rgb_pad)
        pose_valid = (~src_pad).unsqueeze(-1).to(mem_p.dtype)
        rgb_valid = (~rgb_pad).unsqueeze(-1).to(mem_r.dtype)
        pose_gate = torch.sigmoid(self.pose_gate(torch.cat([mem_p, rgb_on_pose], dim=-1)))
        rgb_gate = torch.sigmoid(self.rgb_gate(torch.cat([mem_r, pose_on_rgb], dim=-1)))
        enhanced_pose = self.pose_lateral_norm(mem_p + pose_gate * self.pose_from_rgb(rgb_on_pose)) * pose_valid
        enhanced_rgb = self.rgb_lateral_norm(mem_r + rgb_gate * self.rgb_from_pose(pose_on_rgb)) * rgb_valid
        return super()._fuse(enhanced_pose, src_pad, enhanced_rgb, rgb_pad)
