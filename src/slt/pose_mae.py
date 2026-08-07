"""Same-detector multi-stream masked pose pretraining for Auslan-Daily.

The design is a locally-scaled adaptation of MS-MAE:

* HALPE body, left hand and right hand are separate streams;
* four-frame non-overlapping temporal cubelets are embedded independently;
* different time tokens are masked in each stream (DiffMask);
* a shared Transformer models within- and cross-stream dependencies; and
* reconstruction loss is evaluated only on artificially masked, naturally
  observed keypoints, so AlphaPose failures are not learned as targets.

The feature extractor returns one 1,024-D feature per temporal cubelet.  Those
features can be frozen and passed to the exact 3+3 SL-Transformer, making the
downstream comparison directly analogous to the completed GloFE experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Compact order produced by Config(keypoint_subset="body_hands").
REGION_SLICES = {
    "body": slice(0, 26),
    "left_hand": slice(26, 47),
    "right_hand": slice(47, 68),
}


@dataclass
class PoseMAEOutput:
    loss: torch.Tensor
    region_rmse: Dict[str, float]
    observed_targets: Dict[str, int]
    masked_tokens: Dict[str, int]


def _sine_position(length: int, width: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


def make_diff_masks(patch_valid: torch.Tensor, streams: int,
                    ratio: float, seed: Optional[int] = None) -> torch.Tensor:
    """Return independent fixed-count masks with shape ``(B, streams, N)``.

    Sampling on CPU makes deterministic validation masks device-independent and
    avoids generator-device differences between CPU, CUDA and Apple MPS.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must be in (0,1), got {ratio}")
    batch, tokens = patch_valid.shape
    masks = torch.zeros(batch, streams, tokens, dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    valid_cpu = patch_valid.detach().cpu()
    for b in range(batch):
        indices = valid_cpu[b].nonzero(as_tuple=True)[0]
        if not len(indices):
            continue
        count = max(1, min(len(indices), int(round(len(indices) * ratio))))
        for stream in range(streams):
            order = torch.randperm(len(indices), generator=generator)[:count]
            masks[b, stream, indices[order]] = True
    return masks.to(device=patch_valid.device)


class MultiStreamPoseMAE(nn.Module):
    """Three-stream temporal masked autoencoder with a 1,024-D export path."""

    def __init__(self, hidden: int = 256, encoder_layers: int = 4,
                 heads: int = 8, ff_size: int = 1024,
                 decoder_hidden: int = 256, decoder_layers: int = 2,
                 patch_size: int = 4, dropout: float = 0.1,
                 output_dim: int = 1024, min_confidence: float = 0.1):
        super().__init__()
        if hidden % heads or decoder_hidden % heads:
            raise ValueError("hidden and decoder_hidden must be divisible by heads")
        self.names = tuple(REGION_SLICES)
        self.patch_size = int(patch_size)
        self.hidden = int(hidden)
        self.output_dim = int(output_dim)
        self.min_confidence = float(min_confidence)
        self.region_sizes = {
            name: region.stop - region.start for name, region in REGION_SLICES.items()
        }

        self.patch_embeddings = nn.ModuleDict({
            name: nn.Conv1d(size * 3, hidden, kernel_size=self.patch_size,
                            stride=self.patch_size)
            for name, size in self.region_sizes.items()
        })
        self.mask_tokens = nn.Parameter(torch.zeros(len(self.names), 1, 1, hidden))
        self.stream_embedding = nn.Parameter(torch.zeros(1, len(self.names), 1, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, heads, ff_size, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(
            encoder_layer, encoder_layers, norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False)

        self.decoder_input = nn.Linear(hidden, decoder_hidden)
        decoder_layer = nn.TransformerEncoderLayer(
            decoder_hidden, heads, max(decoder_hidden * 4, ff_size // 2),
            dropout, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(
            decoder_layer, decoder_layers, norm=nn.LayerNorm(decoder_hidden),
            enable_nested_tensor=False)
        self.reconstruction_heads = nn.ModuleDict({
            name: nn.Linear(decoder_hidden, self.patch_size * size * 2)
            for name, size in self.region_sizes.items()
        })
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden * len(self.names), output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mask_tokens, std=0.02)
        nn.init.normal_(self.stream_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pad(self, source: torch.Tensor, source_pad: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        remainder = source.size(1) % self.patch_size
        extra = 0 if remainder == 0 else self.patch_size - remainder
        if extra:
            source = F.pad(source, (0, 0, 0, extra))
            source_pad = F.pad(source_pad, (0, extra), value=True)
        return source, source_pad

    def _tokenize(self, source: torch.Tensor, source_pad: torch.Tensor,
                  masks: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, source_pad = self._pad(source, source_pad)
        if source.size(-1) != 68 * 3:
            raise ValueError(
                f"expected body_hands HALPE features with width 204, got {source.shape}")
        batch, frames, _ = source.shape
        pose = source.reshape(batch, frames, 68, 3)
        patch_valid = (~source_pad).reshape(
            batch, frames // self.patch_size, self.patch_size).any(-1)
        positions = _sine_position(
            patch_valid.size(1), self.hidden, source.device, source.dtype)

        streams = []
        for stream, name in enumerate(self.names):
            region = pose[:, :, REGION_SLICES[name], :].reshape(batch, frames, -1)
            token = self.patch_embeddings[name](region.transpose(1, 2)).transpose(1, 2)
            if masks is not None:
                replacement = self.mask_tokens[stream].expand(batch, token.size(1), -1)
                token = torch.where(masks[:, stream].unsqueeze(-1), replacement, token)
            # Masked tokens still need their time and region identity; without
            # these embeddings the decoder can only learn an unconditional mean.
            token = token + positions.unsqueeze(0) + self.stream_embedding[:, stream]
            token = token.masked_fill(~patch_valid.unsqueeze(-1), 0.0)
            streams.append(token)
        tokens = torch.cat(streams, dim=1)
        token_pad = torch.cat([~patch_valid] * len(self.names), dim=1)
        return tokens, token_pad, pose

    def encode(self, source: torch.Tensor, source_pad: torch.Tensor,
               masks: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, token_pad, pose = self._tokenize(source, source_pad, masks)
        encoded = self.encoder(tokens, src_key_padding_mask=token_pad)
        return encoded, token_pad, pose

    def extract_features(self, source: torch.Tensor, source_pad: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return frozen downstream features and their padding mask."""
        encoded, token_pad, _ = self.encode(source, source_pad)
        count = encoded.size(1) // len(self.names)
        streams = [encoded[:, i * count:(i + 1) * count]
                   for i in range(len(self.names))]
        features = self.feature_fusion(torch.cat(streams, dim=-1))
        patch_pad = token_pad[:, :count]
        features = features.masked_fill(patch_pad.unsqueeze(-1), 0.0)
        return features, patch_pad

    def forward(self, source: torch.Tensor, source_pad: torch.Tensor,
                mask_ratio: float = 0.5, mask_seed: Optional[int] = None
                ) -> PoseMAEOutput:
        padded_source, padded_pad = self._pad(source, source_pad)
        batch, frames, _ = padded_source.shape
        patch_valid = (~padded_pad).reshape(
            batch, frames // self.patch_size, self.patch_size).any(-1)
        masks = make_diff_masks(patch_valid, len(self.names), mask_ratio, mask_seed)
        encoded, token_pad, pose = self.encode(source, source_pad, masks)
        decoded = self.decoder(
            self.decoder_input(encoded), src_key_padding_mask=token_pad)
        tokens = patch_valid.size(1)

        losses = []
        region_rmse: Dict[str, float] = {}
        observed_targets: Dict[str, int] = {}
        masked_tokens: Dict[str, int] = {}
        for stream, name in enumerate(self.names):
            size = self.region_sizes[name]
            part = pose[:, :, REGION_SLICES[name], :]
            target = part[..., :2].reshape(
                batch, tokens, self.patch_size, size, 2)
            naturally_observed = (
                part[..., 2] >= self.min_confidence
            ).reshape(batch, tokens, self.patch_size, size)
            naturally_observed &= (~padded_pad).reshape(
                batch, tokens, self.patch_size).unsqueeze(-1)

            start = stream * tokens
            state = decoded[:, start:start + tokens]
            predicted = self.reconstruction_heads[name](state).reshape_as(target)
            selected = masks[:, stream, :, None, None] & naturally_observed
            count = int(selected.sum().item())
            observed_targets[name] = count
            masked_tokens[name] = int(masks[:, stream].sum().item())
            if count:
                squared = (predicted - target).square()
                loss = squared.masked_select(selected.unsqueeze(-1)).mean()
                losses.append(loss)
                region_rmse[name] = float(loss.detach().sqrt().cpu())
            else:
                region_rmse[name] = 0.0
        if not losses:
            raise RuntimeError("masked batch contains no naturally observed reconstruction targets")
        return PoseMAEOutput(
            loss=torch.stack(losses).mean(), region_rmse=region_rmse,
            observed_targets=observed_targets, masked_tokens=masked_tokens)
