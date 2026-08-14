"""VQ tokeniser for sign language features (pose and/or RGB).

Stage A: learns a discrete codebook over per-frame visual features.
Architecture:
  Features (B,T,F) -> Encoder (small Transformer/Conv) -> Latents (B,T,D)
    -> VQ (EMA or gradient codebook) -> Quantized (B,T,D) -> Decoder -> Reconstruction
Output: discrete code sequences (B,T) or (B,T, num_codebooks) for residual VQ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from config import Config


@dataclass
class VQOutput:
    quantized: torch.Tensor          # (B, T, D)
    codes: torch.Tensor              # (B, T) or (B, T, K) for residual VQ
    codebook_loss: torch.Tensor      # scalar
    commitment_loss: torch.Tensor    # scalar
    perplexity: torch.Tensor         # scalar


class VectorQuantizerEMA(nn.Module):
    """VQ-VAE with EMA codebook updates (Sonic/VQ-VAE-2 style)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99,
                 eps: float = 1e-5, reset_threshold: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        # Dead codes (cluster_size below this after EMA) are revived from live latents.
        self.reset_threshold = reset_threshold

        # Codebook (buffer, EMA-updated). Unit-norm init -> cosine VQ (anti-collapse).
        emb = F.normalize(torch.randn(num_embeddings, embedding_dim), dim=-1)
        self.register_buffer("embeddings", emb)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", emb.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, T, D) latent features
        Returns: quantized, codes, perplexity

        Cosine VQ: latents and codebook are L2-normalised before the nearest-code
        search, so the encoder cannot collapse magnitude to force a single code.
        Dead codes are periodically re-seeded from live latents.
        """
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)                    # (B*T, D)
        fz = F.normalize(flat_z, dim=-1)
        emb = F.normalize(self.embeddings, dim=-1)
        # cosine similarity -> nearest = argmax; distance = -sim
        sim = fz @ emb.t()                           # (B*T, K)
        codes = sim.argmax(dim=-1)                   # (B*T,)
        quantized = self.embeddings[codes].view(B, T, D)

        encodings = F.one_hot(codes, self.num_embeddings).float()   # (B*T, K)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        if self.training and self.decay > 0:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
                embed_sum = fz.t() @ encodings       # accumulate NORMALISED latents (D, K)
                self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = ((self.cluster_size + self.eps)
                                / (n + self.num_embeddings * self.eps) * n)
                self.embeddings.copy_(F.normalize(self.embed_avg / cluster_size.unsqueeze(1), dim=-1))
                # dead-code revival: re-seed unused codes from random live latents
                dead = self.cluster_size < self.reset_threshold
                n_dead = int(dead.sum())
                if n_dead > 0 and fz.size(0) > 0:
                    idx = torch.randint(0, fz.size(0), (n_dead,), device=fz.device)
                    seeds = fz[idx]
                    self.embeddings[dead] = seeds
                    self.embed_avg[dead] = seeds
                    self.cluster_size[dead] = 1.0

        return quantized, codes.view(B, T), perplexity


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ (RVQ) with multiple codebooks for finer quantisation."""

    def __init__(
        self,
        num_codebooks: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        shared_codebook: bool = False,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, decay)
            for _ in range(num_codebooks)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, T, D)
        Returns: quantized (B, T, D), codes (B, T, K), avg_perplexity
        """
        residual = z
        quantized_total = torch.zeros_like(z)
        all_codes = []
        perplexities = []

        for cb in self.codebooks:
            q, codes, perp = cb(residual)
            quantized_total = quantized_total + q
            residual = residual - q
            all_codes.append(codes)
            perplexities.append(perp)

        codes = torch.stack(all_codes, dim=-1)  # (B, T, K)
        avg_perplexity = torch.stack(perplexities).mean()
        return quantized_total, codes, avg_perplexity


class VQEncoder(nn.Module):
    """Small temporal encoder: projects features -> latent dim."""

    def __init__(self, feat_dim: int, latent_dim: int, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), src_pad: (B, T) True=pad
        x = self.input_proj(x)
        return self.encoder(x, src_key_padding_mask=src_pad)


class VQDecoder(nn.Module):
    """Decoder: latent -> reconstruction."""

    def __init__(self, latent_dim: int, feat_dim: int, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, feat_dim)

    def forward(
        self,
        quantized: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_pad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Simple: use quantized as memory, reconstruct autoregressively or in parallel
        # For reconstruction, we can use a causal decoder over the quantized sequence
        B, T, D = quantized.shape
        # Use quantized as both memory and target (teacher-forced reconstruction)
        tgt = quantized
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=quantized.device)
        out = self.decoder(
            tgt, quantized,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.output_proj(out)


class VQTokeniser(nn.Module):
    """Full VQ tokeniser: encoder -> VQ -> decoder."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        feat_dim = self._get_feat_dim()
        latent_dim = cfg.vq_code_dim
        num_codebooks = cfg.vq_num_codebooks
        codebook_size = cfg.vq_codebook_size
        decay = cfg.vq_ema_decay

        self.encoder = VQEncoder(feat_dim, latent_dim, num_layers=3, nhead=8)
        if num_codebooks > 1:
            self.vq = ResidualVectorQuantizer(num_codebooks, codebook_size, latent_dim, decay)
        else:
            self.vq = VectorQuantizerEMA(codebook_size, latent_dim, decay)
        self.decoder = VQDecoder(latent_dim, feat_dim, num_layers=3, nhead=8)

        # Optional: CTC head for linguistic alignment
        if cfg.vq_ctc_weight > 0:
            self.ctc_head = nn.Linear(latent_dim, 1000)  # vocab size placeholder

    def _get_feat_dim(self) -> int:
        # Use config's feature_dim() which correctly computes from num_keypoints,
        # use_confidence, use_velocity, etc. This works for both synthetic (HALPE-136)
        # and real (DWPose) data.
        return self.cfg.feature_dim()

    def forward(
        self,
        feats: torch.Tensor,
        src_pad: torch.Tensor,
        return_reconstruction: bool = True,
    ) -> dict:
        """
        feats: (B, T, F)
        src_pad: (B, T) True=pad
        Returns dict with: quantized, codes, losses, reconstruction, perplexity
        """
        latent = self.encoder(feats, src_pad)
        quantized, codes, perplexity = self.vq(latent)

        # Losses
        commitment_loss = F.mse_loss(quantized.detach(), latent)
        codebook_loss = F.mse_loss(quantized, latent.detach())
        vq_loss = codebook_loss + self.cfg.vq_commitment_cost * commitment_loss

        # Straight-through estimator: copy the quantised value on the forward pass
        # but let reconstruction gradients flow to the ENCODER (otherwise the encoder
        # is trained only by the commitment term -> collapse).
        quantized_st = latent + (quantized - latent).detach()

        out = {
            "quantized": quantized_st,
            "codes": codes,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
        }

        if return_reconstruction:
            recon = self.decoder(quantized_st, src_pad)
            recon_loss = F.mse_loss(recon, feats)
            out["reconstruction"] = recon
            out["recon_loss"] = recon_loss

        return out

    def encode_to_codes(self, feats: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        """Encode features to discrete codes only (for Stage B)."""
        with torch.no_grad():
            latent = self.encoder(feats, src_pad)
            _, codes, _ = self.vq(latent)
        return codes  # (B, T) or (B, T, K)

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to quantized latents."""
        if self.cfg.vq_num_codebooks > 1:
            # Residual VQ: sum codebook embeddings
            quantized = torch.zeros(codes.shape[0], codes.shape[1], self.cfg.vq_code_dim,
                                    device=codes.device, dtype=codes.dtype)
            for k, cb in enumerate(self.vq.codebooks):
                quantized = quantized + cb.embeddings[codes[:, :, k]]
        else:
            quantized = self.vq.embeddings[codes]
        return quantized


def build_vq_tokeniser(cfg: Config) -> VQTokeniser:
    return VQTokeniser(cfg)