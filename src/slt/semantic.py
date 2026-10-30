"""Frozen text-semantic targets for auxiliary SLT supervision."""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSimilarityBank:
    """Text embedding lookup that returns per-batch cosine-similarity labels."""

    def __init__(self, texts: Iterable[str], model_name: str, cache_path: str = ""):
        unique = sorted(set(texts))
        cached = {}
        if cache_path and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("model_name") == model_name:
                cached = payload.get("embeddings", {})
        missing = [text for text in unique if text not in cached]
        if missing:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device="cpu")
            encoded = model.encode(missing, batch_size=64, show_progress_bar=True,
                                   convert_to_tensor=True, normalize_embeddings=True)
            cached.update({text: vector.cpu() for text, vector in zip(missing, encoded)})
            if cache_path:
                parent = os.path.dirname(os.path.abspath(cache_path))
                os.makedirs(parent, exist_ok=True)
                torch.save({"model_name": model_name, "embeddings": cached}, cache_path)
        self.embeddings = cached

    def similarity(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        vectors = torch.stack([self.embeddings[text] for text in texts]).to(device)
        vectors = F.normalize(vectors, dim=-1)
        return vectors @ vectors.t()


class SentenceEmbeddingBank:
    """Cache frozen sentence embeddings and retrieve them by target sentence.

    The implementation deliberately uses Transformers' public AutoModel API
    rather than making the training environment depend on sentence-transformers.
    ``sentence-transformers/*`` encoders expose a standard Transformer followed
    by attention-mask mean pooling, which is reproduced below.  Embeddings are
    normalised so the visual alignment objective has a stable scale.
    """

    def __init__(self, texts: Iterable[str], model_name: str, cache_path: str = ""):
        unique = sorted(set(texts))
        cached: dict[str, torch.Tensor] = {}
        if cache_path and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("model_name") == model_name:
                cached = payload.get("sentence_embeddings", {})
        missing = [text for text in unique if text not in cached]
        if missing:
            cached.update(self._encode(missing, model_name))
            if cache_path:
                parent = os.path.dirname(os.path.abspath(cache_path))
                os.makedirs(parent, exist_ok=True)
                torch.save({"model_name": model_name, "sentence_embeddings": cached}, cache_path)
        if not cached:
            raise ValueError("SentenceEmbeddingBank requires at least one target sentence")
        self.embeddings = cached
        self.embedding_dim = int(next(iter(cached.values())).numel())

    @staticmethod
    @torch.no_grad()
    def _encode(texts: Sequence[str], model_name: str) -> dict[str, torch.Tensor]:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cpu().eval()
        result: dict[str, torch.Tensor] = {}
        for start in range(0, len(texts), 64):
            batch = list(texts[start:start + 64])
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128,
                                return_tensors="pt")
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = F.normalize(pooled, dim=-1).cpu()
            result.update({text: vector for text, vector in zip(batch, pooled)})
        return result

    def lookup(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        return torch.stack([self.embeddings[text] for text in texts]).to(device, non_blocking=True)


class SentenceProjectionHead(nn.Module):
    """Attention-pool encoder memory into a normalised sentence-semantic vector."""

    def __init__(self, model_dim: int, semantic_dim: int):
        super().__init__()
        self.score = nn.Linear(model_dim, 1)
        self.project = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, semantic_dim),
        )

    def forward(self, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        scores = self.score(memory).squeeze(-1).masked_fill(pad, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return F.normalize(self.project(pooled), dim=-1)


def sentence_alignment_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared L2 sentence loss, averaged across examples (not dimensions)."""
    return (predicted - target).pow(2).sum(dim=-1).mean()
