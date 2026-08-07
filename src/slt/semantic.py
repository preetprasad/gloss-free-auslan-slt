"""Sentence-BERT similarity labels for GASLT knowledge transfer."""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch
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
