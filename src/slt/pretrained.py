"""Dependency-light loading of BPEmb word2vec binary target embeddings."""
from __future__ import annotations

import os

import numpy as np
import torch


def _read_token(handle) -> str:
    token = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise EOFError("unexpected end of word2vec file")
        if byte == b" ":
            return token.decode("utf-8")
        if byte not in (b"\n", b"\r"):
            token.extend(byte)


def load_selected_word2vec_binary(path: str, tokens: set[str]):
    """Read only requested rows from the original C word2vec binary format."""
    selected = {}
    with open(path, "rb") as handle:
        count, dimension = (int(value) for value in handle.readline().split())
        width = 4 * dimension
        for _ in range(count):
            token = _read_token(handle)
            raw = handle.read(width)
            if len(raw) != width:
                raise EOFError(f"truncated vector for {token!r}")
            if token in tokens:
                selected[token] = torch.from_numpy(
                    np.frombuffer(raw, dtype="<f4").copy())
            # Some writers add one newline; _read_token also safely skips it.
    return selected, dimension


def initialize_target_embeddings(model, vocab, path: str) -> tuple[int, int]:
    """Copy pretrained rows into ReferencePoseSLT's target lookup by token."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"pretrained target embeddings not found: {path}")
    aliases = {"<bos>": "<s>", "<eos>": "</s>"}
    requested = {aliases.get(token, token) for token in vocab.itos}
    vectors, dimension = load_selected_word2vec_binary(path, requested)
    lookup = model.word_embedding.lookup.weight
    if lookup.size(1) != dimension:
        raise ValueError(
            f"embedding dimension mismatch: model={lookup.size(1)} file={dimension}")
    loaded = 0
    with torch.no_grad():
        for index, token in enumerate(vocab.itos):
            vector = vectors.get(aliases.get(token, token))
            if vector is not None:
                lookup[index].copy_(vector)
                loaded += 1
        lookup[0].zero_()
    return loaded, len(vocab)
