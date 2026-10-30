"""Precompute frozen text sentence targets for a validation-only SLT experiment."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slt.semantic import SentenceEmbeddingBank


def texts_from_manifest(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[1] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    texts = texts_from_manifest(args.manifest)
    bank = SentenceEmbeddingBank(texts, args.model, args.cache)
    print(f"[sentence-targets] examples={len(texts)} unique={len(bank.embeddings)} "
          f"dim={bank.embedding_dim} cache={args.cache}", flush=True)


if __name__ == "__main__":
    main()
