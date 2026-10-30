"""Auditable pseudo-content labels for gloss-free Auslan-Daily experiments.

These are *not* Auslan gloss annotations.  They are deliberately conservative
English content-token sets derived from the translation text.  Keeping this
logic in one place makes the order-invariant and later video-reordered
experiments comparable to the legacy pseudo-CTC recipe.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


PSEUDO_LABEL_VERSION = "content-set-v1"

# Matches the legacy pseudo-CTC filtering rule.  Any improved linguistic rule
# must receive a new version identifier rather than silently changing a run.
STOP_WORDS = set(
    "a an the is are am was were be been being to of in on at and or but i you he she "
    "it we they this that these those my your his her its our their do does did not have "
    "has had will would can could should may might must shall there here then than so as "
    "if for with from up out about into over after . , ? ! ; :".split()
)


def normalise_text(text: str) -> str:
    """Stable text key used in pseudo-label reports and future label maps."""
    return " ".join(text.lower().split())


def content_tokens(text: str) -> list[str]:
    """Return de-duplicated English content tokens in written-text order.

    The order is intentionally retained here for diagnosis only.  The
    order-invariant presence objective treats the returned tokens as a set.
    """
    # Deliberately match the legacy CTC target rule: lower-case whitespace
    # splitting, retain only alphabetic tokens, then remove stop words.  Keeping
    # the label *set* identical lets this experiment isolate order sensitivity.
    words = normalise_text(text).split()
    tokens: list[str] = []
    for word in words:
        if not word.isalpha() or word in STOP_WORDS:
            continue
        if not tokens or tokens[-1] != word:
            tokens.append(word)
    return tokens or ["<g>"]


def build_label_vocab(texts: Iterable[str]) -> dict[str, int]:
    """Build a deterministic zero-based label vocabulary."""
    labels = {"<g>"}
    for text in texts:
        labels.update(content_tokens(text))
    return {label: index for index, label in enumerate(sorted(labels))}


def multi_hot(texts: Iterable[str], label_to_id: dict[str, int]) -> list[list[int]]:
    """Return sorted unique label ids for every text, suitable for a set loss."""
    fallback = label_to_id["<g>"]
    rows: list[list[int]] = []
    for text in texts:
        ids = {label_to_id.get(token, fallback) for token in content_tokens(text)}
        rows.append(sorted(ids))
    return rows


def label_statistics(texts: Iterable[str]) -> dict[str, float | int]:
    """Small, serialisable audit summary for a manifest's pseudo-label set."""
    rows = [content_tokens(text) for text in texts]
    frequencies = Counter(token for row in rows for token in row)
    count = len(rows)
    total = sum(map(len, rows))
    return {
        "pseudo_label_version": PSEUDO_LABEL_VERSION,
        "clips": count,
        "vocabulary": len(frequencies),
        "mean_labels_per_clip": (total / count) if count else 0.0,
        "max_labels_per_clip": max(map(len, rows), default=0),
        "singleton_clips": sum(len(row) == 1 for row in rows),
    }
