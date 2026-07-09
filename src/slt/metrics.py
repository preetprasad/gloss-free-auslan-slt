"""Evaluation metrics: BLEU-4 (sacreBLEU) and WER (jiwer).

BLEU-4 is the Phase 2 decision-gate metric (research gap G3: no WER/BLEU
baseline on Auslan-Daily). WER is reported alongside for completeness.
"""
from __future__ import annotations

from typing import List

import sacrebleu


def corpus_bleu4(hyps: List[str], refs: List[str]) -> float:
    """Corpus BLEU-4 in the 0-100 range (sacreBLEU default is 4-gram)."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score)


def corpus_wer(hyps: List[str], refs: List[str]) -> float:
    """Corpus word error rate (0-1). Lower is better."""
    import jiwer

    hyps2 = [h if h.strip() else "<empty>" for h in hyps]
    refs2 = [r if r.strip() else "<empty>" for r in refs]
    return float(jiwer.wer(refs2, hyps2))


def decision_gate(bleu4: float, threshold: float) -> dict:
    """Phase 2 BLEU-4 gate. Returns a structured, loggable verdict."""
    passed = bleu4 >= threshold
    return {
        "metric": "BLEU-4",
        "value": round(bleu4, 3),
        "threshold": threshold,
        "passed": passed,
        "verdict": (
            f"PASS — gloss-free pose baseline reached BLEU-4 {bleu4:.2f} "
            f"(>= {threshold}); proceed to scale/transfer-learning."
            if passed else
            f"BELOW GATE — BLEU-4 {bleu4:.2f} (< {threshold}); revisit inputs / "
            f"pretraining before scaling."
        ),
    }
