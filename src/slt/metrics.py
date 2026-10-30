"""Evaluation metrics: BLEU-4 (sacreBLEU) and WER (jiwer).

BLEU-4 is the Phase 2 decision-gate metric (research gap G3: no WER/BLEU
baseline on Auslan-Daily). WER is reported alongside for completeness.
"""
from __future__ import annotations

import re
from typing import List

import sacrebleu

# Light detokeniser: undo the Auslan-Daily pre-tokenisation ("hello ." -> "hello.")
# so sacreBLEU's standard 13a tokeniser (which assumes natural detokenised text)
# re-splits both sides consistently and the score is comparable to the literature.
# Applied symmetrically to hyps and refs, so exact rules matter little.
_DETOK_PUNCT = re.compile(r"\s+([.,!?;:%)\]}])")
_DETOK_OPEN = re.compile(r"([(\[{])\s+")
_DETOK_APOS = re.compile(r"\s+'\s*")


def detokenize(text: str) -> str:
    text = _DETOK_PUNCT.sub(r"\1", text)
    text = _DETOK_OPEN.sub(r"\1", text)
    text = _DETOK_APOS.sub("'", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def corpus_bleu4(hyps: List[str], refs: List[str], tokenize: str = "13a") -> float:
    """Corpus BLEU-4 in the 0-100 range (sacreBLEU default is 4-gram).

    ``tokenize`` policy matters and must be reported. Auslan-Daily's `Subtitle`
    text is **already tokenised** (e.g. ``"hello ."`` with a spaced period), and
    our hypotheses are decoded from a vocabulary built over that same tokenised
    text — so both sides live in the same token space. sacreBLEU's default
    ``13a`` would tokenise *again*, shifting n-gram counts and producing a score
    that is not comparable to anything.

    * ``"13a"`` (default now, per supervisor decision) — sacreBLEU's standard
      tokeniser. Requires hyps AND refs to be DETOKENISED first (see
      ``detokenize`` / ``slt.engine``); gives numbers comparable to published BLEU.
    * ``"none"`` — whitespace split; internally consistent but NOT comparable to
      the literature. Only for our-vs-our sanity checks.

    Do NOT "fix" the pre-tokenisation warning with ``force=True``: that silences
    the message without addressing comparability.
    """
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    return float(bleu.score)


def bleu_signature(tokenize: str = "13a", detok: bool = True) -> str:
    """Human-readable record of how BLEU was computed (log this with results)."""
    return f"sacreBLEU|4-gram|tokenize={tokenize}|detok={detok}|smooth=exp"


def corpus_wer(hyps: List[str], refs: List[str]) -> float:
    """Corpus word error rate (0-1). Lower is better."""
    import jiwer

    hyps2 = [h if h.strip() else "<empty>" for h in hyps]
    refs2 = [r if r.strip() else "<empty>" for r in refs]
    return float(jiwer.wer(refs2, hyps2))


def decision_gate(bleu4: float, threshold: float, basis: str = "") -> dict:
    """Phase 2 BLEU-4 gate. Returns a structured, loggable verdict.

    `basis` documents how the threshold was derived (e.g. reference-relative), per
    Atif's decision that the gate is a fraction of a reference model's BLEU-4.
    """
    passed = bleu4 >= threshold
    b = f" [{basis}]" if basis else ""
    return {
        "metric": "BLEU-4",
        "value": round(bleu4, 3),
        "threshold": round(threshold, 3),
        "basis": basis,
        "passed": passed,
        "verdict": (
            f"PASS — gloss-free pose baseline reached BLEU-4 {bleu4:.2f} "
            f"(>= {threshold:.2f}{b}); proceed to scale/transfer-learning."
            if passed else
            f"BELOW GATE — BLEU-4 {bleu4:.2f} (< {threshold:.2f}{b}); revisit inputs / "
            f"pretraining before scaling."
        ),
    }
