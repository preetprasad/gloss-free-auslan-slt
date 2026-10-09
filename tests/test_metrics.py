"""Metric correctness: the conventions that decide whether a reported number means anything.

Every locked value here was measured on the committed headline decode
(results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl), not copied from a write-up.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from slt.metrics import corpus_bleu4, corpus_wer, detokenize  # noqa: E402

DUMP = os.path.join(REPO, "results", "final_ensemble_test_decodes", "hyps_LP=0.3.jsonl")


@pytest.fixture(scope="module")
def headline():
    """(hyps, refs) of the reported operating point, as written by the decoder."""
    if not os.path.exists(DUMP):
        pytest.skip(f"decode dump absent: {DUMP}")
    rows = [json.loads(l) for l in open(DUMP)]
    if rows and str(rows[0]["ref"]).startswith("sha1:"):
        pytest.skip("reference sentences not restored: run `make data`")
    return [r["hyp_beam"] for r in rows], [r["ref"] for r in rows]


# ── detokenise ↔ 13a ────────────────────────────────────────────────────────
# Auslan-Daily subtitles ship pre-tokenised ("hello ."). sacreBLEU's 13a assumes
# detokenised input. The pipeline therefore detokenises both sides first. That is only
# safe if 13a(detokenize(s)) == s: otherwise BLEU is computed over a token space that
# is neither the corpus's nor sacreBLEU's, and is comparable to nothing.

def test_detokenize_13a_is_an_exact_inverse_on_the_whole_dump(headline):
    tok = pytest.importorskip("sacrebleu.tokenizers.tokenizer_13a").Tokenizer13a()
    hyps, refs = headline
    bad = [s for s in hyps + refs
           if tok(detokenize(s)).split() != s.split()]
    assert bad == [], f"{len(bad)}/{len(hyps) + len(refs)} strings do not round-trip, e.g. {bad[:3]}"


@pytest.mark.parametrize("raw, detok", [
    ("hello .", "hello."),
    ("it is sally , ok ?", "it is sally, ok?"),
    ("( yes )", "(yes)"),
    ("i do n't know .", "i do n't know."),   # the apostrophe rule joins, 13a re-splits
])
def test_detokenize_examples(raw, detok):
    assert detokenize(raw) == detok


def test_bleu_is_100_when_hypothesis_equals_reference(headline):
    _, refs = headline
    d = [detokenize(r) for r in refs]
    assert corpus_bleu4(d, d) == pytest.approx(100.0, abs=1e-6)


def test_headline_bleu_reproduces(headline):
    hyps, refs = headline
    b = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
    assert b == pytest.approx(11.802, abs=0.001)


# ── WER: detokenised vs raw is NOT a rounding difference ────────────────────
# jiwer splits on whitespace, so the pre-tokenised " ." is its own token and inflates
# the reference length, which is the denominator. The project reports the detokenised
# value; the raw one is 0.16 lower, larger than any modelling gain in the whole ladder.
# This asymmetry produced a false "a null model beats the ensemble" claim once
# (retracted, the project log 2026-09-09l). Locked so it cannot drift back.

def test_wer_detokenised_is_the_reported_value(headline):
    hyps, refs = headline
    w = corpus_wer([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
    assert w == pytest.approx(0.9069, abs=0.0001)


def test_wer_on_raw_pretokenised_text_differs_and_is_lower(headline):
    hyps, refs = headline
    raw = corpus_wer(hyps, refs)
    detok = corpus_wer([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
    assert raw == pytest.approx(0.7455, abs=0.0001)
    assert raw < detok - 0.1, "the two conventions must stay visibly different"


def test_wer_is_zero_on_identical_text(headline):
    _, refs = headline
    d = [detokenize(r) for r in refs]
    assert corpus_wer(d, d) == pytest.approx(0.0, abs=1e-9)


# ── ROUGE-L uses beta = 1.2, not F1 ─────────────────────────────────────────
# SignJoey and GASLT — two of the three published comparators on this split — compute a
# RECALL-weighted F-measure. This model under-generates (length ratio 0.79-0.86), so
# precision exceeds recall and F1 scores it higher than the convention it is compared to.

def test_rouge_l_beta_is_the_signjoey_convention():
    import metric_suite
    assert metric_suite.ROUGE_BETA == 1.2


def test_rouge_l_is_below_f1_when_precision_exceeds_recall():
    from metric_suite import rouge_l
    hyp, ref = "a b", "a b c d"          # p = 1.0, r = 0.5
    p, r = 1.0, 0.5
    f1 = 2 * p * r / (p + r)
    got = rouge_l(hyp, ref)
    assert got == pytest.approx((1 + 1.2 ** 2) * p * r / (r + 1.2 ** 2 * p))
    assert got < f1


def test_rouge_l_equals_f1_when_precision_equals_recall():
    from metric_suite import rouge_l
    assert rouge_l("a b c", "a b c") == pytest.approx(1.0)


def test_rouge_l_handles_empty_input():
    from metric_suite import rouge_l
    assert rouge_l("", "a b") == 0.0
    assert rouge_l("a b", "") == 0.0
    assert rouge_l("x", "y") == 0.0          # no LCS -> p + r == 0
