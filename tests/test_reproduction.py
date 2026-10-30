"""Tier-0 reproduction as a test: every committed decode re-scores to its recorded value.

No GPU, no dataset, no checkpoints. This is the gate that the reported numbers are what the
repository actually contains. It reuses src/tools/reproduce_headline.py — the same
code a reviewer runs by hand — rather than restating its table here.
"""
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2 = os.path.join(REPO, "src")
sys.path.insert(0, PHASE2)
sys.path.insert(0, os.path.join(PHASE2, "tools"))

import reproduce_headline as rh  # noqa: E402


def _restored():
    """False while the saved outputs hold digests in place of the reference sentences (before `make data`)."""
    first = os.path.join(REPO, rh.CASES[0][1])
    return not os.path.exists(first) or '"ref": "sha1:' not in open(first, encoding="utf-8").readline()


@pytest.fixture(autouse=True)
def _need_references():
    if not _restored():
        pytest.skip("reference sentences not restored: run `make data`")


def _dump(path):
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        pytest.skip(f"dump absent from this checkout: {path}")
    return rh.load(path)


@pytest.mark.parametrize("label, path, rec_bleu, rec_wer",
                         [(c[0], c[1], c[2], c[3]) for c in rh.CASES],
                         ids=[c[1].split("/")[1] + ":" + c[1].split("/")[-1] for c in rh.CASES])
def test_dump_reproduces_recorded_scores(label, path, rec_bleu, rec_wer):
    from slt.metrics import corpus_bleu4, corpus_wer
    hyps, refs = _dump(path)
    assert len(hyps) == 800, f"{path}: expected the 800-clip official split, got {len(hyps)}"
    assert corpus_bleu4(hyps, refs) == pytest.approx(rec_bleu, abs=0.01), label
    if rec_wer is not None:
        assert corpus_wer(hyps, refs) == pytest.approx(rec_wer, abs=0.001), label


def test_reported_system_is_the_first_case():
    """The headline must stay first: the seen/unseen decomposition below uses CASES[0]."""
    assert rh.CASES[0][1] == "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl"
    assert rh.CASES[0][2:4] == (11.802, 0.9069)


def test_seen_unseen_decomposition():
    """35.00 seen / 6.60 unseen. Needs a training manifest, which is not redistributed."""
    hyps, refs = _dump(rh.CASES[0][1])
    su = rh.seen_unseen(hyps, refs)
    if su is None:
        pytest.skip(f"training manifest absent ({rh.TRAIN_MANIFEST}); run `make data`")
    assert su["seen"][0] == 230 and su["unseen"][0] == 570
    assert su["seen"][1] == pytest.approx(35.00, abs=0.05)
    assert su["unseen"][1] == pytest.approx(6.60, abs=0.05)


def test_reproduce_headline_script_exits_zero():
    """The documented one-command check must actually pass, as a reviewer will run it."""
    r = subprocess.run([sys.executable, "tools/reproduce_headline.py"],
                       cwd=PHASE2, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reproduce their recorded BLEU-4 and WER exactly" in r.stdout


def test_reproduce_headline_gate_fails_on_a_mismatch(monkeypatch):
    """The gate must have teeth: README and the tool's own docstring call it a CI gate.

    Runs in-process against a deliberately wrong recorded value; the dumps on disk are untouched.
    """
    monkeypatch.setattr(sys, "argv", ["reproduce_headline.py"])
    cases = list(rh.CASES)
    label, path, bleu, wer, src = cases[0]
    cases[0] = (label, path, bleu + 1.0, wer, src)
    monkeypatch.setattr(rh, "CASES", cases)
    assert rh.main() == 1, "a wrong recorded BLEU-4 did not fail the reproduction gate"


def test_holm_decision_self_test():
    """The pre-registered decision protocol's own assertions."""
    r = subprocess.run([sys.executable, "tools/holm_decision.py", "--self_test"],
                       cwd=PHASE2, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all assertions passed" in r.stdout


def test_holm_decision_reproduces_the_confirmatory_family():
    """Rebuilt from saved artefacts: nothing in the family was adopted."""
    r = subprocess.run([sys.executable, "tools/holm_decision.py", "--reproduce"],
                       cwd=PHASE2, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("--reproduce needs saved arm artefacts not in this checkout:\n" + r.stdout + r.stderr)
    out = r.stdout
    assert "reproduces the recorded {w2LF 1, w1H 1, L60 0.54, w3T 0.60}, nothing adopted" in out
    assert out.count("NOT adopted") == 4, out
    for arm, p_str in (("w2LF", "1.000"), ("w1H", "1.000"), ("L60", "0.536"), ("w3T", "0.597")):
        assert re.search(rf"^\s*\d+\s+{arm}\s+{p_str}\s", out, re.M), f"{arm} p-value moved:\n{out}"
