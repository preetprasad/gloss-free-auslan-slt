"""Wave-12b read-out: W (wrong-sense signs removed from the synthetic clips) against A (adopted corpus), seeds 42/101/202.
Rule and readings: results/PREREG_WAVE12B_WRONGSENSE_2026-10-13.md, written before any run.

  python tools/w12b_readout.py RUNS_DIR --train_manifest M/train.tsv     (RUNS_DIR holds {A,W}_s{42,101,202}/)
  python tools/w12b_readout.py selftest

Per split (beam 5, best.pt): each arm's per-seed scores, then W - A per seed, the number of positive seeds, the mean and
the seed-level 95% t-interval, for BLEU-4, unseen-target BLEU-4, content precision and recall, length ratio, WER,
distinct outputs, and precision/recall restricted to the 20 English word forms whose sign was removed. Every BLEU is
recomputed from the saved hypotheses and must equal the stored value. The verdict line applies the pre-registered rule.
"""
import argparse, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_readouts import KEY, readouts      # noqa: E402  (also puts the code root on sys.path for slt.metrics)
from seed_decision import interval          # noqa: E402
from stitch_lexicon import STOP             # noqa: E402

SEEDS = (42, 101, 202)
HARM = -0.5     # "clearly harmful": the 3-seed mean of W - A is at or below this on beam-5 validation AND on beam-5 test
# the word forms rendered with a wrong-sense sign in the adopted 13,760 clips (measured 13 Oct 2026, 3,257 signs)
AFFECTED = frozenset("can good be know want yet being enough mind wanted believe wants knew matter knowing bothered "
                     "known wanting knows cans".split())
KEYS = ("bleu4", "bleu4_unseen", "content_precision", "content_recall", "len_ratio", "wer", "distinct",
        "affected_precision", "affected_recall")


def affected(hyps, refs):
    hit = nh = nr = 0
    for h, r in zip(hyps, refs):
        ch, cr = ([w for w in t.lower().split() if w in AFFECTED] for t in (h, r))
        hit += sum((Counter(ch) & Counter(cr)).values()); nh += len(ch); nr += len(cr)
    return {"affected_precision": 100 * hit / max(nh, 1), "affected_recall": 100 * hit / max(nr, 1), "affected_ref_tokens": nr}


def harness(a_test):
    """Pre-registered harness check (section 4): arm A is the adopted recipe, whose 3-seed mean beam-5 test score must
    lie within 1.2 of 11.2 (the cross-job band) before W - A is interpreted."""
    ok = abs(a_test - 11.2) <= 1.2
    return ok, (f"HARNESS CHECK: arm A 3-seed mean beam-5 test BLEU-4 {a_test:.3f} is "
                + ("within 11.2 +/- 1.2" if ok else
                   "OUTSIDE 11.2 +/- 1.2 -> DO NOT INTERPRET the verdict below until the cause is found"))


def verdict(d_val, d_test):
    harm = d_val <= HARM and d_test <= HARM
    return (("CLEARLY HARMFUL (both means <= -0.5): W is rejected; the adopted corpus stays as it is" if harm else
             "NOT CLEARLY HARMFUL: W is adopted as a correctness fix for corpora built from now on; no gain is claimed")
            + (" | futility read-out: validation mean <= 0, W is not a lever" if d_val <= 0 else
               " | futility read-out: validation mean > 0, not stopped; still no claim (no Stage 2 is planned)"))


def load(runs, arm, seed, split, train):
    j = json.load(open(os.path.join(runs, f"{arm}_s{seed}", f"beam5_{split}_best.json")))
    r = readouts(j["beam_hyps"], j["refs"], STOP, train)
    assert abs(r["bleu4"] - j["beam_result"]["metrics"][KEY]) < 1e-6, f"{arm}_s{seed} {split}: recomputed BLEU differs"
    r.update(affected(j["beam_hyps"], j["refs"]))
    return r, j["refs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--train_manifest", required=True)
    a = ap.parse_args()
    train = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split())
             for l in open(a.train_manifest, encoding="utf-8") if "\t" in l}
    mean, a_mean = {}, {}
    for split in ("val", "test"):
        got = {(arm, s): load(a.runs, arm, s, split, train) for arm in "AW" for s in SEEDS}
        assert all(got["A", s][1] == got["W", s][1] == got["A", 42][1] for s in SEEDS), "references differ between runs"
        r = {k: v[0] for k, v in got.items()}
        print(f"== {split}: n {r['A', 42]['n']}, unseen n {r['A', 42]['unseen_n']}, "
              f"affected-word tokens in the references {r['A', 42]['affected_ref_tokens']}")
        for key in KEYS:
            d = [r["W", s][key] - r["A", s][key] for s in SEEDS]
            m, lo, hi = interval(d)
            print(f"  {key:18s} A " + " ".join(f"{r['A', s][key]:8.3f}" for s in SEEDS)
                  + " | W " + " ".join(f"{r['W', s][key]:8.3f}" for s in SEEDS)
                  + " | W - A " + " ".join(f"{x:+.3f}" for x in d)
                  + f" | positive {sum(x > 0 for x in d)}/3 | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if key == "bleu4":
                mean[split] = m
                a_mean[split] = sum(r["A", s][key] for s in SEEDS) / len(SEEDS)
    print(harness(a_mean["test"])[1])
    print(f"VERDICT (W - A beam-5 BLEU-4, 3-seed means: validation {mean['val']:+.3f}, test {mean['test']:+.3f}): "
          + verdict(mean["val"], mean["test"]))


def selftest():
    assert verdict(-0.6, -0.5).startswith("CLEARLY HARMFUL") and "not a lever" in verdict(-0.6, -0.5)
    assert verdict(-0.6, -0.49).startswith("NOT CLEARLY") and verdict(-0.49, -2.0).startswith("NOT CLEARLY")
    assert "not stopped" in verdict(0.01, 0.3) and "not a lever" in verdict(0.0, 0.3)
    assert harness(11.134)[0] and harness(10.1)[0] and harness(12.3)[0] and not harness(9.9)[0] and not harness(12.5)[0]
    assert "DO NOT INTERPRET" in harness(9.0)[1] and "within" in harness(11.0)[1]
    r = affected(["you can go .", "it is good good ."], ["can you go .", "it was not good ."])
    assert r["affected_ref_tokens"] == 2 and abs(r["affected_precision"] - 100 * 2 / 3) < 1e-9 and r["affected_recall"] == 100.0, r
    m, lo, hi = interval([0.1, -0.2, 0.4])
    assert abs(m - 0.1) < 1e-12 and lo < m < hi
    assert len(AFFECTED) == 20
    print("selftest OK")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
