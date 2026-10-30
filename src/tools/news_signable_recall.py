#!/usr/bin/env python3
"""Mandatory News secondary (results/PREREG_WAVE4_2026-09-22.md, addenda of 22 Sep 23:55 and 24 Sep 08:24).

In-vocab content recall on the News test set, split by whether the content TYPE is signable under the adopted lexicon
(morphology=True, paren_alias=False, MM-WLAuslan Train+Valid labels). In-vocab = the type occurs in the News REAL
training text. Recall = pooled multiset intersection of hypothesis and reference content words within a class /
pooled reference content tokens in that class, on beam-5 best.pt TEST decodes.

GATE (asserted before any recall is printed): 934 / 1,821 signable / unsignable test types; 2,379 in-vocab types,
878 signable / 1,501 not.

  python tools/news_signable_recall.py --labels "data/raw/MM-WLAuslan/Annotation/Labels & Split" \
      --news_train news_train.tsv --news_test news_test.tsv --runs <dir with nA_s*/ nSA_s*/ beam5_test_best.json>
"""
import argparse, json, os, statistics as st, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_lexicon import build_lexicon, content_words, coverage   # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def texts(manifest):
    return [l.rstrip("\n").split("\t", 1)[1] for l in open(manifest) if "\t" in l]


def recall(hyps, refs, keep):
    hit = tot = 0
    for h, r in zip(hyps, refs):
        rc = Counter(w for w in content_words(r.lower()) if w in keep)
        hc = Counter(w for w in content_words(h.lower()) if w in keep)
        hit += sum((rc & hc).values()); tot += sum(rc.values())
    return 100.0 * hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--news_train", required=True)
    ap.add_argument("--news_test", required=True); ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    lex = build_lexicon([os.path.join(a.labels, "Train.json"), os.path.join(a.labels, "Valid.json")],
                        morphology=True, paren_alias=False)
    test_refs = texts(a.news_test)
    types = {w for t in test_refs for w in content_words(t.lower())}
    sign = {w for w in types if coverage(w + " .", lex)[0] > 0}
    # in-vocab = the type is a (lower-cased, whitespace-split) TOKEN of the News real training text, i.e. a word the
    # model can emit. This is the definition that produced the pinned 2,379; the content_words regex gives 2,462.
    train_types = {w for t in texts(a.news_train) for w in t.lower().split()}
    inv = types & train_types
    counts = (len(sign), len(types - sign), len(inv), len(inv & sign), len(inv - sign))
    print("types: signable %d / unsignable %d | in-vocab %d (signable %d / not %d)" % counts)
    assert counts == (934, 1821, 2379, 878, 1501), f"GATE FAILED: {counts} != (934, 1821, 2379, 878, 1501)"
    print("GATE OK: reproduces the pre-registered type counts")
    classes = {"signable": inv & sign, "unsignable": inv - sign}
    res = {}
    for arm in ("nA", "nSA"):
        for s in (42, 101, 202, 303, 404):
            f = os.path.join(a.runs, f"{arm}_s{s}", "beam5_test_best.json")
            if not os.path.exists(f):
                print(f"missing {f}"); continue
            j = json.load(open(f)); H, R = j["beam_hyps"], j["refs"]
            assert [" ".join(r.split()) for r in R] == [" ".join(r.split()) for r in test_refs], f"{f}: refs differ"
            res[(arm, s)] = {c: recall(H, R, k) for c, k in classes.items()}
            print(f"{arm:3s} s{s}: " + " | ".join(f"{c} {v:.2f}%" for c, v in res[(arm, s)].items()))
    seeds = [s for s in (42, 101, 202, 303, 404) if ("nA", s) in res and ("nSA", s) in res]
    for c in classes:
        d = [res[("nSA", s)][c] - res[("nA", s)][c] for s in seeds]
        if len(d) >= 2:
            m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5
            print(f"nSA - nA {c:10s}: {' '.join(f'{x:+.2f}' for x in d)} | mean {m:+.2f} [{m-hw:+.2f}, {m+hw:+.2f}] "
                  f"| same sign as mean {sum((x > 0) == (m > 0) for x in d)}/{len(d)}")


if __name__ == "__main__":
    main()
