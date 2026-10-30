"""Content-word failure analysis (the 2026-09-08 numbers: 70.1% zero-correct, 16.3% ordered vs 16.5% unordered).

Those numbers existed only as prose; no committed script computed them. Definitions, stated here:
  * content words = lower-cased whitespace tokens of the (pre-tokenised) text that are alphabetic and not in
    the chosen stop list;
  * clips whose reference has no content word are excluded (the recorded denominator is 786, not 800);
  * zero / all correct = no / every reference content word appears in the hypothesis content-word multiset;
  * recall, unordered = pooled multiset intersection / pooled reference content words;
  * recall, ordered   = pooled LCS length of the two content sequences / pooled reference content words.

python tools/content_failure_analysis.py --hyps FILE.jsonl [--hyp_key hyp_beam] [--stop train_ctc|stitch_lexicon|...]
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def stoplist(name):
    if name == "train_ctc":
        import train_ctc; return train_ctc.STOP
    mod = __import__(name)
    return mod.STOP


def content(text, stop):
    return [w for w in text.lower().split() if w.isalpha() and w not in stop]


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def analyse(pairs, stop):
    n = zero = allc = tot = hit_bag = hit_lcs = 0
    for ref, hyp in pairs:
        r, h = content(ref, stop), content(hyp, stop)
        if not r:
            continue
        n += 1; tot += len(r)
        inter = sum((Counter(r) & Counter(h)).values())
        hit_bag += inter; hit_lcs += lcs(r, h)
        zero += inter == 0; allc += inter == len(r)
    return dict(n=n, zero=zero, zero_pct=100 * zero / n, all=allc, all_pct=100 * allc / n,
                mean_content=tot / n, recall_unordered=100 * hit_bag / tot, recall_ordered=100 * hit_lcs / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--hyp_key", default="hyp_beam")
    ap.add_argument("--stop", default="train_ctc")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.hyps)]
    r = analyse([(x["ref"], x[a.hyp_key]) for x in rows], stoplist(a.stop))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))


if __name__ == "__main__":
    assert lcs("a b c d".split(), "b x d".split()) == 2 and lcs([], ["a"]) == 0
    main()
