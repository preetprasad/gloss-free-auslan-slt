"""Exact paired McNemar test between two continuous_id_probe runs (`--dump_pred` CSVs), per variant.

python tools/paired_probe_mcnemar.py A.csv B.csv [--variants hands body_hands ...]
Prints, per variant: shared spans, top-1 of each, spans only A got right, only B got right, two-sided exact p.
"""
import argparse

import pandas as pd
from scipy.stats import binomtest


def mcnemar(a, b, variant):
    m = a[a.variant == variant].merge(b[b.variant == variant], on="span", suffixes=("_a", "_b"))
    only_a = int(((m.correct_a == 1) & (m.correct_b == 0)).sum())
    only_b = int(((m.correct_a == 0) & (m.correct_b == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return len(m), 100 * m.correct_a.mean(), 100 * m.correct_b.mean(), only_a, only_b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--variants", nargs="+", default=["hands", "body_hands", "hands [HAND-LOCAL]"])
    x = ap.parse_args()
    a, b = pd.read_csv(x.a), pd.read_csv(x.b)
    for v in x.variants:
        n, ta, tb, oa, ob, p = mcnemar(a, b, v)
        print(f"{v:22s} spans {n:4d} | top-1 A {ta:5.1f}% B {tb:5.1f}% | only-A {oa:3d} only-B {ob:3d} | exact p = {p:.2e}")


if __name__ == "__main__":
    t = pd.DataFrame({"variant": ["hands"] * 4, "span": list("wxyz"), "correct": [1, 1, 0, 0]})
    u = pd.DataFrame({"variant": ["hands"] * 4, "span": list("wxyz"), "correct": [1, 0, 0, 1]})
    assert mcnemar(t, u, "hands")[3:5] == (1, 1)
    main()
