#!/usr/bin/env python3
"""Paired sentence bootstrap of the mean LLM-judge adequacy difference between two systems.

Committed 2026-09-24 so the recorded t3-errs6 Part-2 readout (PREREG_WAVE4, "semantic re-grade (job 10288)") is
re-runnable: S6 − RGB-fusion mean adequacy +0.052, 95% [−0.007, +0.113], 2,000 resamples, numpy default_rng(0).

  python tools/adequacy_bootstrap.py --a results/semantic_regrade_10288/s6.graded_hf.csv \
      --b results/semantic_regrade_10288/rgbfusion.graded_hf.csv [--expect 0.052,-0.007,0.113]
"""
import argparse, csv, sys
import numpy as np


def grades(path):
    rows = sorted(csv.DictReader(open(path, encoding="utf-8")), key=lambda r: int(r["idx"]))
    return [r["ref"] for r in rows], np.array([float(r["adequacy"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect", default="", help="mean,lo,hi on record; exit 1 on mismatch (rounded to 3 dp)")
    a = ap.parse_args()
    ra, ga = grades(a.a); rb, gb = grades(a.b)
    assert ra == rb, "the two files are not aligned on the same references"
    d = ga - gb
    rng = np.random.default_rng(a.seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"mean adequacy A {ga.mean():.3f} B {gb.mean():.3f} | A - B {d.mean():+.3f} 95% [{lo:+.3f}, {hi:+.3f}] "
          f"({a.n} resamples, seed {a.seed}) | A higher {int((d > 0).sum())} lower {int((d < 0).sum())} same {int((d == 0).sum())}")
    if a.expect:
        e = [f"{float(x):+.3f}" for x in a.expect.split(",")]
        got = [f"{v:+.3f}" for v in (d.mean(), lo, hi)]   # compare as printed (lo sits at ~-0.0075)
        if got != e:
            print(f"MISMATCH vs record {e}: got {got}"); return 1
        print("reproduces the recorded readout exactly")


if __name__ == "__main__":
    sys.exit(main())
