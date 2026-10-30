"""Seed-level paired decision for one arm against its in-batch control (R2 protocol rules 3 and 7).

Reads the per-seed summary JSONs written by w3summary / w4summary ({arm: {beam5_val, beam5_test, ...}}), and
prints per-seed deltas, sign counts, the mean and a two-sided 95% seed-level t-interval for beam-5 validation
(the Stage-1 criterion) and beam-5 test. It decides nothing by itself; the pre-registration does.

python tools/seed_decision.py --control w3C --arm w3T summary_s42.json summary_s101.json ...
"""
import argparse, json, statistics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}   # two-sided 95%, df = n - 1


def interval(d):
    m = statistics.mean(d)
    if len(d) < 2:
        return m, float("nan"), float("nan")
    hw = T975[len(d) - 1] * statistics.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw


def deltas(rows, control, arm, key):
    return [r[arm][key] - r[control][key] for r in rows if arm in r and control in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in a.files]
    for key in ("beam5_val", "beam5_test"):
        d = deltas(rows, a.control, a.arm, key)
        m, lo, hi = interval(d)
        print(f"{a.arm} - {a.control} {key:10s} n={len(d)} per-seed " + " ".join(f"{x:+.3f}" for x in d)
              + f" | positive {sum(x > 0 for x in d)}/{len(d)} | mean {m:+.3f} | 95% t [{lo:+.3f}, {hi:+.3f}]")
    for arm in (a.control, a.arm):
        print(f"  {arm:8s} beam-5 val " + " ".join(f"{r[arm]['beam5_val']:.3f}" for r in rows if arm in r)
              + " | test " + " ".join(f"{r[arm]['beam5_test']:.3f}" for r in rows if arm in r))


if __name__ == "__main__":
    m, lo, hi = interval([4.863, 5.228, 4.820])
    assert abs(m - 4.970) < 1e-3 and lo < m < hi
    main()
