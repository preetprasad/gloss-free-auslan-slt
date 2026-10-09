"""Wave-11 readout: validation-only selection (results/PREREG_WAVE11_VALSELECT_2026-10-09.md).

usage: w11_readout.py <runs_dir> <control_arm> <tie> <arm> [<arm> ...]   (arms in order of increasing cost)
Prints per-arm three-seed means, the arm selected on mean beam-5 validation BLEU-4 (the cheapest arm within <tie>
of the highest mean), and paired differences from the control with 95% t-intervals.
"""
import json, math, os, statistics as st, sys

KEY = "tokenize=13a;detok=True"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def load(runs, arm, seed):
    d = os.path.join(runs, f"{arm}_s{seed}")
    p = json.load(open(os.path.join(d, "progress.json")))
    b = lambda split: json.load(open(os.path.join(d, f"beam5_{split}_best.json")))["beam_result"]["metrics"][KEY]
    return dict(greedy_val=p["best"]["bleu4"], epoch=p["best"]["epoch"], beam5_val=b("val"), beam5_test=b("test"))


def interval(d):
    m = st.mean(d)
    if len(d) < 2: return m, float("nan"), float("nan")
    h = T95[len(d) - 1] * st.stdev(d) / math.sqrt(len(d))
    return m, m - h, m + h


def select(means, arms, tie):
    """arms are in order of increasing cost: the cheapest arm whose mean is within <tie> of the highest mean."""
    best = max(means[a] for a in arms)
    return next(a for a in arms if means[a] > best - tie)


def main():
    runs, control, tie, arms = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
    seeds = [42, 101, 202]; rows = {}
    for a in dict.fromkeys([control] + arms):
        got = {}
        for s in seeds:
            try: got[s] = load(runs, a, s)
            except Exception as e: print(f"  missing {a}_s{s}: {type(e).__name__}")
        rows[a] = got
    print(f"{'arm':8s} n  greedy val  beam5 val  beam5 test   per-seed beam5 val / test")
    for a, got in rows.items():
        if not got: continue
        print(f"{a:8s} {len(got)}  {st.mean(v['greedy_val'] for v in got.values()):9.3f}  "
              f"{st.mean(v['beam5_val'] for v in got.values()):9.3f}  {st.mean(v['beam5_test'] for v in got.values()):9.3f}   "
              + "  ".join(f"s{s} {v['beam5_val']:.3f}/{v['beam5_test']:.3f}" for s, v in got.items()))
    full = [a for a in arms if len(rows[a]) == len(seeds)]
    if len(full) != len(arms):
        print("NOT ALL ARMS COMPLETE: no selection is made."); return
    means = {a: st.mean(v["beam5_val"] for v in rows[a].values()) for a in arms}
    sel = select(means, arms, tie)
    print(f"SELECTED ON VALIDATION: {sel} (mean beam-5 validation {means[sel]:.3f}); its mean beam-5 test "
          f"{st.mean(v['beam5_test'] for v in rows[sel].values()):.3f}")
    for ref in dict.fromkeys([control, arms[0]]):
        if len(rows[ref]) != len(seeds): continue
        for a in arms:
            if a == ref: continue
            for k in ("beam5_val", "beam5_test"):
                m, lo, hi = interval([rows[a][s][k] - rows[ref][s][k] for s in seeds])
                print(f"  {a} - {ref} {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def self_test():
    assert select({"a": 10.0, "b": 10.05, "c": 10.30}, ["a", "b", "c"], 0.10) == "c"
    assert select({"a": 10.0, "b": 10.05, "c": 10.08}, ["a", "b", "c"], 0.10) == "a"   # all within the margin
    assert select({"a": 10.0, "b": 10.20, "c": 10.25}, ["a", "b", "c"], 0.10) == "b"   # c leads b by only 0.05
    assert select({"a": 2.0, "b": 1.9}, ["a", "b"], 0.05) == "a"
    assert select({"a": 1.90, "b": 1.94}, ["a", "b"], 0.05) == "a" and select({"a": 1.90, "b": 1.96}, ["a", "b"], 0.05) == "b"
    m, lo, hi = interval([1.0, 2.0, 3.0]); assert abs(m - 2) < 1e-9 and abs(hi - (2 + 4.303 / math.sqrt(3))) < 1e-6
    print("self-test passed")


if __name__ == "__main__":
    self_test() if sys.argv[1:] == ["--self_test"] else main()
