#!/usr/bin/env python3
"""Wave 12a: how often does the pre-registered rule (tools/w12a_news_decomp_readout.py) return each reading at three seeds?

A simulation, not a result: it uses the rule's own functions (summarise, component, reading) on invented per-seed
deltas whose spread is taken from the five recorded paired deltas nSA - nA of jobs 7950 and 16398
(results/PREREG_WAVE4_2026-09-22.md, outcome table of 24 Sep; validation and test, beam-5 BLEU-4). Assumptions, all
labelled in the printout: S = nS - nA and A = nSA - nS are drawn independently, each with the spread of the recorded
bundle delta (so the simulated bundle S + A is sqrt(2) times as variable as recorded: a conservative choice, since the
two share the nS run); validation and test deltas are independent; the true validation effect is the recorded ratio (0.265 / 0.336)
times the true test effect.

A second table covers the case in which nA and nSA reproduce job 7950 exactly at seeds 42 / 101 / 202 (the 512-frame nA
of job 116620 did, to three decimals, in all three seeds): then the bundle deltas B are the recorded ones and only nS is
new. In that table S is drawn with the spread above, A = B - S per seed, and the true A is the recorded mean of B minus the
true S. It is an approximation (S and the recorded B share the nA run).

  python tools/w12a_power.py            prints both tables
  python tools/w12a_power.py --self_test
"""
import argparse, os, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from w12a_news_decomp_readout import component, reading, summarise      # noqa: E402

# nSA - nA per seed 42 / 101 / 202 / 303 / 404, beam-5 BLEU-4 (PREREG_WAVE4, outcome of job 16398)
B_VAL = [-0.188, 0.040, 0.353, 0.556, 0.566]
B_TEST = [0.231, 0.602, 0.399, 0.379, 0.068]
RATIO_VAL = st.mean(B_VAL) / st.mean(B_TEST)
N_SIM = 4000


def simulate(s_test, a_test, sd_test, sd_val, n_sim=20000, seed=0, fixed_b=None):
    """True test effects of S and A -> share of simulations ending in each reading and in each component status.
    fixed_b = (validation deltas, test deltas) of the bundle: then B is that, S is drawn and A = B - S (a_test and the
    validation ratio are not used; the validation effect of S is its test effect times mean(B val) / mean(B test))."""
    rng = np.random.default_rng(seed)
    codes, status = {}, {"S": {}, "A": {}}
    for _ in range(n_sim):
        draw = lambda mu, sd: [float(x) for x in rng.normal(mu, sd, 3)]
        ratio = RATIO_VAL if fixed_b is None else st.mean(fixed_b[0]) / st.mean(fixed_b[1])
        S = {"val": summarise(draw(ratio * s_test, sd_val)), "test": summarise(draw(s_test, sd_test))}
        if fixed_b is None:
            A = {"val": summarise(draw(RATIO_VAL * a_test, sd_val)), "test": summarise(draw(a_test, sd_test))}
            B = {m: summarise([x + y for x, y in zip(S[m]["per_seed"], A[m]["per_seed"])]) for m in ("val", "test")}
        else:
            B = {"val": summarise(list(fixed_b[0])), "test": summarise(list(fixed_b[1]))}
            A = {m: summarise([b - x for b, x in zip(B[m]["per_seed"], S[m]["per_seed"])]) for m in ("val", "test")}
        c = reading(S, A, B)[0]
        codes[c] = codes.get(c, 0) + 1
        for k, v in (("S", S), ("A", A)):
            r = component(v["val"], v["test"]); status[k][r] = status[k].get(r, 0) + 1
    f = lambda d: {k: v / n_sim for k, v in sorted(d.items())}
    return f(codes), {k: f(v) for k, v in status.items()}


def main():
    sd_t, sd_v = st.stdev(B_TEST), st.stdev(B_VAL)
    print(f"recorded bundle deltas, five seeds: test mean {st.mean(B_TEST):+.3f} sd {sd_t:.3f}; validation mean "
          f"{st.mean(B_VAL):+.3f} sd {sd_v:.3f}; assumed validation / test effect ratio {RATIO_VAL:.2f}")
    print("assumed spread of S and A per seed: test sd %.3f and 1.5x that; validation sd %.3f and 1.5x that" % (sd_t, sd_v))
    print(f"share of {N_SIM:,} simulated experiments per row at three seeds; true TEST effects in BLEU-4 (S = stitched "
          "clips, A = augmentation on top)\n")
    print(f"{'true S':>7s} {'true A':>7s} {'spread':>7s} | {'P(S res.)':>9s} {'P(A res.)':>9s} | " + " ".join(f"{c:>5s}" for c in
          ("R0", "R1", "R2", "R3", "R4")))
    for s, a in ((0.34, 0.0), (0.17, 0.17), (0.0, 0.34), (0.0, 0.0), (0.5, 0.0), (0.6, 0.6)):
        for mult in (1.0, 1.5):
            codes, status = simulate(s, a, sd_t * mult, sd_v * mult, N_SIM)
            print(f"{s:7.2f} {a:7.2f} {mult:6.1f}x | {status['S'].get('resolved', 0):9.2f} {status['A'].get('resolved', 0):9.2f} | "
                  + " ".join(f"{codes.get(c, 0):5.2f}" for c in ("R0", "R1", "R2", "R3", "R4")))
    fb = (B_VAL[:3], B_TEST[:3]); bm = st.mean(fb[1])
    print(f"\nIF nA AND nSA REPRODUCE JOB 7950 (seeds 42 / 101 / 202): the bundle is the recorded one, test mean {bm:+.3f}, "
          f"validation mean {st.mean(fb[0]):+.3f}, so R0 cannot occur and only nS is new; true A = {bm:+.3f} - true S")
    print(f"{'true S':>7s} {'true A':>7s} {'spread':>7s} | {'P(S res.)':>9s} {'P(A res.)':>9s} | " + " ".join(f"{c:>5s}" for c in
          ("R0", "R1", "R2", "R3", "R4")))
    for s_ in (bm, 0.2, 0.0, -0.2):
        for mult in (1.0, 1.5):
            codes, status = simulate(s_, bm - s_, sd_t * mult, sd_v * mult, N_SIM, fixed_b=fb)
            print(f"{s_:7.2f} {bm - s_:7.2f} {mult:6.1f}x | {status['S'].get('resolved', 0):9.2f} {status['A'].get('resolved', 0):9.2f} | "
                  + " ".join(f"{codes.get(c, 0):5.2f}" for c in ("R0", "R1", "R2", "R3", "R4")))
    # smallest true effect with an even chance of being resolved when the other component is zero
    print("\nthe test interval of a component excludes zero at three seeds only if its mean exceeds 4.303 * sd / sqrt(3) = "
          f"{4.303 * sd_t / 3 ** 0.5:.2f} (spread {sd_t:.2f}) or {4.303 * 1.5 * sd_t / 3 ** 0.5:.2f} (spread {1.5 * sd_t:.2f})")


def self_test():
    # no noise, large effects: the reading is deterministic
    codes, status = simulate(1.0, 1.0, 1e-6, 1e-6, 50)
    assert codes == {"R1": 1.0} and status["S"] == {"resolved": 1.0} and status["A"] == {"resolved": 1.0}, (codes, status)
    codes, _ = simulate(-1.0, 2.0, 1e-6, 1e-6, 50)
    assert codes == {"R2": 1.0}, codes        # S clearly negative = "absent", A resolved, the bundle positive
    codes, _ = simulate(-1.0, -1.0, 1e-6, 1e-6, 50)
    assert codes == {"R0": 1.0}, codes        # the bundle is not reproduced
    # shares add to one
    codes, status = simulate(0.2, 0.1, 0.2, 0.3, 300)
    assert abs(sum(codes.values()) - 1) < 1e-9 and all(abs(sum(v.values()) - 1) < 1e-9 for v in status.values())
    # fixed bundle: S + A = B exactly in every draw, R0 never occurs (B is positive), no noise -> the reading follows S
    fb = ([0.1, 0.1, 0.1], [0.4, 0.4, 0.4])
    codes, status = simulate(0.4, 0.0, 1e-6, 1e-6, 50, fixed_b=fb)
    assert codes == {"R1": 1.0} and status["S"] == {"resolved": 1.0}, (codes, status)    # A is ~0 +- 1e-6: any status
    codes, status = simulate(-0.4, 0.8, 1e-6, 1e-6, 50, fixed_b=fb)      # S clearly negative, so A = B - S = 0.8
    assert codes == {"R2": 1.0} and status["A"] == {"resolved": 1.0}, (codes, status)
    codes, _ = simulate(0.2, 0.2, 0.3, 0.3, 300, fixed_b=fb)
    assert "R0" not in codes and abs(sum(codes.values()) - 1) < 1e-9, codes
    assert abs(st.stdev(B_TEST) - 0.1996) < 1e-3 and abs(st.mean(B_TEST) - 0.3358) < 1e-3 and abs(st.mean(B_VAL) - 0.2654) < 1e-3
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--self_test", action="store_true")
    self_test() if ap.parse_args().self_test else main()
