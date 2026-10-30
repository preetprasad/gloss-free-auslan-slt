#!/usr/bin/env python3
"""The round-2 decision protocol, computed instead of hand-worked (CLAUDE.md section 9, rules 3-9).

Every Holm-corrected claim in this project was arithmetic done by hand: `grep -rli holm` over the tree
returns nothing.  This is the tool.  It implements, exactly as the pre-registrations state them:

  (a) Stage-1 futility  -- stop an arm whose 3-seed mean paired beam-5 VALIDATION delta is <= 0
                           (PREREG_WAVE1_R2 "Decision rules", PREREG_WAVE2_R2 section 4).
  (b) one-sided paired t-test over the per-seed beam-5 TEST deltas (seed_decision.py is two-sided only).
  (c) Holm-Bonferroni step-down over the pre-registered family of m hypotheses.
  (d) adoption = Holm rejection AND 5-seed mean validation delta > 0 AND both Stage-2 seeds positive on test.
  (e) max-T simultaneous intervals for the k-th correlated look at the 800-clip test split
      (protocol rule 9), by bootstrap over the shared test sentences.

Usage:
  python tools/holm_decision.py --self_test
  python tools/holm_decision.py --reproduce            # rebuild the recorded family from results/ artefacts
  python tools/holm_decision.py --family fam.json      # {"alpha":0.05,"arms":{NAME:{"val":[..],"test":[..],"n_stage1":3}}}
  python tools/holm_decision.py --maxt --system S6=.../hyps_LP=0.3.jsonl --system F=... --baseline S6
"""
import argparse
import json
import os
import statistics as st
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")   # Honours/results


# ---------------------------------------------------------------- the four rules

def futility(val_deltas, n_stage1=3):
    """Rule (a). Stage-1 stops the arm when the mean of the first n_stage1 validation deltas is <= 0."""
    m = st.mean(val_deltas[:n_stage1])
    return m, m <= 0


def one_sided_p(deltas, mu=0.0):
    """Rule (b). H1: true mean > mu. Returns (t, p). An arm stopped at Stage 1 is given p = 1 by the prereg."""
    t, p2 = stats.ttest_1samp(deltas, mu)
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def holm(pvals, alpha=0.05):
    """Rule (c). Step-down Holm-Bonferroni. Returns rows (name, p, level, reject) in the tested order.

    Once one hypothesis retains, every later (larger-p) one retains too -- that is the step-down part,
    and it is the half a hand calculation gets wrong."""
    m = len(pvals)
    order = sorted(pvals, key=lambda kv: kv[1])
    rows, still = [], True
    for rank, (name, p) in enumerate(order, start=1):
        level = alpha / (m - rank + 1)
        still = still and p <= level
        rows.append((name, p, level, still))
    return rows


def adopt(val_deltas, test_deltas, rejected, n_stage1=3):
    """Rule (d). All three conditions, reported separately so a failure names itself."""
    c_a = rejected
    c_b = st.mean(val_deltas) > 0
    fresh = test_deltas[n_stage1:]
    c_c = len(fresh) > 0 and all(x > 0 for x in fresh)
    return {"holm_reject": c_a, "val_mean_positive": c_b, "fresh_seeds_positive": c_c,
            "adopt": c_a and c_b and c_c}


# ---------------------------------------------------------------- rule (e): max-T

def simultaneous_band(boot, observed, alpha=0.05):
    """Studentised max-T band over a family of J correlated comparisons.

    boot: (B, J) bootstrap deltas on the SHARED resamples; observed: (J,).
    Returns (se, per_comparison_halfwidth, maxT_halfwidth, q_max).  Because max_j |.| >= |.|_j on every
    resample, q_max >= q_j always, so the simultaneous band is never narrower than the per-comparison one.
    """
    boot, observed = np.asarray(boot, float), np.asarray(observed, float)
    se = boot.std(axis=0, ddof=1)
    z = np.abs(boot - observed) / se
    q_each = np.quantile(z, 1 - alpha, axis=0)
    q_max = float(np.quantile(z.max(axis=1), 1 - alpha))
    return se, q_each * se, q_max * se, q_max


def bootstrap_deltas(systems, baseline, refs, n=2000, seed=0):
    """Paired sentence bootstrap of corpus BLEU-4 deltas vs `baseline`, all systems on the SAME resamples."""
    sys.path.insert(0, os.path.dirname(HERE))
    from slt.metrics import corpus_bleu4                                             # noqa: E402
    names = [k for k in systems if k != baseline]
    N = len(refs)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, N, size=(n, N))
    observed = np.array([corpus_bleu4(systems[k], refs, "13a") - corpus_bleu4(systems[baseline], refs, "13a")
                         for k in names])
    boot = np.empty((n, len(names)))
    for b, ii in enumerate(draws):
        R = [refs[i] for i in ii]
        base = corpus_bleu4([systems[baseline][i] for i in ii], R, "13a")
        for j, k in enumerate(names):
            boot[b, j] = corpus_bleu4([systems[k][i] for i in ii], R, "13a") - base
    return names, observed, boot


# ---------------------------------------------------------------- reporting

def report(arms, alpha=0.05):
    """arms: {name: {"val": [...], "test": [...], "n_stage1": 3, "stopped": bool}} in the family."""
    print(f"\nStage 1 (futility, beam-5 VALIDATION, first {min(a.get('n_stage1', 3) for a in arms.values())} seeds)")
    pvals = []
    for name, a in arms.items():
        n1 = a.get("n_stage1", 3)
        m, stop = futility(a["val"], n1)
        if stop:
            pvals.append((name, 1.0))
            print(f"  {name:8s} mean val delta {m:+.4f}  -> STOP (futile); enters Holm with p = 1")
        else:
            t, p = one_sided_p(a["test"])
            pvals.append((name, p))
            print(f"  {name:8s} mean val delta {m:+.4f}  -> CONTINUE; 5-seed test t={t:+.3f} one-sided p={p:.3f}")

    print(f"\nHolm-Bonferroni step-down, m = {len(pvals)}, alpha = {alpha}")
    print(f"  {'rank':<5}{'arm':<10}{'raw p':>8}{'Holm level':>12}   verdict")
    rows = holm(pvals, alpha)
    for rank, (name, p, level, rej) in enumerate(rows, start=1):
        print(f"  {rank:<5}{name:<10}{p:>8.3f}{level:>12.4f}   {'REJECT' if rej else 'retain'}")

    print("\nAdoption rule (Holm reject AND 5-seed val mean > 0 AND both fresh seeds positive on test)")
    rej = {n: r for n, _, _, r in rows}
    for name, a in arms.items():
        d = adopt(a["val"], a["test"], rej[name], a.get("n_stage1", 3))
        flags = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in list(d.items())[:3])
        print(f"  {name:8s} {flags}  ->  {'ADOPT' if d['adopt'] else 'NOT adopted'}")
    return rows


# ---------------------------------------------------------------- the project's own recorded family

def recorded_family():
    """Rebuild {w2LF, w1H, L60, w3T} from the saved artefacts, not from CLAUDE.md.

    w1H / L60 Stage 1 : results/hand_masking_and_longer_training/runs/<arm>_s<seed>/beam5_{val,test}_best.json
    L60 Stage 2       : results/letter_fix_and_longer_training_stage2/summary_s{303,404}.json   (fresh A2 control)
    w3T 5 seeds       : results/person_tracking_stage2/summary_s{42,101,202,303,404}.json
    w2LF Stage 1      : NOT read from disk -- no wave-2 Stage-1 summary exists under results/.  Only the
                        seed-42/101 VALIDATION deltas (-1.230, +0.611) are in a results artefact
                        (PREREG_WAVE3 line 70); seed 202 (+0.306) and all three TEST deltas are recorded
                        only in CLAUDE.md, which is working memory and has carried wrong figures before.
                        Nothing here is load-bearing: the arm stops on the sign of the 3-seed validation
                        mean and enters Holm with p = 1, which PREREG_WAVE2_R2 and PREREG_WAVE3 both
                        record independently, and its test deltas enter no computation.  If a wave-2
                        summary JSON is recovered from Kaya, repoint this at it.
    """
    def w1(arm, seed, split):
        p = f"{RESULTS}/hand_masking_and_longer_training/runs/{arm}_s{seed}/beam5_{split}_best.json"
        d = json.load(open(p))
        assert d["beam"] == 5 and d["length_penalty"] == 0.6 and d["beam_result"]["n"] == 800, p
        assert d["manifest"].endswith(f"{split}.tsv"), (p, d["manifest"])   # the val/test mix-up guard
        assert d["checkpoint"].endswith(f"{arm}_s{seed}/best.pt"), (p, d["checkpoint"])
        return d["beam_result"]["metrics"]["tokenize=13a;detok=True"]

    def summ(path, ctrl, arm):
        d = json.load(open(path))
        return (d[arm]["beam5_val"] - d[ctrl]["beam5_val"], d[arm]["beam5_test"] - d[ctrl]["beam5_test"])

    s1 = [42, 101, 202]
    arms = {}
    for arm in ("w1H", "w1L60"):
        arms[arm] = {"val": [w1(arm, s, "val") - w1("w1A", s, "val") for s in s1],
                     "test": [w1(arm, s, "test") - w1("w1A", s, "test") for s in s1]}
    for s in (303, 404):                                             # L60 Stage 2, fresh A2 control
        v, t = summ(f"{RESULTS}/letter_fix_and_longer_training_stage2/summary_s{s}.json", "A2", "L60")
        arms["w1L60"]["val"].append(v)
        arms["w1L60"]["test"].append(t)
    w3 = [summ(f"{RESULTS}/person_tracking_stage2/summary_s{s}.json", "w3C", "w3T") for s in s1 + [303, 404]]
    arms["w3T"] = {"val": [v for v, _ in w3], "test": [t for _, t in w3]}
    # w2LF: stopped at Stage 1 (PREREG_WAVE2_R2 OUTCOME / PREREG_WAVE3 line 70 + CLAUDE.md 09-22j(5))
    arms["w2LF"] = {"val": [-1.230, +0.611, +0.306], "test": [-0.243, +0.296, +0.026],
                    "unsourced": "seed-202 val and all 3 test deltas are CLAUDE.md-only (see docstring)"}
    return {"w2LF": arms["w2LF"], "w1H": arms["w1H"], "L60": arms["w1L60"], "w3T": arms["w3T"]}


def reproduce():
    arms = recorded_family()
    print("Recorded confirmatory family, rebuilt from results/ artefacts (m = 4, alpha = 0.05)")
    for name, a in arms.items():
        tag = "   [!] " + a["unsourced"] if a.get("unsourced") else ""
        print(f"  {name:6s} val  " + " ".join(f"{x:+6.3f}" for x in a["val"]) + f"  mean {st.mean(a['val']):+.3f}{tag}")
        print(f"  {'':6s} test " + " ".join(f"{x:+6.3f}" for x in a["test"]) + f"  mean {st.mean(a['test']):+.3f}")
    rows = report(arms)
    got = {n: round(p, 2) for n, p, _, _ in rows}
    want = {"w2LF": 1.0, "w1H": 1.0, "L60": 0.54, "w3T": 0.60}        # CLAUDE.md "Holm family final"
    assert got == want, f"does not reproduce the recorded family: {got} != {want}"
    assert not any(r for _, _, _, r in rows), "recorded outcome is: nothing adopted"
    print("\ncheck: reproduces the recorded {w2LF 1, w1H 1, L60 0.54, w3T 0.60}, nothing adopted")


# ---------------------------------------------------------------- self-test

def self_test():
    # (c) Holm on a hand-worked example: m=4, alpha=.05, levels .0125 .0167 .025 .05
    # a .010 <= .0125 reject | b .020 > .0167 retain | c .030 > .025 retain | d .040 <= .05 BUT step-down retains
    rows = holm([("a", 0.01), ("b", 0.02), ("c", 0.03), ("d", 0.04)], 0.05)
    lv = [round(l, 4) for _, _, l, _ in rows]
    assert lv == [0.0125, 0.0167, 0.025, 0.05], lv
    assert [r for _, _, _, r in rows] == [True, False, False, False], rows
    # step-down: everything after the first retain retains, even when its own p clears its level
    rows = holm([("a", 0.30), ("b", 0.001)], 0.05)      # b would clear .05 alone; a fails at .025 first
    assert [(n, r) for n, _, _, r in rows] == [("b", True), ("a", False)], rows
    # pure step-down: a's p=.04 clears its own level .05, but b failed at .025 first, so a retains
    rows = holm([("a", 0.04), ("b", 0.03), ("c", 0.001)], 0.05)
    assert [(n, round(l, 4), r) for n, _, l, r in rows] == [
        ("c", 0.0167, True), ("b", 0.025, False), ("a", 0.05, False)], rows

    # (b) one-sided p is half the two-sided p when the effect is in the tested direction, and 1 - half when not
    d = [0.9, 0.4, 0.7, 0.2, 0.6]
    _, p2 = stats.ttest_1samp(d, 0)
    t, p1 = one_sided_p(d)
    assert t > 0 and abs(p1 - p2 / 2) < 1e-12, (p1, p2)
    t, p1 = one_sided_p([-x for x in d])
    assert t < 0 and abs(p1 - (1 - p2 / 2)) < 1e-12, (p1, p2)

    # (a) futility uses only the first 3 seeds and the boundary is <= 0
    assert futility([1.0, -1.0, 0.0, 9.0, 9.0]) == (0.0, True)
    assert futility([1.0, -1.0, 0.3, -9.0, -9.0])[1] is False

    # (d) each condition can fail alone
    assert adopt([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], True)["adopt"]
    assert not adopt([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], False)["adopt"]          # Holm
    assert not adopt([-1, -1, -1, 1, 1], [1, 1, 1, 1, 1], True)["adopt"]        # val mean
    assert not adopt([1, 1, 1, 1, 1], [1, 1, 1, -1, 1], True)["adopt"]          # fresh seed 303 negative

    # (e) max-T is never narrower than per-comparison, and widens with family size
    rng = np.random.default_rng(0)
    obs = np.array([0.5, 0.3, 0.1, 0.0, -0.2])
    boot = obs + rng.multivariate_normal(np.zeros(5), 0.4 * np.eye(5) + 0.6, size=4000)
    se, per, sim, q = simultaneous_band(boot, obs)
    assert np.all(sim >= per - 1e-12), (per, sim)
    assert q > np.quantile(np.abs(boot[:, :1] - obs[:1]) / se[0], 0.95), q
    _, per2, sim2, q2 = simultaneous_band(boot[:, :2], obs[:2])
    assert q2 < q, (q2, q)                     # 5 correlated looks need a wider band than 2
    print("self-test: all assertions passed "
          "(Holm levels + step-down, one-sided p, futility boundary, adoption conditions, max-T width)")


# ---------------------------------------------------------------- max-T on real dumps

def run_maxt(specs, baseline, n, seed, alpha, expect):
    sys.path.insert(0, os.path.dirname(HERE))
    from slt.metrics import corpus_bleu4, detokenize                             # noqa: E402
    systems, refs = {}, None
    for spec in specs:
        name, path = spec.split("=", 1)
        rows = sorted((json.loads(l) for l in open(path, encoding="utf-8")), key=lambda r: r["idx"])
        R = [detokenize(r["ref"]) for r in rows]
        if refs is None:
            refs = R
        assert R == refs, f"{name}: references differ from the first system -- different split or order"
        systems[name] = [detokenize(r["hyp_beam"]) for r in rows]
    for spec in expect:                                    # guard: re-score to a number already on record
        name, value = spec.split("=", 1)
        got = corpus_bleu4(systems[name], refs, "13a")
        assert abs(got - float(value)) <= 0.0015, f"{name} re-scores to {got:.3f}, record says {value}"
        print(f"check: {name} reproduces the recorded {value}")
    names, obs, boot = bootstrap_deltas(systems, baseline, refs, n, seed)
    se, per, sim, q = simultaneous_band(boot, obs, alpha)
    k = len(names)
    print(f"\nmax-T over {k} correlated looks at the same {len(refs)} test sentences "
          f"(baseline {baseline}, B={n}, alpha={alpha})")
    print(f"  {'look':<10}{'observed':>10}{'se':>8}   per-comparison {100*(1-alpha):g}%      "
          f"max-T simultaneous {100*(1-alpha):g}%")
    for j, nm in enumerate(names):
        print(f"  {nm:<10}{obs[j]:>+10.3f}{se[j]:>8.3f}   [{obs[j]-per[j]:+6.3f}, {obs[j]+per[j]:+6.3f}]"
              f"{'  *' if obs[j]-per[j] > 0 else '   '}       "
              f"[{obs[j]-sim[j]:+6.3f}, {obs[j]+sim[j]:+6.3f}]{'  *' if obs[j]-sim[j] > 0 else ''}")
    print(f"  q(max-T) = {q:.3f} over these {k} looks.  A further look enters a family of {k+1}, and q is "
          f"non-decreasing in family size (self-test), so re-run this with the new look's own column included: "
          f"at se ~ {se.mean():.3f} the bar is AT LEAST ~{q*se.mean():+.2f}, not exactly it.")
    print("  * = interval excludes 0.  Protocol rule 9: an official-test ensemble claim needs the max-T column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--reproduce", action="store_true", help="rebuild the recorded family from results/")
    ap.add_argument("--family", help="JSON {alpha, arms:{NAME:{val:[],test:[],n_stage1:3}}}")
    ap.add_argument("--maxt", action="store_true")
    ap.add_argument("--system", action="append", default=[], help="NAME=hyps.jsonl (repeatable)")
    ap.add_argument("--baseline", default="S6")
    ap.add_argument("--expect", action="append", default=[], help="NAME=BLEU on record; abort on mismatch")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    if a.self_test:
        self_test()
    if a.reproduce:
        reproduce()
    if a.family:
        spec = json.load(open(a.family))
        report(spec["arms"], spec.get("alpha", a.alpha))
    if a.maxt:
        run_maxt(a.system, a.baseline, a.n, a.seed, a.alpha, a.expect)
    if not any((a.self_test, a.reproduce, a.family, a.maxt)):
        ap.print_help()


if __name__ == "__main__":
    main()
