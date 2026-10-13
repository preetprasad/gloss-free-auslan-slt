#!/usr/bin/env python3
"""Wave 12a read-out: the adopted News bundle split into stitched clips and pose augmentation.

Rules: results/PREREG_WAVE12A_NEWSDECOMP_2026-10-13.md, written before any nS number existed. Arms, all trained in
ONE array job (jobs/slurm/w12a_news_decomp.sbatch) at seeds 42 / 101 / 202:
    nA  real News clips only          nS  + 10,614 stitched clips, no pose augmentation
    nSA + stitched clips + pose augmentation (the adopted bundle)          nRA real + pose augmentation (optional)
Per seed, beam-5 BLEU-4 of best.pt (sacreBLEU 13a on detokenised text):
    S = nS - nA        A = nSA - nS        B = nSA - nA = S + A

  python tools/w12a_news_decomp_readout.py --runs RUNS --labels LABELS_DIR --news_train T --news_val V --news_test E
  python tools/w12a_news_decomp_readout.py --self_test

Reused without change: seed_decision.interval (two-sided 95% seed-level t-interval), arm_readouts.readouts (BLEU
re-score, WER, length ratio, content precision and recall, unseen-target BLEU) and news_signable_recall.recall. The
type gate of the mandatory News secondary (PREREG_WAVE4, addenda of 22 and 24 Sep) is repeated here because that
tool keeps it inside main(). No SciPy: it is not installed on Kaya.
"""
import argparse, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from seed_decision import interval                                        # noqa: E402
from arm_readouts import KEY, readouts                                    # noqa: E402
from news_signable_recall import recall, texts                            # noqa: E402
from stitch_lexicon import STOP, build_lexicon, content_words, coverage   # noqa: E402
from slt.metrics import corpus_bleu4, detokenize                          # noqa: E402

SEEDS = (42, 101, 202)
CORE = ("nA", "nS", "nSA")
# contrast -> (arm coefficients, what it measures). I needs the optional arm nRA.
CONTRASTS = {
    "S": ({"nS": 1, "nA": -1}, "stitched clips without augmentation (with their extra steps and output types)"),
    "A": ({"nSA": 1, "nS": -1}, "pose augmentation on top of the stitched clips"),
    "B": ({"nSA": 1, "nA": -1}, "the adopted bundle"),
    "R": ({"nRA": 1, "nA": -1}, "pose augmentation without stitched clips (optional arm)"),
    "I": ({"nSA": 1, "nS": -1, "nRA": -1, "nA": 1}, "interaction: A minus R (optional arm)"),
}
# beam-5 validation / test of job 7950, three decimals (results/news_stage2_decision/decision.txt). Used for one note only.
RECORDED = {"nA": {42: (2.009, 1.873), 101: (2.218, 1.699), 202: (2.004, 1.874)},
            "nSA": {42: (1.821, 2.103), 101: (2.257, 2.302), 202: (2.357, 2.273)}}
# in-vocab content recall on test, signable / unsignable, two decimals (results/news_stage2_decision/signable_recall.txt)
RECORDED_RECALL = {"nA": {42: (4.49, 1.84), 101: (4.53, 2.37), 202: (4.39, 2.03)},
                   "nSA": {42: (5.43, 2.69), 101: (6.10, 2.09), 202: (5.89, 3.10)}}
SOUP = ("soup_val", "soup_test")      # last-5 weight average: descriptive, never a criterion
METRICS = ("beam5_val", "beam5_test", "greedy_val", "bleu4_unseen", "content_precision", "content_recall",
           "len_ratio", "wer", "signable", "unsignable")


# ------------------------------------------------------------------ the pre-registered rule (sections 5 and 6)

def summarise(d):
    m, lo, hi = interval(d)
    return {"per_seed": d, "n": len(d), "positive": sum(x > 0 for x in d), "mean": m, "lo": lo, "hi": hi}


def component(val, test):
    """Status of one component (S or A) from its validation and test summaries."""
    if val["mean"] > 0 and test["positive"] == test["n"] and test["lo"] > 0:
        return "resolved"          # test interval above zero = one-sided p <= 0.025 (Bonferroni over S and A)
    if val["mean"] <= 0 and test["mean"] <= 0:
        return "absent"
    return "unresolved"


def reading(S, A, B):
    """S, A, B: {"val": summary, "test": summary}. Returns (code, sentence)."""
    if not (B["val"]["mean"] > 0 and B["test"]["mean"] > 0):
        return "R0", "the gain of the bundle is not reproduced in this job at three seeds; no decomposition is read"
    s, a = component(S["val"], S["test"]), component(A["val"], A["test"])
    if s == "resolved":
        tail = {"resolved": "augmentation adds a further resolved gain",
                "absent": "no further gain from augmentation was measured",
                "unresolved": "the further effect of augmentation is not resolved"}[a]
        return "R1", "the gain remains without augmentation; " + tail
    if s == "absent":
        return "R2", ("no gain without augmentation was measured at three seeds (mean at or below zero on validation "
                      "and on test); augmentation is " + a)
    if a == "resolved":
        return "R3", "augmentation adds a resolved gain on top of the stitched clips; the part without it is not resolved"
    return "R4", "not attributed at three seeds: the gain stays reported for the bundle"


def secondary_pattern(sig, unsig):
    """Mandatory News secondary for one contrast. A class rises if its mean is > 0 and at most one seed is not positive
    (the 24 Sep addendum of PREREG_WAVE4 read 4 of 5 seeds as a rise; at three seeds this is 2 of 3)."""
    up = lambda x: x["mean"] > 0 and x["positive"] >= x["n"] - 1
    return {(True, True): "both", (True, False): "signable only", (False, True): "unsignable only",
            (False, False): "neither"}[(up(sig), up(unsig))]


# Readings of the pattern, fixed in PREREG_WAVE4 section 2 for the stitched-clip lever. They apply to S and to B.
# A, R and I change no sign content, so their pattern is reported without a reading.
SECONDARY_READING = {"signable only": "consistent with grounding of the signs the lexicon covers",
                     "both": "not specific to signable types, so not grounding (regularisation, the extra optimiser "
                             "steps or, in B, augmentation)",
                     "unsignable only": "not grounding", "neither": "no transfer shown"}


# ------------------------------------------------------------------ data

def news_classes(labels, news_train, news_test):
    """The pinned definition of the mandatory secondary; the same code and gate as news_signable_recall.main()."""
    lex = build_lexicon([os.path.join(labels, "Train.json"), os.path.join(labels, "Valid.json")],
                        morphology=True, paren_alias=False)
    refs = texts(news_test)
    types = {w for t in refs for w in content_words(t.lower())}
    sign = {w for w in types if coverage(w + " .", lex)[0] > 0}
    inv = types & {w for t in texts(news_train) for w in t.lower().split()}
    counts = (len(sign), len(types - sign), len(inv), len(inv & sign), len(inv - sign))
    assert counts == (934, 1821, 2379, 878, 1501), f"GATE FAILED: {counts} != (934, 1821, 2379, 878, 1501)"
    print("secondary gate OK: 934 / 1,821 signable / unsignable test types; 2,379 in-vocab (878 / 1,501)")
    return {"signable": inv & sign, "unsignable": inv - sign}


def load_run(d, refs, train_targets, classes):
    """One finished run -> metric row. Every beam-5 dump is re-scored and must equal its recorded BLEU."""
    p = json.load(open(os.path.join(d, "progress.json")))
    assert p.get("status") == "done", f"{d}: training did not finish"
    row = {"greedy_val": p["best"]["bleu4"], "epoch": p["best"]["epoch"]}
    ws = lambda xs: [" ".join(x.split()) for x in xs]
    for split in ("val", "test"):
        j = json.load(open(os.path.join(d, f"beam5_{split}_best.json")))
        assert ws(j["refs"]) == ws(refs[split]), f"{d}: {split} references differ from the manifest"
        r = readouts(j["beam_hyps"], j["refs"], STOP, train_targets)
        assert abs(r["bleu4"] - j["beam_result"]["metrics"][KEY]) < 1e-6, f"{d}: {split} BLEU does not re-score"
        row[f"beam5_{split}"] = r["bleu4"]
        if split == "test":
            row.update({k: r[k] for k in ("bleu4_unseen", "unseen_n", "content_precision", "content_recall",
                                          "len_ratio", "wer", "distinct")})
            row.update({c: recall(j["beam_hyps"], j["refs"], keep) for c, keep in classes.items()})
            row["hyps"] = j["beam_hyps"]
    for split in ("val", "test"):
        try:
            row[f"soup_{split}"] = json.load(open(os.path.join(d, f"beam5_{split}_soup_last5.json")))["beam_result"]["metrics"][KEY]
        except (OSError, KeyError, ValueError):
            row[f"soup_{split}"] = float("nan")
    try:
        row["job"] = open(os.path.join(d, "slurm_job.txt")).read().strip().splitlines()[-1].split()[0].split("_")[0]
    except (OSError, IndexError):
        row["job"] = "?"
    return row


def contrast(rows, coef, metric, seeds=SEEDS):
    return summarise([sum(c * rows[(arm, s)][metric] for arm, c in coef.items()) for s in seeds])


def pooled_bootstrap(rows, refs, names, n, seed=0):
    """Paired sentence bootstrap pooled over seeds: one resample of the test sentences per draw, shared by every arm
    and seed; the statistic is the mean over seeds of the contrast in corpus BLEU-4. TEST SAMPLING ONLY: it holds the
    trained models fixed, so it does not contain seed-to-seed variation."""
    import numpy as np
    arms = sorted({a for k in names for a in CONTRASTS[k][0]})
    H = {(a, s): [detokenize(h) for h in rows[(a, s)]["hyps"]] for a in arms for s in SEEDS}
    R = [detokenize(r) for r in refs]
    rng = np.random.default_rng(seed); N = len(R); out = {k: [] for k in names}
    for _ in range(n):
        ii = rng.integers(0, N, N); Rb = [R[i] for i in ii]
        b = {k: corpus_bleu4([v[i] for i in ii], Rb, "13a") for k, v in H.items()}
        for k in names:
            out[k].append(st.mean(sum(c * b[(a, s)] for a, c in CONTRASTS[k][0].items()) for s in SEEDS))
    return {k: {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)),
                "share_positive": sum(x > 0 for x in v) / n} for k, v in out.items()}


# ------------------------------------------------------------------ report

def fmt(x):
    return (f"{' '.join(f'{v:+.3f}' for v in x['per_seed'])} | positive {x['positive']}/{x['n']} | "
            f"mean {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]")


def main(a):
    refs = {"val": texts(a.news_val), "test": texts(a.news_test)}
    train_targets = {" ".join(t.lower().split()) for t in texts(a.news_train)}
    classes = news_classes(a.labels, a.news_train, a.news_test)
    rows, missing = {}, []
    for arm in CORE + ("nRA",):
        for s in SEEDS:
            d = os.path.join(a.runs, f"{arm}_s{s}")
            done = os.path.exists(os.path.join(a.runs, f"DONE_{arm}_s{s}"))
            if not (done and os.path.isdir(d)):
                if arm in CORE or os.path.isdir(d):
                    missing.append(f"{arm}_s{s}")
                continue
            rows[(arm, s)] = load_run(d, refs, train_targets, classes)
    arms = [arm for arm in CORE + ("nRA",) if all((arm, s) in rows for s in SEEDS)]
    print(f"\nunseen test targets (reference not verbatim in the real News training text): "
          f"{next(iter(rows.values()))['unseen_n'] if rows else '?'} of {len(refs['test'])}")
    print(f"\n{'run':9s} job      ep  greedy  b5 val  b5 test  unseen  cont P  cont R  len    WER    signable unsign. distinct")
    for (arm, s), r in rows.items():
        print(f"{arm + '_s' + str(s):9s} {r['job']:8s} {r['epoch']:3d} {r['greedy_val']:6.3f}  {r['beam5_val']:6.3f}  "
              f"{r['beam5_test']:6.3f}  {r['bleu4_unseen']:6.3f}  {r['content_precision']:5.2f}  {r['content_recall']:5.2f}  "
              f"{r['len_ratio']:5.3f}  {r['wer']:5.3f}  {r['signable']:6.2f}  {r['unsignable']:6.2f}  {r['distinct']:5d}")
    for arm in arms:
        print(f"mean {arm:4s}: beam-5 validation {st.mean(rows[(arm, s)]['beam5_val'] for s in SEEDS):.3f} | "
              f"test {st.mean(rows[(arm, s)]['beam5_test'] for s in SEEDS):.3f}")
    jobs = sorted({r["job"] for r in rows.values()})
    if len(jobs) > 1:
        print(f"\nRUNS FROM MORE THAN ONE JOB: {jobs}. A reading is allowed only for one rerun of a task that died of a "
              "node fault before its first epoch, recorded in the outcome (pre-registration, section 9).")
    for arm, rec in RECORDED.items():
        if arm in arms:
            same = all(abs(rows[(arm, s)][f"beam5_{sp}"] - rec[s][i]) < 5.1e-4 for s in SEEDS for i, sp in enumerate(("val", "test")))
            print(f"note: {arm} {'REPRODUCES' if same else 'DOES NOT reproduce'} the beam-5 scores of job 7950 to three decimals "
                  "(a note on repeatability across jobs; never a criterion)")
            # same decodes -> the recall definition (lexicon, content words) must give the recorded recall as well
            assert not same or all(abs(rows[(arm, s)][c] - RECORDED_RECALL[arm][s][i]) < 6e-3
                                   for s in SEEDS for i, c in enumerate(("signable", "unsignable"))), \
                f"GATE FAILED: {arm} reproduces job 7950 but its signable / unsignable recall differs from the record"
    if missing or not all(arm in arms for arm in CORE):
        print(f"\nNOT ALL NINE RUNS ARE COMPLETE (missing or without a DONE marker: {missing}). NO READING IS MADE.")
        return None
    names = [k for k, (coef, _) in CONTRASTS.items() if all(arm in arms for arm in coef)]
    out = {"jobs": jobs, "contrasts": {}, "runs": {f"{arm}_s{s}": {k: v for k, v in r.items() if k != "hyps"}
                                                  for (arm, s), r in rows.items()}}
    for k in names:
        coef, what = CONTRASTS[k]
        out["contrasts"][k] = {m: contrast(rows, coef, m) for m in METRICS}
        print(f"\n===== {k} = {' '.join(('+' if c > 0 else '-') + arm for arm, c in coef.items())}: {what} =====")
        for m in METRICS:
            print(f"  {m:18s} {fmt(out['contrasts'][k][m])}")
    for m in ("beam5_val", "beam5_test"):      # the decomposition is exact per seed
        for i in range(len(SEEDS)):
            c = out["contrasts"]
            assert abs(c["S"][m]["per_seed"][i] + c["A"][m]["per_seed"][i] - c["B"][m]["per_seed"][i]) < 1e-9
    if a.boot:
        out["bootstrap"] = pooled_bootstrap(rows, refs["test"], names, a.boot)
        print(f"\npooled paired sentence bootstrap on test, {a.boot} resamples (TEST SAMPLING ONLY, no seed variation):")
        for k, b in out["bootstrap"].items():
            print(f"  {k}: 95% [{b['lo']:+.3f}, {b['hi']:+.3f}] | share of resamples above zero {100 * b['share_positive']:.1f}%")
    c = out["contrasts"]
    pick = lambda k: {"val": c[k]["beam5_val"], "test": c[k]["beam5_test"]}
    code, sentence = reading(pick("S"), pick("A"), pick("B"))
    out["components"] = {k: component(c[k]["beam5_val"], c[k]["beam5_test"]) for k in ("S", "A")}
    out["reading"] = {"code": code, "sentence": sentence}
    print(f"\ncomponent status (section 5): S {out['components']['S']} | A {out['components']['A']}")
    print(f"PRE-REGISTERED READING {code}: {sentence}")
    if c["B"]["beam5_test"]["mean"] > 0:
        out["share_S_test"] = c["S"]["beam5_test"]["mean"] / c["B"]["beam5_test"]["mean"]
        print(f"share of the bundle's mean test gain measured without augmentation: {100 * out['share_S_test']:.0f}% "
              "(a ratio of two noisy means; a point estimate, never a claim)")
    out["secondary"] = {k: secondary_pattern(c[k]["signable"], c[k]["unsignable"]) for k in names}
    print("\nmandatory News secondary, in-vocab content recall on test: which class rises (mean > 0, at most one seed "
          "not positive; descriptive)")
    out["secondary_relative"] = {}
    for k, v in out["secondary"].items():
        coef = CONTRASTS[k][0]
        pair = [arm for arm, x in coef.items() if x > 0], [arm for arm, x in coef.items() if x < 0]
        rel = ""
        if len(pair[0]) == 1 and len(pair[1]) == 1:       # S, A, B, R compare two arms; the interaction I does not
            mean = lambda arm, m: st.mean(rows[(arm, s)][m] for s in SEEDS)
            r = {m: mean(pair[0][0], m) / mean(pair[1][0], m) - 1 for m in ("signable", "unsignable")}
            out["secondary_relative"][k] = r
            rel = f" | relative to {pair[1][0]}: signable {100 * r['signable']:+.0f}%, unsignable {100 * r['unsignable']:+.0f}%"
        print(f"  {k}: {v}" + (f" -> {SECONDARY_READING[v]}" if k in ("S", "B") else "") + rel)
    soup = [m for m in SOUP if all(rows[(arm, s)][m] == rows[(arm, s)][m] for arm in arms for s in SEEDS)]   # not NaN
    if len(soup) == len(SOUP):
        out["soup"] = {k: {m: contrast(rows, CONTRASTS[k][0], m) for m in SOUP} for k in names}
        print("\nlast-5 weight average, beam-5 (descriptive; NEVER a criterion and not used by the reading):")
        for k in names:
            for m in SOUP:
                print(f"  {k} {m:10s} {fmt(out['soup'][k][m])}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return out


def self_test():
    sm = summarise
    # interval: the recorded three-seed bundle deltas of job 7950 (PREREG_WAVE4, correction of 24 Sep)
    t = sm([0.231, 0.602, 0.399]); assert abs(t["mean"] - 0.411) < 1e-3 and abs(t["lo"] + 0.051) < 2e-3 and abs(t["hi"] - 0.872) < 2e-3, t
    v = sm([-0.188, 0.040, 0.353]); assert abs(v["mean"] - 0.068) < 1e-3 and abs(v["lo"] + 0.606) < 2e-3, v
    tight, zero, neg = sm([0.30, 0.35, 0.40]), sm([0.10, -0.20, 0.25]), sm([-0.10, -0.30, 0.05])
    assert tight["lo"] > 0 and zero["lo"] < 0 < zero["hi"]
    # component: every branch, and each condition of "resolved" can fail alone
    assert component(tight, tight) == "resolved"
    assert component(neg, tight) == "unresolved"                       # validation mean not above zero
    assert component(tight, sm([0.9, 0.9, -0.1])) == "unresolved"      # one seed negative on test
    assert component(tight, sm([0.1, 0.5, 0.9])) == "unresolved"       # 3/3 positive but the interval includes zero
    assert component(neg, neg) == "absent" and component(neg, zero) == "unresolved" and component(zero, neg) == "unresolved"
    assert component(sm([0.0, 0.0, 0.0]), sm([0.0, 0.0, 0.0])) == "absent"      # the boundary is "at or below zero"
    pk = lambda val, test: {"val": val, "test": test}
    B = pk(tight, tight)
    assert reading(pk(tight, tight), pk(tight, tight), pk(neg, tight))[0] == "R0"      # bundle absent on validation
    assert reading(pk(tight, tight), pk(tight, tight), pk(tight, neg))[0] == "R0"      # bundle absent on test
    assert reading(pk(tight, tight), pk(zero, zero), B)[0] == "R1"
    assert "no further gain" in reading(pk(tight, tight), pk(neg, neg), B)[1]
    assert reading(pk(neg, neg), pk(tight, tight), B)[0] == "R2"
    assert reading(pk(zero, zero), pk(tight, tight), B)[0] == "R3"
    assert reading(pk(zero, zero), pk(zero, zero), B)[0] == "R4" and reading(pk(zero, zero), pk(neg, neg), B)[0] == "R4"
    # the bundle as recorded in job 7950 would itself be unresolved at three seeds
    assert component(v, t) == "unresolved"
    assert [secondary_pattern(x, y) for x, y in ((tight, tight), (tight, neg), (neg, tight), (neg, neg))] == \
        ["both", "signable only", "unsignable only", "neither"]
    two_of_three, one_of_three = sm([0.5, 0.4, -0.1]), sm([0.9, -0.1, -0.2])
    assert two_of_three["mean"] > 0 and two_of_three["positive"] == 2 and secondary_pattern(two_of_three, neg) == "signable only"
    assert one_of_three["mean"] > 0 and one_of_three["positive"] == 1 and secondary_pattern(one_of_three, neg) == "neither"
    assert set(SECONDARY_READING) == {"both", "signable only", "unsignable only", "neither"}
    # contrasts: the coefficients give S + A = B, and I = A - R
    rows = {(arm, s): {"m": x + s / 1000} for arm, x in (("nA", 1.0), ("nS", 1.3), ("nSA", 1.5), ("nRA", 1.1)) for s in SEEDS}
    g = lambda k: contrast(rows, CONTRASTS[k][0], "m")["mean"]
    assert abs(g("S") + g("A") - g("B")) < 1e-12 and abs(g("I") - (g("A") - g("R"))) < 1e-12 and abs(g("S") - 0.3) < 1e-12
    # bootstrap: identical systems give exactly zero; a better system is above zero in most resamples
    ref = [f"the cat sat on the mat number {i} today ." for i in range(40)]
    bad = [f"a dog ran in the park number {i} today ." for i in range(40)]
    r2 = {(arm, s): {"hyps": h} for s in SEEDS for arm, h in (("nA", bad), ("nS", bad), ("nSA", ref))}
    b = pooled_bootstrap(r2, ref, ["S", "A"], 50)
    assert b["S"]["lo"] == b["S"]["hi"] == 0.0 and b["A"]["lo"] > 0 and b["A"]["share_positive"] == 1.0, b
    print("self-test passed (interval, component, reading R0-R4, secondary pattern, contrast algebra, bootstrap)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--runs"); ap.add_argument("--labels"); ap.add_argument("--news_train")
    ap.add_argument("--news_val"); ap.add_argument("--news_test")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--out", default="")
    args = ap.parse_args()
    self_test() if args.self_test else main(args)
