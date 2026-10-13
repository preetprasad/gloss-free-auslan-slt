"""Wave-12h read-out: extra pre-training before the main training, new seeds.

Pre-registration: results/PREREG_WAVE12H_EXTRAPRETRAIN_2026-10-13.md (arms C, PJ, PS; Stage-1 seeds 505/606/707,
Stage-2 seeds 808/909). This tool prints what that file says is reported and applies the rule it fixes; it adds none.

  python tools/w12h_readout.py RUNS_DIR --train_manifest TRAIN.tsv [--expect_unseen 549 570]
  python tools/w12h_readout.py selftest
  python tools/w12h_readout.py power        (needs SciPy: run locally, Kaya's environment has none)

Only arms carrying the DONE marker written by jobs/slurm/w12h_pretrain.sbatch are read. Writes RUNS_DIR/summary_s<seed>.json
(the format tools/seed_decision.py reads) and, once Stage 1 is complete, RUNS_DIR/family_PJ.json (the format
tools/holm_decision.py --family reads; that tool needs SciPy and gives the exact one-sided p-value).
"""
from __future__ import annotations

import json, math, os, statistics as st, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_readouts import KEY, readouts   # noqa: E402  BLEU-4, WER, length ratio, content precision/recall, unseen BLEU
from seed_decision import interval       # noqa: E402  mean and two-sided 95% t-interval over seeds (n <= 7)

ARMS = ("C", "PJ", "PS")
STAGE1, STAGE2 = (505, 606, 707), (808, 909)
T95_ONE = {2: 2.920, 4: 2.132}           # one-sided 5% critical values of Student's t, by degrees of freedom
T975_DF9 = 2.262                         # two-sided 95%, df 9 (ten pooled seeds; description only)
STOP_SHA1 = "9a5644054eb66398e74f23188f9fc49b18c20b04"   # sha1 of the sorted stop list (71 words), wave-11 snapshot
CONTRASTS = (("C", "PJ"), ("C", "PS"), ("PJ", "PS"))   # (reference, arm): arm - reference
PRIMARY = ("beam5_val", "beam5_test")
SECONDARY = ("beam5_val_unseen", "beam5_test_unseen", "content_precision", "content_recall", "len_ratio", "wer",
             "distinct", "soup_beam5_val", "soup_beam5_test", "val_best", "greedy_test")
# PJ - C per seed 42/101/202/303/404 as recorded in results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md, section 11.
# These seeds suggested the hypothesis; they never enter the test below.
W5_VAL = (0.096, 0.673, 0.862, 0.904, 1.376)
W5_TEST = (-0.698, 0.156, 1.511, -0.284, 0.495)


def load_arm(d, train, stop):
    p = json.load(open(os.path.join(d, "progress.json")))
    out = {"val_best": p["best"]["bleu4"], "best_epoch": p["best"]["epoch"], "greedy_test": p["test"]["bleu4"]}
    for split in ("val", "test"):
        j = json.load(open(os.path.join(d, f"beam5_{split}_best.json")))
        r = readouts(j["beam_hyps"], j["refs"], stop, train)
        rec = j["beam_result"]["metrics"][KEY]
        assert abs(r["bleu4"] - rec) < 1e-6, f"{d} {split}: recomputed BLEU-4 {r['bleu4']} != recorded {rec}"
        out[f"beam5_{split}"] = rec
        out[f"beam5_{split}_unseen"] = r["bleu4_unseen"]
        out[f"unseen_n_{split}"] = r["unseen_n"]
        if split == "test":
            out.update({k: r[k] for k in ("content_precision", "content_recall", "len_ratio", "wer", "distinct")})
        soup = json.load(open(os.path.join(d, f"beam5_{split}_soup_last5.json")))
        out[f"soup_beam5_{split}"] = soup["beam_result"]["metrics"][KEY]
    return out


def collect(runs, train, stop, expect_unseen=None):
    """{seed: {arm: read-outs}} over the arms that carry DONE; writes summary_s<seed>.json."""
    rows = {}
    for s in STAGE1 + STAGE2:
        got = {}
        for arm in ARMS:
            d = os.path.join(runs, f"{arm}_s{s}")
            if not os.path.exists(os.path.join(d, "DONE")):
                if os.path.isdir(d):
                    print(f"  {arm}_s{s}: no DONE marker (failed or unfinished) -- excluded")
                continue
            got[arm] = load_arm(d, train, stop)
            if arm != "C":   # the first cycle is evaluated once, at its last epoch; that score is logged, never used
                pre = json.load(open(os.path.join(runs, f"pre_{arm}_s{s}", "progress.json")))
                assert pre["best"]["epoch"] == pre["epochs"] and not pre["test_set_used"], f"pre_{arm}_s{s}: {pre['best']}"
                got[arm]["pre_val_final"] = pre["best"]["bleu4"]
            if expect_unseen:
                n = (got[arm]["unseen_n_val"], got[arm]["unseen_n_test"])
                assert n == tuple(expect_unseen), f"{arm}_s{s}: unseen-target clips {n}, expected {expect_unseen}"
        if got:
            rows[s] = got
            json.dump(got, open(os.path.join(runs, f"summary_s{s}.json"), "w"), indent=1)
    return rows


def deltas(rows, seeds, ref, arm, key):
    """Paired differences arm - ref over the given seeds, only where both arms finished."""
    return [rows[s][arm][key] - rows[s][ref][key] for s in seeds if s in rows and arm in rows[s] and ref in rows[s]]


def decide(val1, test1, val2, test2):
    """The pre-registered rule for PJ - C. Arguments: paired beam-5 deltas of the Stage-1 and Stage-2 seeds."""
    if len(val1) < len(STAGE1) or len(test1) < len(STAGE1):
        return {"stage": "incomplete: Stage 1 needs all three seeds of C and PJ", "adopt": False}
    fut = st.mean(val1)
    if fut <= 0:
        return {"stage": "STOPPED at Stage 1 for futility (mean validation difference <= 0)",
                "futility_mean_val": fut, "adopt": False}
    if len(val2) < len(STAGE2) or len(test2) < len(STAGE2):
        return {"stage": "Stage 1 passed: Stage 2 (seeds 808 and 909) is due; no adoption decision at Stage 1",
                "futility_mean_val": fut, "adopt": False}
    val, test = list(val1) + list(val2), list(test1) + list(test2)
    sd = st.stdev(test)
    t = st.mean(test) / (sd / math.sqrt(len(test))) if sd > 0 else (math.inf if st.mean(test) > 0 else -math.inf)
    crit = T95_ONE[len(test) - 1]
    cond = {"t_rejects_at_0.05_one_sided": t > crit, "val_mean_positive": st.mean(val) > 0,
            "both_stage2_seeds_positive_on_test": all(x > 0 for x in test2)}
    return {"stage": "final (five new seeds)", "futility_mean_val": fut, "t": t, "t_crit": crit, **cond,
            "adopt": all(cond.values())}


def reading(val, test):
    """Section 7: a direction is stated only with five seeds, a validation mean of that sign and a test interval
    wholly on that side of zero; otherwise the contrast is a bound."""
    if len(test) < len(STAGE1) + len(STAGE2):
        return "bound (fewer than five seeds: no direction stated)"
    _, lo, hi = interval(test)
    if st.mean(val) > 0 and lo > 0:
        return "POSITIVE (validation mean > 0 and test interval wholly above zero)"
    if st.mean(val) < 0 and hi < 0:
        return "NEGATIVE (validation mean < 0 and test interval wholly below zero)"
    return "bound (no direction resolved)"


def line(label, d):
    if not d:
        return f"  {label:32s} no complete pair"
    m, lo, hi = interval(d)
    return (f"  {label:32s} n={len(d)} per-seed " + " ".join(f"{x:+.3f}" for x in d)
            + f" | positive {sum(x > 0 for x in d)}/{len(d)} | mean {m:+.3f} | 95% t [{lo:+.3f}, {hi:+.3f}]")


def report(rows, runs=None):
    seeds = [s for s in STAGE1 + STAGE2 if s in rows]
    print(f"seeds with at least one finished arm: {seeds}")
    for arm in ARMS:
        got = [(s, rows[s][arm]) for s in seeds if arm in rows[s]]
        if got:
            print(f"  {arm:3s} beam-5 val/test " + "  ".join(f"s{s} {r['beam5_val']:.3f}/{r['beam5_test']:.3f}" for s, r in got)
                  + f" | means {st.mean(r['beam5_val'] for _, r in got):.3f} / {st.mean(r['beam5_test'] for _, r in got):.3f}")
    for arm in ARMS[1:]:
        got = [(s, rows[s][arm]["pre_val_final"]) for s in seeds if arm in rows[s]]
        print(f"  {arm:3s} first cycle, greedy validation BLEU-4 of its final weights: " + "  ".join(f"s{s} {v:.3f}" for s, v in got))
    for ref, arm in CONTRASTS:
        print(f"=== {arm} - {ref} ===")
        for key in PRIMARY + SECONDARY:
            print(line(key, deltas(rows, STAGE1 + STAGE2, ref, arm, key)))
    d = decide(deltas(rows, STAGE1, "C", "PJ", "beam5_val"), deltas(rows, STAGE1, "C", "PJ", "beam5_test"),
               deltas(rows, STAGE2, "C", "PJ", "beam5_val"), deltas(rows, STAGE2, "C", "PJ", "beam5_test"))
    print("=== PRIMARY, family {PJ - C}, m = 1, one-sided alpha 0.05 (pre-registration section 6) ===")
    for k, v in d.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  => PJ {'ADOPTED' if d['adopt'] else 'NOT adopted'}")
    new_v, new_t = (deltas(rows, STAGE1 + STAGE2, "C", "PJ", k) for k in PRIMARY)
    if len(new_v) >= len(STAGE1) and runs:
        json.dump({"alpha": 0.05, "arms": {"PJ": {"val": new_v, "test": new_t, "n_stage1": len(STAGE1)}}},
                  open(os.path.join(runs, "family_PJ.json"), "w"), indent=1)
        print(f"  wrote {os.path.join(runs, 'family_PJ.json')} for tools/holm_decision.py --family (exact p; needs SciPy)")
    if len(new_t) == len(STAGE1) + len(STAGE2):
        for name, old, new in (("validation", W5_VAL, new_v), ("test", W5_TEST, new_t)):
            x = list(old) + new; hw = T975_DF9 * st.stdev(x) / math.sqrt(len(x))
            print(f"  DESCRIPTION ONLY, never a test: ten seeds (five of wave 5 + five new), PJ - C beam-5 {name} "
                  f"mean {st.mean(x):+.3f} [{st.mean(x) - hw:+.3f}, {st.mean(x) + hw:+.3f}]")
    print("=== SECONDARY (estimates; intervals uncorrected; never a criterion) ===")
    for ref, arm in (("C", "PS"), ("PJ", "PS")):
        v, t = (deltas(rows, STAGE1 + STAGE2, ref, arm, k) for k in PRIMARY)
        if len(t) >= 2:
            m, lo, hi = interval(t)
            print(f"  {arm} - {ref}: validation mean {st.mean(v):+.3f}; test {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(t)} seeds "
                  f"-> {reading(v, t)}")
    uv, u = (deltas(rows, STAGE1 + STAGE2, "C", "PJ", k) for k in ("beam5_val_unseen", "beam5_test_unseen"))
    if len(u) >= 2:
        m, lo, hi = interval(u)
        print(f"  PJ - C on targets unseen in training: validation mean {st.mean(uv):+.3f}; test {m:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"over {len(u)} seeds -> {reading(uv, u)}")
    return d


def power():
    from scipy import stats
    sd = st.stdev(W5_TEST)
    print(f"sd of the five recorded wave-5 PJ - C beam-5 test differences: {sd:.3f} (mean {st.mean(W5_TEST):+.3f})")
    for n in (3, 5):
        df = n - 1; crit = stats.t.ppf(0.95, df)
        assert abs(crit - T95_ONE[df]) < 5e-4, (crit, T95_ONE[df])
        print(f"n = {n} new seeds: one-sided 5% critical t {crit:.3f}; observed mean needed at this sd {crit * sd / math.sqrt(n):+.2f}")
        for eff in (0.24, 0.50, 0.78, 1.00, 1.50):
            print(f"   true difference {eff:+.2f}: power {1 - stats.nct.cdf(crit, df, eff / (sd / math.sqrt(n))):.2f}")


def selftest():
    # 1. the recorded wave-5 PJ - C differences: the rule must return "not adopted" for the reasons on record
    d = decide(W5_VAL[:3], W5_TEST[:3], W5_VAL[3:], W5_TEST[3:])
    m, lo, hi = interval(list(W5_TEST))
    assert abs(m - 0.236) < 2e-3 and abs(lo + 0.811) < 2e-3 and abs(hi - 1.283) < 2e-3, (m, lo, hi)   # PREREG_WAVE5 s.11
    assert d["stage"].startswith("final") and not d["adopt"] and not d["t_rejects_at_0.05_one_sided"], d
    assert d["val_mean_positive"] and not d["both_stage2_seeds_positive_on_test"] and abs(d["t"] - 0.626) < 5e-3, d
    # 2. a clear gain is adopted; 3. one negative Stage-2 seed blocks it; 4. futility; 5. Stage 1 only; 6. incomplete
    assert decide([.5, .4, .6], [1.0, 1.2, 0.9], [.5, .5], [1.1, 1.0])["adopt"]
    assert not decide([.5, .4, .6], [1.0, 1.2, 0.9], [.5, .5], [1.1, -0.01])["adopt"]
    assert not decide([.5, .4, .6], [1.0, 1.2, 0.9], [-2.0, -2.0], [1.1, 1.0])["adopt"]          # 5-seed val mean <= 0
    assert decide([-.5, .2, .1], [9, 9, 9], [1, 1], [9, 9])["stage"].startswith("STOPPED")
    assert decide([.5, .2, .1], [1, 1.1, 1.2], [], [])["stage"].startswith("Stage 1 passed")
    assert decide([.5, .2], [1, 1.1], [1, 1], [1, 1])["stage"].startswith("incomplete")
    assert reading([-.1, -.2, -.3, -.2, -.1], [-1.0, -1.2, -0.9, -1.1, -1.0]).startswith("NEGATIVE")
    assert reading([.1, .2, .3, .2, .1], [1.0, 1.2, 0.9, 1.1, 1.0]).startswith("POSITIVE")
    assert reading([.1, .2, .3, .2, .1], [-1.0, -1.2, -0.9, -1.1, -1.0]).startswith("bound")     # signs disagree
    assert reading([-.1, -.2, -.3, -.2, -.1], [-1.0, 1.2, -0.9, -1.1, -1.0]).startswith("bound")  # interval includes zero
    assert reading([-.1, -.2, -.3], [-1.0, -1.2, -0.9]).startswith("bound (fewer")
    try:   # agreement with tools/holm_decision.py where SciPy exists (it does not on Kaya)
        from holm_decision import holm, one_sided_p
        t, p = one_sided_p(list(W5_TEST))
        assert abs(p - 0.283) < 2e-3 and abs(t - d["t"]) < 1e-9, (t, p)                             # PREREG_WAVE5 s.11
        for test in ([1.0, 1.2, 0.9, 1.1, 1.0], [0.9, -0.4, 1.4, 0.3, 0.8], [0.05, 0.30, 0.61, 0.20, 0.42]):
            mine = decide([1, 1, 1], test[:3], [1, 1], test[3:])["t_rejects_at_0.05_one_sided"]
            assert mine == holm([("PJ", one_sided_p(test)[1])], 0.05)[0][3], test
        note = "holm_decision.py agrees"
    except ImportError:
        note = "holm_decision.py not importable here (no SciPy): agreement check skipped"
    # 7. files: DONE gate, summary files, the BLEU recomputation guard
    stop, train = {"the", "is"}, {"the cat sat ."}
    refs = ["the cat sat .", "a bird flies high .", "we go home now ."]
    with tempfile.TemporaryDirectory() as tmp:
        for s in STAGE1[:2]:
            for i, arm in enumerate(ARMS):
                dd = os.path.join(tmp, f"{arm}_s{s}"); os.makedirs(dd)
                hyps = [refs[0], refs[1] if i else "a dog runs .", refs[2] if i > 1 else "we go ."]
                dump = {"beam_hyps": hyps, "refs": refs,
                        "beam_result": {"metrics": {KEY: readouts(hyps, refs, stop, train)["bleu4"]}}}
                for split in ("val", "test"):
                    for ck in ("best", "soup_last5"):
                        json.dump(dump, open(os.path.join(dd, f"beam5_{split}_{ck}.json"), "w"))
                json.dump({"best": {"bleu4": 1.0 + i, "epoch": 3}, "test": {"bleu4": 2.0}}, open(os.path.join(dd, "progress.json"), "w"))
                if arm != "C":
                    os.makedirs(os.path.join(tmp, f"pre_{arm}_s{s}"))
                    json.dump({"best": {"bleu4": 0.5, "epoch": 9}, "epochs": 9, "test_set_used": False},
                              open(os.path.join(tmp, f"pre_{arm}_s{s}", "progress.json"), "w"))
                if not (s == STAGE1[1] and arm == "PS"):
                    open(os.path.join(dd, "DONE"), "w").close()
        rows = collect(tmp, train, stop, expect_unseen=(2, 2))
        assert set(rows[STAGE1[0]]) == set(ARMS) and set(rows[STAGE1[1]]) == {"C", "PJ"}, rows.keys()
        assert rows[STAGE1[0]]["PS"]["pre_val_final"] == 0.5 and "pre_val_final" not in rows[STAGE1[0]]["C"]
        assert json.load(open(os.path.join(tmp, f"summary_s{STAGE1[0]}.json")))["PJ"]["beam5_test"] > rows[STAGE1[0]]["C"]["beam5_test"]
        assert len(deltas(rows, STAGE1, "PJ", "PS", "beam5_test")) == 1 and len(deltas(rows, STAGE1, "C", "PJ", "wer")) == 2
        assert report(rows, tmp)["stage"].startswith("incomplete") and not os.path.exists(os.path.join(tmp, "family_PJ.json"))
        bad = os.path.join(tmp, f"C_s{STAGE1[0]}", "beam5_test_best.json")
        j = json.load(open(bad)); j["beam_result"]["metrics"][KEY] += 0.5; json.dump(j, open(bad, "w"))
        try:
            collect(tmp, train, stop); raise SystemExit("selftest FAILED: a wrong recorded BLEU was accepted")
        except AssertionError:
            pass
    print(f"selftest passed ({note})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--expect_unseen", type=int, nargs=2, metavar=("VAL", "TEST"),
                    help="abort unless every arm has exactly this many unseen-target validation / test clips")
    a = ap.parse_args()
    import hashlib
    from stitch_lexicon import STOP
    assert len(STOP) == 71 and hashlib.sha1(" ".join(sorted(STOP)).encode()).hexdigest() == STOP_SHA1, \
        "stitch_lexicon.STOP is not the 71-word stop list of the earlier read-outs"
    train = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split())
             for l in open(a.train_manifest, encoding="utf-8") if "\t" in l}
    report(collect(a.runs, train, STOP, a.expect_unseen), a.runs)


if __name__ == "__main__":
    {"selftest": selftest, "power": power}.get(sys.argv[1] if len(sys.argv) == 2 else "", main)()
