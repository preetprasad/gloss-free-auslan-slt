#!/usr/bin/env python3
"""Independent verifier for the News decoding test (PREREG_WAVE4 addendum 24 Sep 16:10; job 27382).
Does NOT import tools/news_decode_analysis.py or bp_readout.py. Reads raw dumps/logs/json itself and scores
through slt/metrics.py (corpus_bleu4 on detokenize()d text, 13a). Self-checks reproduce recorded numbers first.
Run from src:  python ../results/news_decoding_length_check/verify_news_decode.py
"""
import json, os, re, statistics as st, sys
import sacrebleu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
LPS = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "2.0"]
T975_DF4 = 2.776
PUB = {"SL-Transformer": 2.11, "GASLT": 2.72, "SL-Luong": 2.81}
# Recorded beam-5 / LP 0.6 test (PREREG_WAVE4 Stage-2 table) and job-21744 nSA val argmax (addendum text).
REC_B5 = {"nA": [1.873, 1.699, 1.874, 1.612, 1.958], "nSA": [2.103, 2.302, 2.273, 1.991, 2.026]}
REC_NSA_VAL = [2.113, 2.382, 2.872, 2.551, 2.739]
REC_NSA_VAL_LP06 = [1.854, 2.120, 2.516, 2.314, 2.390]


def rows(d, lp):
    rs = [json.loads(l) for l in open(os.path.join(d, f"hyps_LP={lp}.jsonl"))]
    idx = [r["idx"] for r in rs]
    assert sorted(idx) == list(range(len(rs))), f"{d} LP{lp}: idx not a permutation"
    rs.sort(key=lambda r: r["idx"])
    return [r["hyp_beam"] for r in rs], [r["ref"] for r in rs]


def logrows(path):
    out = {}
    for l in open(path):
        m = re.match(r"\s+LP=([0-9.]+)\s+BLEU-4=\s*([0-9.]+)", l)
        if m:
            out[m.group(1)] = float(m.group(2))
    assert set(out) == set(LPS), f"{path}: LP rows {sorted(out)}"
    return out


def s13(h, r):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")


def obj(h, r, tok):
    if tok == "13a":
        return sacrebleu.corpus_bleu([detokenize(x) for x in h], [[detokenize(x) for x in r]], tokenize="13a")
    return sacrebleu.corpus_bleu(h, [r], tokenize="none")


def sweep(d):
    lg = logrows(d + ".log")
    sc = {}
    for lp in LPS:
        h, r = rows(d, lp)
        sc[lp] = s13(h, r)
        assert abs(sc[lp] - lg[lp]) < 0.0006, f"{d} LP{lp}: {sc[lp]:.4f} vs log {lg[lp]}"
    return sc, lg


def ci(v):
    m = st.mean(v); hw = T975_DF4 * st.stdev(v) / len(v) ** 0.5
    return m, m - hw, m + hw


def fmt(v): return " / ".join(f"{x:+.3f}" for x in v)


def main():
    B = os.path.join(HERE)
    res = {}
    # ---- self-checks first (recorded numbers) ----
    for arm in ("nA", "nSA"):
        for i, s in enumerate(SEEDS):
            rec = json.load(open(os.path.join(B, "runs_news", f"{arm}_s{s}", "beam5_test_best.json")))
            assert rec["beam"] == 5 and abs(rec["length_penalty"] - 0.6) < 1e-9
            assert rec["checkpoint"].endswith(f"wave4/runs_news/{arm}_s{s}/best.pt"), rec["checkpoint"]
            b = s13(rec["beam_hyps"], rec["refs"])
            assert abs(b - REC_B5[arm][i]) < 0.0005, f"recorded beam5 {arm} {s}: {b:.4f} vs {REC_B5[arm][i]}"
            assert abs(b - rec["beam_result"]["metrics"]["tokenize=13a;detok=True"]) < 1e-9
    for i, s in enumerate(SEEDS):
        sc, _ = sweep(os.path.join(B, "news_ens", f"s{i}_val"))
        assert abs(max(sc.values()) - REC_NSA_VAL[i]) < 0.0006 and abs(sc["0.6"] - REC_NSA_VAL_LP06[i]) < 0.0006
    print("SELF-CHECK PASSED: 10 recorded beam-5/LP0.6 test values, 5 nSA val argmax values and 5 nSA val LP0.6 "
          "values (job 21744) reproduce through slt/metrics.py")

    # ---- per model ----
    for arm in ("nA", "nSA"):
        for i, s in enumerate(SEEDS):
            vd = os.path.join(B, "news_ens", f"s{i}_val") if arm == "nSA" else os.path.join(B, "news_decode", f"{arm}_s{s}_val")
            td = os.path.join(B, "news_decode", f"{arm}_s{s}_test")
            vsc, vlg = sweep(vd)
            tsc, tlg = sweep(td)
            order = sorted(LPS, key=lambda x: -vsc[x])
            lp = order[0]
            assert max(LPS, key=lambda x: vlg[x]) == lp, "log argmax differs from recomputed argmax"
            margin = vsc[order[0]] - vsc[order[1]]
            h, r = rows(td, lp)
            vh, vr = rows(vd, lp)
            assert len(vr) == 700 and len(r) == 699
            rec = json.load(open(os.path.join(B, "runs_news", f"{arm}_s{s}", "beam5_test_best.json")))
            assert rec["refs"] == r, f"{arm} {s}: refs not identical"
            bh, br = rec["beam_hyps"], rec["refs"]
            a13, b13_ = obj(h, r, "13a"), obj(bh, br, "13a")
            anone, bnone = obj(h, r, "none"), obj(bh, br, "none")
            assert abs(a13.score - s13(h, r)) < 1e-9
            res[(arm, s)] = dict(lp=lp, margin=margin, val=vsc[lp], test=a13.score, base=b13_.score,
                                 test_none=anone.score, base_none=bnone.score,
                                 bp=(b13_.bp, a13.bp), lr=(b13_.sys_len / b13_.ref_len, a13.sys_len / a13.ref_len),
                                 bp_none=(bnone.bp, anone.bp), lr_none=(bnone.sys_len / bnone.ref_len, anone.sys_len / anone.ref_len),
                                 test_argmax=max(LPS, key=lambda x: tsc[x]), test_at_lp_log=tlg[lp])
            x = res[(arm, s)]
            print(f"{arm:3s} s{s:<3d} val-LP {lp} (val {x['val']:.3f}, margin to 2nd {x['margin']:.3f}, LP-edge={lp in ('0.2','2.0')}) "
                  f"| tuned test 13a {x['test']:.3f} (log {x['test_at_lp_log']:.3f}) none {x['test_none']:.3f} "
                  f"| beam5 LP0.6 13a {x['base']:.3f} none {x['base_none']:.3f} | [test-argmax LP {x['test_argmax']}, unused]")

    print("\nPRIMARY Δ_dec (13a) = tuned test - recorded beam-5/LP0.6 test, df 4, t 2.776")
    for arm in ("nSA", "nA"):
        d = [res[(arm, s)]["test"] - res[(arm, s)]["base"] for s in SEEDS]
        m, lo, hi = ci(d)
        print(f"  {arm}: {fmt(d)} | mean {m:+.3f} sd {st.stdev(d):.3f} 95% t [{lo:+.3f}, {hi:+.3f}] | positive {sum(v > 0 for v in d)}/5")
        if arm == "nSA":
            met = sum(v > 0 for v in d) == 5 and lo > 0
            print(f"  RULE (5/5 positive AND lower bound > 0): {'MET' if met else 'NOT MET'}")
        # same with rounded recorded values (the rule text says 'recorded')
        d2 = [round(res[(arm, s)]["test"], 3) - REC_B5[arm][i] for i, s in enumerate(SEEDS)]
        m2, lo2, hi2 = ci(d2)
        print(f"  {arm} vs 3-decimal recorded values: mean {m2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

    print("\nΔ_dec under tokenize='none' (descriptive robustness check, not the criterion)")
    for arm in ("nSA", "nA"):
        d = [res[(arm, s)]["test_none"] - res[(arm, s)]["base_none"] for s in SEEDS]
        m, lo, hi = ci(d)
        print(f"  {arm}: {fmt(d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | positive {sum(v > 0 for v in d)}/5")

    d = [res[("nSA", s)]["test"] - res[("nA", s)]["test"] for s in SEEDS]
    m, lo, hi = ci(d)
    print(f"\nSECONDARY nSA - nA tuned (13a): {fmt(d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | positive {sum(v > 0 for v in d)}/5")
    d = [res[("nSA", s)]["base"] - res[("nA", s)]["base"] for s in SEEDS]
    m, lo, hi = ci(d)
    print(f"  (beam-5/LP0.6 for reference: mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}])")

    for key, lab in (("test", "13a-detok"), ("test_none", "tokenize=none")):
        v = [res[("nSA", s)][key] for s in SEEDS]
        m, lo, hi = ci(v)
        print(f"SECONDARY absolute nSA tuned {lab}: {' / '.join(f'{x:.3f}' for x in v)} | mean {m:.3f} [{lo:.3f}, {hi:.3f}] "
              f"| lower bound above: {[n for n, p in PUB.items() if lo > p] or 'none'} "
              f"| upper bound below: {[n for n, p in PUB.items() if hi < p] or 'none'}")

    print("\nSECONDARY length: BP and sys/ref length ratio, before (beam-5 LP0.6) -> after (tuned)")
    for arm in ("nSA", "nA"):
        for s in SEEDS:
            x = res[(arm, s)]
            print(f"  {arm:3s} s{s:<3d} 13a BP {x['bp'][0]:.3f}->{x['bp'][1]:.3f} ratio {x['lr'][0]:.3f}->{x['lr'][1]:.3f} "
                  f"| none BP {x['bp_none'][0]:.3f}->{x['bp_none'][1]:.3f} ratio {x['lr_none'][0]:.3f}->{x['lr_none'][1]:.3f}")
        for k in ("bp", "lr", "bp_none", "lr_none"):
            print(f"  {arm} mean {k}: {st.mean(res[(arm, s)][k][0] for s in SEEDS):.3f} -> {st.mean(res[(arm, s)][k][1] for s in SEEDS):.3f}")

    r = rows(os.path.join(B, "news_decode", "nSA_s42_test"), "1.0")
    print(f"\nREF TOKENISATION: refs with attached final punctuation {sum(bool(re.search(r'[^ ][.?!,]$', x)) for x in r[1])}/699; "
          f"hyps with spaced final punctuation {sum(bool(re.search(r' [.?!,]$', x)) for x in r[0])}/699")


if __name__ == "__main__":
    main()
