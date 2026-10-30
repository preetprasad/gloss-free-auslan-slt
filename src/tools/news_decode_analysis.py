#!/usr/bin/env python3
"""Readout for the News decoding test (results/PREREG_WAVE4_2026-09-22.md, addendum 24 Sep 16:10; job 27382).
Written before any tuned-decode test output existed.

For each News best.pt (nA / nSA x seeds 42/101/202/303/404): LP = argmax of the model's own beam-10 VALIDATION sweep
(nSA: <news_ens>/s{i}_val, i = seed index; nA: <news_decode>/nA_s{seed}_val); test read at that LP from
<news_decode>/<arm>_s{seed}_test. Gate: every read LP row re-scores to its log. Compares with the recorded beam-5 /
LP 0.6 test (<runs_news>/<arm>_s<seed>/beam5_test_best.json). Scores 13a-detok and tokenize='none'.
"""
import argparse, json, os, re, statistics as st, sys
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.metrics import corpus_bleu4, detokenize   # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
LPS = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "2.0"]
T975 = {4: 2.776}
PUB = {"SL-Transformer": 2.11, "GASLT": 2.72, "SL-Luong": 2.81}


def load(d, lp):
    rows = sorted((json.loads(l) for l in open(os.path.join(d, f"hyps_LP={lp}.jsonl"))), key=lambda r: r["idx"])
    return [r["hyp_beam"] for r in rows], [r["ref"] for r in rows]


def logged(path):
    out = {}
    for l in open(path):
        m = re.match(r"\s+LP=([0-9.]+)\s+BLEU-4=\s*([0-9.]+)", l)
        if m: out[m.group(1)] = float(m.group(2))
    return out


def b13(h, r): return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")
def bnone(h, r): return sacrebleu.corpus_bleu(h, [r], tokenize="none").score
def lenratio(h, r): return sum(len(x.split()) for x in h) / sum(len(x.split()) for x in r)


def pick(d):
    lg = logged(d + ".log"); sc = {}
    for lp in LPS:
        h, r = load(d, lp); b = b13(h, r)
        assert abs(b - lg[lp]) < 0.0015, f"gate: {d} LP {lp} {b:.3f} vs log {lg[lp]}"
        sc[lp] = b
    return max(LPS, key=lambda x: (sc[x], -LPS.index(x))), sc


def interval(d):
    m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news_ens", required=True); ap.add_argument("--news_decode", required=True)
    ap.add_argument("--runs_news", required=True)
    a = ap.parse_args()
    res = {}
    for arm in ("nA", "nSA"):
        for i, s in enumerate(SEEDS):
            vdir = os.path.join(a.news_ens, f"s{i}_val") if arm == "nSA" else os.path.join(a.news_decode, f"{arm}_s{s}_val")
            lp, vsc = pick(vdir)
            tdir = os.path.join(a.news_decode, f"{arm}_s{s}_test")
            tlp, tsc = pick(tdir)          # gate only; the test argmax is NOT used
            h, r = load(tdir, lp)
            rec = json.load(open(os.path.join(a.runs_news, f"{arm}_s{s}", "beam5_test_best.json")))
            rh, rr = rec["beam_hyps"], rec["refs"]
            assert [" ".join(x.split()) for x in rr] == [" ".join(x.split()) for x in r], f"{arm} {s}: refs differ"
            res[(arm, s)] = dict(lp=lp, val=vsc[lp], test=b13(h, r), test_none=bnone(h, r), lr=lenratio(h, r),
                                 base=b13(rh, rr), base_none=bnone(rh, rr), base_lr=lenratio(rh, rr))
            x = res[(arm, s)]
            print(f"{arm:3s} s{s}: val-LP {lp} (val {x['val']:.3f}) | test tuned {x['test']:.3f} / none {x['test_none']:.3f} "
                  f"len {x['lr']:.2f} | recorded beam-5 LP0.6 {x['base']:.3f} / none {x['base_none']:.3f} len {x['base_lr']:.2f}")
    for arm in ("nA", "nSA"):
        d = [res[(arm, s)]["test"] - res[(arm, s)]["base"] for s in SEEDS]; m, lo, hi = interval(d)
        print(f"\nΔ_dec {arm}: {' '.join(f'{v:+.3f}' for v in d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | +{sum(v > 0 for v in d)}/5")
        if arm == "nSA":
            ok = sum(v > 0 for v in d) == 5 and lo > 0
            print("PRIMARY READING:", "decoding under-generates on News; validation-tuned LP fixes part of it" if ok
                  else "bounded estimate (rule not met)")
    d = [res[("nSA", s)]["test"] - res[("nA", s)]["test"] for s in SEEDS]; m, lo, hi = interval(d)
    print(f"[secondary] nSA - nA under tuned decode: {' '.join(f'{v:+.3f}' for v in d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for key, lab in (("test", "13a-detok"), ("test_none", "tokenize=none")):
        v = [res[("nSA", s)][key] for s in SEEDS]; m, lo, hi = interval(v)
        above = [n for n, p in PUB.items() if lo > p]
        print(f"[secondary] nSA tuned absolute ({lab}): {' '.join(f'{x:.3f}' for x in v)} | mean {m:.3f} [{lo:.3f}, {hi:.3f}] "
              f"| lower bound above: {above or 'none'}")


if __name__ == "__main__":
    main()
