#!/usr/bin/env python3
"""Secondary readout the addendum mandates but tools/news_decode_analysis.py does not print: brevity penalty
(and sacreBLEU 13a sys/ref length ratio) before (recorded beam-5 LP0.6) and after (beam 10, validation-argmax LP).
Reuses the pre-written tool's load/pick/b13, so the LP selection and gate are identical. Self-check: every BLEU
recomputed here must equal the tool's b13 (and therefore slt.metrics.corpus_bleu4) and the recorded Stage-2 values.
Run from src/."""
import json, os, statistics as st, sys
import sacrebleu
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import news_decode_analysis as T
from slt.metrics import detokenize

R = "../results/news_decoding_length_check"
REC = {("nA", 42): 1.873, ("nA", 101): 1.699, ("nA", 202): 1.874, ("nA", 303): 1.612, ("nA", 404): 1.958,
       ("nSA", 42): 2.103, ("nSA", 101): 2.302, ("nSA", 202): 2.273, ("nSA", 303): 1.991, ("nSA", 404): 2.026}


def stats(h, r):
    b = sacrebleu.corpus_bleu([detokenize(x) for x in h], [[detokenize(x) for x in r]], tokenize="13a")
    assert abs(b.score - T.b13(h, r)) < 1e-9
    return b.score, b.bp, b.sys_len / b.ref_len


for arm in ("nA", "nSA"):
    before, after = [], []
    for i, s in enumerate(T.SEEDS):
        vdir = f"{R}/news_ens/s{i}_val" if arm == "nSA" else f"{R}/news_decode/{arm}_s{s}_val"
        lp, _ = T.pick(vdir)
        h, r = T.load(f"{R}/news_decode/{arm}_s{s}_test", lp)
        rec = json.load(open(f"{R}/runs_news/{arm}_s{s}/beam5_test_best.json"))
        b0 = stats(rec["beam_hyps"], rec["refs"]); b1 = stats(h, r)
        assert abs(b0[0] - REC[(arm, s)]) < 0.0015, (arm, s, b0[0])   # reproduces the recorded Stage-2 test
        before.append(b0); after.append(b1)
        print(f"{arm:3s} s{s}: LP {lp} | before BLEU {b0[0]:.3f} BP {b0[1]:.3f} len {b0[2]:.3f} | "
              f"after BLEU {b1[0]:.3f} BP {b1[1]:.3f} len {b1[2]:.3f}")
    for lab, xs in (("before", before), ("after", after)):
        print(f"{arm} {lab}: mean BP {st.mean(x[1] for x in xs):.3f}, mean 13a len ratio {st.mean(x[2] for x in xs):.3f}, "
              f"BP<1 in {sum(x[1] < 1 for x in xs)}/5")
print("self-check passed: all 10 recorded beam-5 test values reproduce; BLEU identical to the tool's b13")
