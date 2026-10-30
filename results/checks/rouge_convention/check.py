"""ROUGE-L tokenisation convention and ROUGE-L / BLEU-1 ratio (dissertation Results, Discussion,
Conclusion, appendix table 'ROUGE-L of three single models by tokenisation' and the metrics row
of the supplementary results table).  Recomputes everything from the saved beam-5 test outputs."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import metric_suite as M
from slt.metrics import detokenize

SG = ROOT / "results" / "semantic_adequacy_grading"
MODELS = [("Baseline model (DWPose + CTC)", "dwpose_ctc", 9.52),
          ("40-epoch pre-trained model", "vlp", 10.27),
          ("RGB-fusion model", "rgbfusion", 10.33)]
# Published Auslan-Daily Communication, pose input (benchmark paper, Table 3): ROUGE-L, BLEU-1.
# Transcribed inputs for the ratio, not results of this repository.
PUBLISHED = [("SL-Luong", 37.27, 30.15), ("SL-Transformer", 35.65, 31.31), ("GASLT", 35.74, 28.19)]
CLAIM = {  # dissertation Table (appendix) values: detokenised b1.0 / tokenised b1.0 / tokenised b1.2
    "dwpose_ctc": (16.56, 36.22, 35.70), "vlp": (17.74, 36.68, 36.27), "rgbfusion": (17.84, 37.67, 37.08)}
CLAIM_RATIO = {"SL-Luong": 1.236, "SL-Transformer": 1.139, "GASLT": 1.268}
bad = []
def chk(name, got, want, nd=2):
    ok = round(got, nd) == round(want, nd)
    if not ok: bad.append(name)
    return "ok" if ok else "MISMATCH (dissertation %s)" % want

def rouge(h, r, beta, detok):
    M.ROUGE_BETA = beta
    f = (lambda s: detokenize(s)) if detok else (lambda s: s)
    return 100.0 * sum(M.rouge_l(f(a), f(b)) for a, b in zip(h, r)) / len(h)

print("ROUGE-L of three single models, official Communication test set")
res = {}
for label, key, rec_b4 in MODELS:
    rows = [json.loads(l) for l in open(SG / ("hyps_%s.jsonl" % key))]
    h = [r["hyp_beam"] for r in rows]; r_ = [r["ref"] for r in rows]
    b4, b1, _, _ = M.corpus(h, [detokenize(x) for x in r_]) if False else M.corpus(
        [detokenize(x) for x in h], [detokenize(x) for x in r_], h, r_)
    # gate: the BLEU-4 recorded by the original job (beam 5) must be reproduced first
    if round(b4, 2) != rec_b4:
        sys.exit("GATE FAILED: %s BLEU-4 %.3f != recorded %.2f" % (key, b4, rec_b4))
    d10, t10, t12 = rouge(h, r_, 1.0, True), rouge(h, r_, 1.0, False), rouge(h, r_, 1.2, False)
    res[key] = (label, len(rows), b4, b1, d10, t10, t12)
print("gate: BLEU-4 of the saved outputs reproduces the recorded 9.52 / 10.27 / 10.33")
print()
print("%-30s %5s %12s %12s %12s" % ("model (n clips)", "", "detok b=1.0", "token b=1.0", "token b=1.2"))
for key, (label, n, b4, b1, d10, t10, t12) in res.items():
    c = CLAIM[key]
    print("%-30s n=%d  %6.2f  %6.2f  %6.2f   [%s | %s | %s]" % (label, n, d10, t10, t12,
          chk(key + " detok", d10, c[0]), chk(key + " tok1.0", t10, c[1]), chk(key + " tok1.2", t12, c[2])))
print()
print("tokenisation swing at beta = 1.0 (whitespace-tokenised minus detokenised), ROUGE-L points")
for key, (label, n, b4, b1, d10, t10, t12) in res.items():
    print("  %-30s %+.2f" % (label, t10 - d10))
sw = {k: v[5] - v[4] for k, v in res.items()}
print("more than 19 points for the baseline and the RGB-fusion outputs: %s" % (sw["dwpose_ctc"] > 19 and sw["rgbfusion"] > 19))
if not (sw["dwpose_ctc"] > 19 and sw["rgbfusion"] > 19): bad.append("19 points")
print("(the 40-epoch pre-trained outputs move %.2f points)" % sw["vlp"])

print()
print("convention: ROUGE-L = mean over sentences of the F_beta of the longest common subsequence, x100")
print("  slt/metric_suite default beta = %.1f; tokens = whitespace split, punctuation kept" % 1.2)
print("  n clips per model: %s; one saved run per model (one file each)" % sorted({v[1] for v in res.values()}))
src = (ROOT / "results" / "semantic_adequacy_grading" / "dump_hyps.py").read_text()
print("  beam width of the saved outputs: %s (default of the script that wrote them)" %
      re.search(r'"--beam", type=int, default=(\d+)', src).group(1))

print()
print("ROUGE-L (beta 1.2) divided by BLEU-1")
print("  published systems (inputs transcribed from the benchmark paper's Table 3):")
for n, rl, b1 in PUBLISHED:
    q = rl / b1
    print("    %-15s ROUGE-L %.2f / BLEU-1 %.2f = %.3f  [%s]" % (n, rl, b1, q, chk(n, q, CLAIM_RATIO[n], 3)))
print("  single models of this thesis:")
qs = []
for key, (label, n, b4, b1, d10, t10, t12) in res.items():
    qs.append(t12 / b1)
    print("    %-30s ROUGE-L %.2f / BLEU-1 %.2f = %.3f" % (label, t12, b1, t12 / b1))
print("  range for the three single models: %.3f to %.3f  [%s]" % (min(qs), max(qs),
      chk("range", min(qs), 1.119, 3) if chk("rangehi", max(qs), 1.151, 3) == "ok" else "MISMATCH"))
if round(min(qs), 3) != 1.119 or round(max(qs), 3) != 1.151: bad.append("ratio range")
if bad: sys.exit("DISAGREES: %s" % bad)
