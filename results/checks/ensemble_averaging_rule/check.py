"""Wave 12d: averaging rule of the final ensemble (power mean of the members' probabilities vs the geometric mean).

Recomputes from the saved n-best lists of the five rules (validation, 800 clips, beam 10, six members; Kaya jobs
138382 decode, 138383 read-out): candidate choice at each length penalty, BLEU-4 (sacreBLEU 13a on detokenised text),
the paired bootstrap (98.75% intervals, 2,000 resamples, LP re-chosen inside each resample) and the brevity penalty.
Gates: the control reproduces the reported ensemble (12.470 / WER 0.9164, 800/800 hypotheses of job 120153) and the
recomputed bootstrap equals the one the original job saved.
Run:  cd src && python3 ../results/checks/ensemble_averaging_rule/check.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "tools")]
import w12d_powermean as W                                   # the original read-out's own scoring/bootstrap code
from slt.metrics import corpus_bleu4, corpus_wer, detokenize

POWERS, LPS = W.POWERS, W.LPS
tag = lambda p: f"{p:g}"
refs = json.load(open(HERE / "refs.json"))["refs"]
n = len(refs)
meta = json.load(open(HERE / "meta.json"))
nb = {p: sum((json.load(open(HERE / f"nbest_p{tag(p)}_part{k}.json"))["nbest"] for k in (0, 1)), []) for p in POWERS}
assert all(len(v) == n == 800 for v in nb.values())
hyps = {p: {lp: [W.pick(c, lp) for c in nb[p]] for lp in LPS} for p in POWERS}
R = [detokenize(r) for r in refs]

# ---- gates ----
h03 = [detokenize(h) for h in hyps[0.0][0.3]]
bleu, wer = corpus_bleu4(h03, R, "13a"), corpus_wer(h03, R)
shards = [json.load(open(ROOT / "results" / "ensemble_member_subsets_ablation" / f"shard{i}of4.json")) for i in range(4)]
prior = [s for s in shards if "63" in s["hyps"]]
assert len(prior) == 1 and prior[0]["refs"] == refs and prior[0]["ckpt_md5"] == meta["ckpt_md5"]
same = sum(x == y for x, y in zip(prior[0]["hyps"]["63"], hyps[0.0][0.3]))
ok = abs(bleu - 12.470) < 5e-4 and abs(wer - 0.9164) < 5e-5 and same == n
print(f"GATE {'PASSED' if ok else 'FAILED'}: control (geometric mean) at LP 0.3: BLEU-4 {bleu:.2f} ({bleu:.3f}), WER {wer:.4f}; "
      f"hypotheses identical to job 120153: {same}/{n}")
if not ok: sys.exit(1)

# ---- statistics ----
ST = np.array([[[W.sent_stats(detokenize(h), r) for h, r in zip(hyps[p][lp], R)] for lp in LPS] for p in POWERS], float)
full, crit, _ = W.criteria(ST, 2000, 1000)                  # seed 0, as in the original read-out
saved = json.load(open(HERE / "job_summary_w12d_scores.json"))
for (p, c) in zip(POWERS[1:], crit):
    s = saved["criteria"][tag(p)]
    assert abs(c["delta"] - s["delta"]) < 1e-9 and np.allclose(c["bonf"], s["bonf"], atol=1e-9) \
        and abs(c["half_agree"] - s["half_agree"]) < 1e-12, f"bootstrap for p = {p} differs from the saved job output"
print("GATE PASSED: recomputed point differences and 98.75% bootstrap intervals equal those saved by job 138383")

two = lambda x: f"{(0.0 if abs(x) < 0.005 else x):+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
name = lambda p: {0.0: "geometric mean (exponent 0)"}.get(p, f"power mean, exponent {tag(p)}")
best = full.argmax(1)
print(f"\nRules compared: geometric mean and power means of the members' probabilities with exponents "
      f"{', '.join(tag(p) for p in POWERS[1:])} ({n} validation clips, six members, beam 10; no test clip)")
print(f"Interval level: {100 * (1 - W.ALPHA):g}% per comparison (Bonferroni over {len(POWERS) - 1} comparisons), 2000 paired resamples, "
      f"length penalty re-chosen inside each resample")
print("\nEach rule at its validation-chosen length penalty:")
print(f"{'rule':>30s} {'LP':>4s} {'BLEU-4':>7s} {'BP':>6s} {'n-gram prec.':>12s}")
BP, PR = {}, {}
for i, p in enumerate(POWERS):
    S = ST[i, best[i]].sum(0)
    BP[p] = float(np.exp(min(0.0, 1 - S[1] / S[0])))
    PR[p] = float(100 * np.exp(np.log(S[2:6] / S[6:10]).mean()))
    print(f"{name(p):>30s} {LPS[best[i]]:4.1f} {full[i, best[i]]:7.2f} {BP[p]:6.3f} {PR[p]:12.2f}")

print("\nBLEU-4 difference to the geometric mean, each rule at its own length penalty:")
for p, c in zip(POWERS[1:], crit):
    print(f"  exponent {tag(p):>4s}: {two(c['delta'])} ({two(c['bonf'][0])} to {two(c['bonf'][1])})   "
          f"[{c['delta']:+.3f}; 98.75% interval [{c['bonf'][0]:+.3f}, {c['bonf'][1]:+.3f}]]   "
          f"{'measurably different' if c['different'] else 'not distinguishable from the geometric mean'}")
print("\nBrevity penalty (dissertation rounding): geometric mean " + f"{BP[0.0]:.2f}" + ", exponent 1 " + f"{BP[1.0]:.2f}" +
      ", exponent 2 " + f"{BP[2.0]:.2f}" + "; n-gram precision falls from " + f"{PR[0.0]:.2f} to {PR[1.0]:.2f} (exponent 1) and {PR[2.0]:.2f} (exponent 2)")
nd = sum(c["different"] for c in crit)
print(f"\nRules measurably different from the geometric mean: {nd} of {len(crit)}")
assert nd == 0
