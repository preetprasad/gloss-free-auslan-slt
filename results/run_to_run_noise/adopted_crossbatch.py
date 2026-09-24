"""Cross-batch noise of the ADOPTED recipe: five identical-config runs per seed (verified by adopted_identity.py),
each from a different job. Pooled within-seed sd (df = sum of n_s - 1) with a chi-square 95% CI."""
import glob, json, math, os, statistics, sys
G = "/group/pmc099/pprasad"; K = "tokenize=13a;detok=True"
RUNS = {"pa_aug": "pose_aug/runs/pa_aug_s{s}", "w1A": "wave1_r2/runs/w1A_s{s}", "w3C": "wave3_track/runs/w3C_s{s}",
        "r3_111": "wave3_ratioaug/runs/r3_111_s{s}", "a3ctl": "wave3_augdose/runs/a3ctl_s{s}"}
def test_b5(d):
    for f in ("beam5_test_best.json", "beam5_test.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            j = json.load(open(p)); return j["beam_result"]["metrics"][K] if "beam_result" in j else j["metrics"][K]
    return None
if len(sys.argv) < 2:          # on Kaya: extract only (no scipy there)
    json.dump({s: {n: test_b5(f"{G}/{p.format(s=s)}") for n, p in RUNS.items()} for s in (42, 101, 202)},
              open("adopted_values.json", "w"), indent=1); sys.exit(0)
from scipy.stats import chi2
ALL = json.load(open(sys.argv[1]))
ss, df = 0.0, 0
for s in ("42", "101", "202"):
    vals = ALL[s]
    x = [v for v in vals.values() if v is not None]
    ss += sum((v - statistics.mean(x)) ** 2 for v in x); df += len(x) - 1
    print(f"s{s}: " + " ".join(f"{n} {v:.3f}" for n, v in vals.items() if v is not None)
          + f" | mean {statistics.mean(x):.3f} sd {statistics.stdev(x):.3f} range {max(x)-min(x):.3f}")
sd = math.sqrt(ss / df)
lo, hi = sd * math.sqrt(df / chi2.ppf(0.975, df)), sd * math.sqrt(df / chi2.ppf(0.025, df))
print(f"POOLED within-seed cross-batch sd (beam-5 test) {sd:.3f} [95% {lo:.3f}, {hi:.3f}], df {df} | "
      f"paired single-run band +-{1.96*sd*math.sqrt(2):.2f}")
