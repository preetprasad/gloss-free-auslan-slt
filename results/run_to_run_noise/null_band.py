"""W5-NULL (job 7952): run-level noise of the historical (non-deterministic) mode, six replicates at seed 42.
Pre-registered (PREREG_WAVE4 §4): sd_hat with its chi-square 95% CI, and the paired single-run band 1.96*sd*sqrt(2).
Never an empirical percentile of n <= 6."""
import json, math, statistics, sys
from scipy.stats import chi2
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "null_metrics.json"))
for key in ("beam5_test", "beam5_val", "val_best"):
    x = [r[key] for r in rows.values()]; n = len(x); df = n - 1
    sd = statistics.stdev(x)
    lo, hi = sd * math.sqrt(df / chi2.ppf(0.975, df)), sd * math.sqrt(df / chi2.ppf(0.025, df))
    band = 1.96 * sd * math.sqrt(2)
    print(f"{key:10s} n={n} mean {statistics.mean(x):.3f} sd {sd:.3f} [95% {lo:.3f}, {hi:.3f}] | paired single-run band "
          f"+-{band:.2f} [{1.96*lo*math.sqrt(2):.2f}, {1.96*hi*math.sqrt(2):.2f}] | range {min(x):.3f}-{max(x):.3f}")
