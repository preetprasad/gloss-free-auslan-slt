"""Wave 4 (job 7949): stitched data without the boundary trim (T) and two realisations of half as many
sentences (Q) against the adopted corpus (A). Recomputed from the five per-seed summary files."""
import json, re, sys, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}   # as in tools/seed_decision.py

def r2(x):  # round the 3-decimal value half away from zero, as the dissertation does
    from decimal import Decimal, ROUND_HALF_UP
    return f"{Decimal(f'{x:.3f}').quantize(Decimal('0.01'), ROUND_HALF_UP):+}"

def interval(d):
    m = statistics.mean(d); hw = T975[len(d) - 1] * statistics.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw

rows = {int(f.stem.split("_s")[-1]): json.load(open(f)) for f in HERE.glob("summary_*.json")}
seeds = sorted(rows)
print("seeds with summaries:", seeds, "(read from the saved job output summary_*.json, job 7949)")
def delta(arm, key, ss):
    return [rows[s][arm][key] - rows[s]["w4A"][key] for s in ss]
s3 = [s for s in seeds if "w4T" in rows[s]]
vT = interval(delta("w4T", "beam5_val", s3))
print(f"GATE T-A beam-5 validation mean {vT[0]:+.3f} (recorded +0.256 in the pre-registration outcome)")
if round(vT[0], 3) != 0.256:
    sys.exit("GATE FAILED")

tT = delta("w4T", "beam5_test", s3); m, lo, hi = interval(tT)
print(f"\nT (no trim) - A, beam-5 test, seeds {s3}: per seed " + " ".join(f"{x:+.2f}" for x in tT))
print(f"T - A test mean {m:+.2f} (95% t {r2(lo)} to {r2(hi)}; unrounded {lo:+.3f} to {hi:+.3f}) [c17-051]")
print(f"T - A test lower in {sum(x < 0 for x in tT)} of {len(tT)} seeds (all three seeds lower: {all(x < 0 for x in tT)}) [c17-051]")
print(f"number of seeds {len(tT)} [c17-050]")

sq = [s for s in seeds if "w4Q" in rows[s]]
tQ = delta("w4Q", "beam5_test", sq); m, lo, hi = interval(tQ)
print(f"\nQ (two realisations, half the sentences) - A, beam-5 test, seeds {sq}: per seed " + " ".join(f"{x:+.2f}" for x in tQ))
print(f"Q - A test mean {m:+.2f} (95% t {r2(lo)} to {r2(hi)}; unrounded {lo:+.3f} to {hi:+.3f}) [c17-059]")
print(f"number of seeds {len(sq)} [c17-060]")

# settings, read from the build code and the job script
src = (ROOT / "src/tools/build_posestitch_corpus_v2.py").read_text()
dflt = re.search(r'"--trim_frac",\s*type=float,\s*default=([0-9.]+)', src).group(1)
print(f"\nbuilder default --trim_frac {float(dflt)*100:.0f}% of each isolated clip's frames at each end (arm A uses the default) [c07-035 c15-016 c17-050]")
sb = (ROOT / "jobs/slurm/w4_prep.sbatch").read_text()
print("arm T passes --trim_frac 0.0:", "--trim_frac 0.0" in sb, "[c17-050]")
print("arm Q passes --realizations 2 (A: 1) with the first 6,880 of A's sentences:", "--realizations 2" in sb and "6,880" in sb, "[c17-060]")
