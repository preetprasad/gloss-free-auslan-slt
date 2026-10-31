"""Run-to-run noise bands: recomputed from saved per-run beam-5 test BLEU-4 scores (800 official test clips).
Gates: pooled sd and null-band sd must equal the values the original read-out tools recorded."""
import json, math, re, statistics, sys
from pathlib import Path
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NULL = ROOT / "results/run_to_run_noise/null_metrics.json"
NULL_OUT = ROOT / "results/run_to_run_noise/null_band_output.txt"
ADOPT = ROOT / "results/run_to_run_noise/adopted_values.json"
ADOPT_OUT = ROOT / "results/run_to_run_noise/adopted_crossbatch_output.txt"
K = "tokenize=13a;detok=True"


def sd_ci(sd, df):
    return sd * math.sqrt(df / chi2.ppf(0.975, df)), sd * math.sqrt(df / chi2.ppf(0.025, df))


# ---- six replicates, one seed, one job (job 7952) ----
x = [r["beam5_test"] for r in json.load(open(NULL)).values()]
sd6 = statistics.stdev(x); band6 = 1.96 * sd6 * math.sqrt(2)
rec = re.search(r"beam5_test\s+n=6 mean [\d.]+ sd ([\d.]+)", open(NULL_OUT).read()).group(1)
print(f"GATE six replicates sd {sd6:.3f}, recorded {rec} (read from the saved job output null_band_output.txt)")
if f"{sd6:.3f}" != rec:
    sys.exit("GATE FAILED")
lo, hi = sd_ci(sd6, 5)
print(f"Baseline configuration, one seed, six runs in one job submission: n={len(x)}, test BLEU-4 sd {sd6:.2f} [95% {lo:.2f}, {hi:.2f}]")
print(f"Within-job standard deviation of six identical-seed runs: {sd6:.2f}")
print(f"Paired single-run 95% band within one job submission: +-{band6:.1f} BLEU-4 (1.96 x sd x sqrt 2)  [c02-010 c14-029 c14-073]")

# ---- adopted configuration, five job submissions per seed ----
A = json.load(open(ADOPT))
ss = df = 0
rng = {}
for s, vals in A.items():
    v = list(vals.values()); m = statistics.mean(v)
    ss += sum((t - m) ** 2 for t in v); df += len(v) - 1
    rng[s] = max(v) - min(v)
    print(f"seed {s}: five job submissions {min(v):.3f}-{max(v):.3f}, range {rng[s]:.3f}")
sdp = math.sqrt(ss / df); bandp = 1.96 * sdp * math.sqrt(2)
rec = re.search(r"POOLED.*?\) ([\d.]+)", open(ADOPT_OUT).read()).group(1)
print(f"GATE pooled within-seed sd {sdp:.3f}, recorded {rec} (read from the saved job output adopted_crossbatch_output.txt)")
if f"{sdp:.3f}" != rec:
    sys.exit("GATE FAILED")
print(f"Pooled within-seed standard deviation of the adopted configuration across {len(next(iter(A.values())))} job submissions: {sdp:.2f} (df {df})  [c02-010 c14-073]")
print(f"Paired single-run 95% band across job submissions: +-{bandp:.1f} BLEU-4  [c02-010 c14-029 c14-073]")
print(f"Differences below about 1 BLEU-4 point lie within the measurement variation (bands +-{band6:.1f} within a job, +-{bandp:.1f} across jobs)  [c02-010 c14-029 c14-073]")

# ---- same seed, same configuration, two submissions: 1.1:1 stitch (job 1196567/1199393) vs control of job 1215231 ----
log = open(HERE / "stitch_ratio_job_logs.txt").read().splitlines()
r111 = {}; arm = None; grab = False
for l in log:
    m = re.search(r"=== ARM (sc_r111_s\d+) ", l)
    if m: arm = m.group(1); grab = False
    elif arm and "[beam5]" in l: grab = True
    elif arm and grab and "BLEU-4=" in l:
        r111[arm] = float(l.split("BLEU-4=")[1]); arm = None; grab = False
pc = {s: json.load(open(ROOT / f"results/pose_augmentation_decodes/pa_ctrl_s{s}/beam5_test.json"))["beam_result"]["metrics"][K] for s in (42, 101, 202)}
gaps = {s: abs(r111[f"sc_r111_s{s}"] - pc[s]) for s in pc}
print("same seed, same configuration, different job submission (beam-5 test BLEU-4, read from the saved job output for 1.1:1 runs):")
for s in pc: print(f"  seed {s}: {r111[f'sc_r111_s{s}']:.2f} vs {pc[s]:.2f}, gap {gaps[s]:.2f}")
print(f"Largest gap between two submissions of one configuration at one seed: {max(gaps.values()):.2f}")
six = list(r111.values()) + list(pc.values())
print(f"Six same-configuration runs (three seeds x two submissions): mean {statistics.mean(six):.3f}, standard deviation {statistics.stdev(six):.2f}")

# ---- figure setup ----
n = json.load(open(ROOT / "results/pose_augmentation_decodes/pa_ctrl_s42/beam5_test.json"))["beam_result"]
print(f"Figure setup: {n['n']} official Communication test clips, beam width 5 (saved decode of job 1215231)  [c20-033]")
for name, key in (("second row (control of the later job, wave1 w1A)", "w1A"), ("third row (first job, pa_aug)", "pa_aug")):
    v = [A[s][key] for s in ("42", "101", "202")]
    print(f"{name}: 3 seeds {', '.join(f'{t:.2f}' for t in v)}; three-seed mean {statistics.mean(v):.2f}, range {min(v):.2f}-{max(v):.2f}  [c20-035]")
print(f"Second and third rows are one configuration in 2 job submissions of 3 seeds each; per-seed gaps "
      + ", ".join(f"{abs(A[s]['w1A']-A[s]['pa_aug']):.2f}" for s in ("42", "101", "202")) + "  [c20-036]")

# ---- deterministic repeats in separate submissions ----
P = json.load(open(HERE / "det_pairs.json"))
print("deterministic-mode repeats of one configuration, two submissions (wave 6 and wave 16 controls; read from the saved job outputs):")
d = {s: abs(P[s]["wave6_A"] - P[s]["wave16_A"]) for s in P}
for s in P: print(f"  seed {s}: {P[s]['wave6_A']:.3f} vs {P[s]['wave16_A']:.3f}, gap {d[s]:.3f}")
print(f"Of 3 pairs, {sum(1 for g in d.values() if g == 0)} bit-identical score and {sum(1 for g in d.values() if g > 0)} differ, by {d['101']:.2f} and {d['202']:.2f} BLEU-4")
