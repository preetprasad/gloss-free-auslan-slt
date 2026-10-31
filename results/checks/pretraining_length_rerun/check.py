"""Pre-training length (wave 11 re-run) and News frame limit, selected on validation alone.

Recomputes means, the pre-registered selection rule and paired t-intervals from the stored beam-5 BLEU-4 of each run
(cluster jobs 116619 and 116620; scores read from the saved job outputs, see SOURCES.md).
"""
import json, math, re, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
runs = json.load(open(HERE / "runs.json"))
T95 = {2: 4.303}
SEEDS = (42, 101, 202)
PRE = ["P0", "P40", "P80", "P128", "P256"]


def mean(arm, key): return st.mean(runs[f"{arm}_s{s}"][key] for s in SEEDS)


def interval(a, b, key):
    d = [runs[f"{a}_s{s}"][key] - runs[f"{b}_s{s}"][key] for s in SEEDS]
    m = st.mean(d); h = T95[2] * st.stdev(d) / math.sqrt(3)
    return m, m - h, m + h


# GATE: the job recorded these P0 / P40 three-seed means (beam-5 validation 10.450 / 10.192, test 10.122 / 9.232)
gate = {("P0", "beam5_val"): 10.450, ("P40", "beam5_val"): 10.192, ("P0", "beam5_test"): 10.122, ("P40", "beam5_test"): 9.232}
for (a, k), v in gate.items():
    g = mean(a, k)
    print(f"GATE {a} mean {k} {g:.3f} (recorded {v:.3f})")
    if abs(g - v) > 0.0006: sys.exit("GATE FAILED")
assert all(r["done"] and r["n_val"] == 800 and r["n_test"] == 800 and r["beam"] == 5 for r in runs.values() if r["kind"] == "pre")

# ---- settings, read from the recorded job scripts
sb = (ROOT / "jobs" / "slurm" / "w11_vlpepochs.sbatch").read_text()
E = [int(x) for x in re.search(r"^E=\(([^)]*)\)", sb, re.M).group(1).split()]
S = [int(x) for x in re.search(r"^E=\(.*\); S=\(([^)]*)\)", sb, re.M).group(1).split()]
print(f"pre-training lengths in the job script w11_vlpepochs.sbatch: {E[1:]} epochs (0 = no pre-training); seeds {S}  [c07-020 c07-021]")
pe = {a: sorted({runs[f'{a}_s{s}']['pretrain_epochs'] for s in SEEDS}) for a in PRE[1:]}
print("pre-training epochs completed per arm (saved progress files): " + ", ".join(f"{a}: {v[0]}" for a, v in pe.items()) + "  [c07-020]")
nsd = {a: sorted(r["seed"] for r in runs.values() if r["arm"] == a) for a in PRE}
print(f"seeds per setting: {len(SEEDS)} (" + "; ".join(f"{a}: {len(v)}" for a, v in nsd.items()) + f") = {len(SEEDS)} seeds  [c07-021 c09-029 c09-034]")
assert all(v == sorted(S) for v in nsd.values())

print("\nPRE-TRAINING RE-RUN (beam-5 BLEU-4, mean of three seeds)")
for a in PRE:
    print(f"  {a:5s} validation {mean(a,'beam5_val'):.3f}  test {mean(a,'beam5_test'):.3f}")
means = {a: mean(a, "beam5_val") for a in PRE[1:]}
best = max(means.values())
sel = next(a for a in PRE[1:] if means[a] > best - 0.10)
print(f"pre-registered rule (tie 0.10, as in the recorded read-out command): cheapest length within 0.10 of the best mean validation {best:.3f}")
print(f"SELECTED ON VALIDATION: {sel} = {sel[1:]} pre-training epochs")
for a in ("P40", "P80", "P128", "P256"):
    print(f"  {a} - P0 validation {interval(a,'P0','beam5_val')[0]:+.3f} [{interval(a,'P0','beam5_val')[1]:+.2f}, {interval(a,'P0','beam5_val')[2]:+.2f}]"
          f"; test {interval(a,'P0','beam5_test')[0]:+.3f} [{interval(a,'P0','beam5_test')[1]:+.2f}, {interval(a,'P0','beam5_test')[2]:+.2f}]")
topv = max(PRE[1:], key=lambda a: mean(a, "beam5_val")); topt = max(PRE[1:], key=lambda a: mean(a, "beam5_test"))
print(f"among the pre-trained lengths the highest mean is {topv} on validation and {topt} on test")

print("\nNEWS FRAME LIMIT (150 epochs, News real clips only, three seeds)")
for a, f in (("nF256", 256), ("nF512", 512)):
    print(f"  {f} frames: mean beam-5 validation BLEU-4 {mean(a,'beam5_val'):.2f} ({mean(a,'beam5_val'):.3f}); test {mean(a,'beam5_test'):.3f}  [c11-032]")
m, lo, hi = interval("nF512", "nF256", "beam5_val")
print(f"  512 - 256 validation {m:+.2f} ({m:+.3f}); 95% t-interval over the 3 seeds {lo:+.2f} to {hi:+.2f}  [c11-033]")
nm = {256: mean("nF256", "beam5_val"), 512: mean("nF512", "beam5_val")}
nsel = 256 if nm[256] > nm[512] - 0.05 else 512
print(f"  rule (cheapest within 0.05 of the best mean validation): {nsel} frames selected  [c11-034 c06-026]")
nv = (ROOT / "jobs" / "slurm" / "w11_newsframes.sbatch").read_text()
fr = re.search(r"^F=\(([^)]*)\)", nv, re.M).group(1).split()
nseeds = re.search(r"^F=\(.*; S=\(([^)]*)\)", nv, re.M).group(1).split()
ep = re.search(r'"\$s" (\d+) -- --max_src_frames', nv).group(1)
print(f"  job script w11_newsframes.sbatch (the wave-11 re-run, not the earlier runs): frame limits {fr}, {ep} epochs, seeds {nseeds}")
# the earlier News-only runs: frame limits read from the recorded job scripts
K = ROOT / "jobs" / "slurm"
old = {}
for name, pat in (("w4_news.sbatch", r"w4run nA .*--max_src_frames (\d+)"), ("news_bpe.sbatch", r"--max_src_frames (\d+)"),
                  ("w12a_news_decomp.sbatch", r"-- --max_src_frames (\d+)")):
    m = re.search(pat, (K / name).read_text())
    old[name] = int(m.group(1))
print("  frame limit in the earlier News-only job scripts: " + ", ".join(f"{k} {v}" for k, v in old.items()) + f"; selected on validation in wave 11: {nsel}  [c06-026]")
assert set(old.values()) == {512} and nsel == 256
# the earlier single-run comparison (appendix row, job 1196181): arms read from news_long.sbatch
nl = (K / "news_long.sbatch").read_text()
arms = re.findall(r"^run (news_long_(\d+)) (\d+)$", nl, re.M)
nlep = re.search(r"--epochs (\d+)", nl).group(1)
assert all(a[1] == a[2] for a in arms)
lims = [int(a[2]) for a in arms]
print(f"  news_long.sbatch (earlier single-run job, {nlep} epochs, outputs not saved locally) compares: {min(lims)}-frame against {max(lims)}-frame  [c18-069]")
