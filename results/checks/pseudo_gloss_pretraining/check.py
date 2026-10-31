"""Wave 23: CTC-only pseudo-gloss first stage (X) against a compute-matched full-loss first stage (Y) and no first stage (C).

Recomputes beam-5 BLEU-4 of best.pt for C, X, Y x seeds 42/101/202 on validation and official test from the saved
hypotheses (800 clips each), then the paired differences, their 95% t intervals (df = 2) and sign counts.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

SEEDS = (42, 101, 202)
T975_DF2 = 4.302652729911275   # two-sided 95% Student t, 2 degrees of freedom

bleu, cache = {}, {}
for a in "CXY":
    for s in SEEDS:
        for sp in ("val", "test"):
            rows = [json.loads(l) for l in open(HERE / "runs" / f"{a}_s{s}_{sp}.jsonl")]
            assert len(rows) == 800
            cache[a, s, sp] = rows
            bleu[a, s, sp] = corpus_bleu4([detokenize(r["hyp"]) for r in rows], [detokenize(r["ref"]) for r in rows])

# references must be the official ones (when `make data` has written the manifests)
for sp in ("val", "test"):
    man = ROOT / "comm_run" / "manifests" / f"{sp}.tsv"
    if man.exists():
        refs = [l.rstrip("\n").split("\t")[1].strip() for l in open(man) if "\t" in l]
        assert refs == [r["ref"] for r in cache["C", 42, sp]], f"{sp} references differ from the manifest"
        print(f"{sp} references match comm_run/manifests/{sp}.tsv: 800/800")

# GATE 1: the recomputed scores equal what the job itself recorded (summary_w23.json)
rec = json.load(open(HERE / "summary_w23.json"))
for a in "CXY":
    for s in SEEDS:
        for sp, k in (("val", "beam5_val"), ("test", "beam5_test")):
            if abs(bleu[a, s, sp] - rec[f"{a}_s{s}"][k]) > 1e-6:
                sys.exit(f"GATE FAILED {a} s{s} {sp}: {bleu[a, s, sp]} vs recorded {rec[a + '_s' + str(s)][k]}")
print("GATE: all 18 recomputed beam-5 BLEU-4 scores (validation and test) equal the job's recorded values to 1e-6")
# earlier jobs with the same control recipe (local per-item outputs): wave 16 control A and wave 6 control A (official test, beam 5)
earlier = {}
rec_e = json.load(open(HERE / "earlier_recorded.json"))
for tag in ("wave16A", "wave6A"):
    for s in SEEDS:
        rows = [json.loads(l) for l in open(HERE / "runs" / f"earlier_{tag}_s{s}_test.jsonl")]
        assert len(rows) == 800 and [r["ref"] for r in rows] == [r["ref"] for r in cache["C", s, "test"]]
        earlier[tag, s] = corpus_bleu4([detokenize(r["hyp"]) for r in rows], [detokenize(r["ref"]) for r in rows])
        if abs(earlier[tag, s] - rec_e[f"{tag}_s{s}"]) > 1e-6:
            sys.exit(f"GATE FAILED {tag} s{s}: {earlier[tag, s]} vs recorded {rec_e[tag + '_s' + str(s)]}")
print("GATE: the 6 recomputed beam-5 test scores of the earlier jobs (wave 16 and wave 6 controls) equal those jobs' recorded values to 1e-6")


def sg(x, d=2):
    return f"{x:+.{d}f}"


def contrast(p, q, sp):
    d = np.array([bleu[p, s, sp] - bleu[q, s, sp] for s in SEEDS])
    h = T975_DF2 * d.std(ddof=1) / np.sqrt(3)
    return d, d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum())


print("\nSetting: beam 5, best.pt by validation, 800 clips per split, seeds 42/101/202; "
      "X = CTC-only first stage then 40-epoch training, Y = 40 further full-loss epochs then 40-epoch training, C = neither")
print("\nBeam-5 BLEU-4 per seed (42 / 101 / 202) and mean")
for a in "CXY":
    for sp in ("val", "test"):
        v = [bleu[a, s, sp] for s in SEEDS]
        print(f"  {a} {sp}: " + " / ".join(f"{x:.3f}" for x in v) + f"   mean {np.mean(v):.3f}")

print("\nPaired differences, 95% t interval over three seeds (df = 2)")
tag = {("X", "Y", "val"): "c17-102", ("X", "Y", "test"): "c17-103", ("Y", "C", "test"): "c17-104", ("X", "C", "test"): "c17-105"}
for p, q, sp in (("X", "Y", "val"), ("X", "Y", "test"), ("Y", "C", "test"), ("X", "C", "test"), ("Y", "C", "val"), ("X", "C", "val")):
    d, m, lo, hi, pos = contrast(p, q, sp)
    t = tag.get((p, q, sp))
    line = (f"  {p} - {q} {sp}: mean {sg(m)} ({sg(lo)} to {sg(hi)}); higher in {pos} of 3 seeds; per seed "
            + " ".join(sg(x, 3) for x in d))
    print(line + (f"  [{t}]" if t else ""))
    if (p, q, sp) == ("X", "Y", "val"):
        print(f"  X - Y validation: higher in {pos} of 3 seeds, interval {sg(lo)} to {sg(hi)}  [c17-102]")
    if (p, q, sp) == ("X", "Y", "test"):
        print(f"  X - Y test: higher in {pos} of 3 seeds  [c17-103]")

print("\nControl against earlier jobs of the same recipe: beam-5 test of control C (this job) minus the earlier control, per seed")
for tag, name in (("wave16A", "wave 16 control A"), ("wave6A", "wave 6 control A")):
    print(f"  {name}: " + "; ".join(f"seed {s}: this job {bleu['C', s, 'test']:.3f}, earlier {earlier[tag, s]:.3f}, difference {bleu['C', s, 'test'] - earlier[tag, s]:+.3f}" for s in SEEDS))
gap = bleu["C", 101, "test"] - earlier["wave16A", 101]
print(f"  the control differed from the wave 16 control (same scores as the wave-5 control the read-out compares with) by {abs(gap):.2f} at seed 101, and by {abs(bleu['C', 42, 'test'] - earlier['wave16A', 42]):.2f} and {abs(bleu['C', 202, 'test'] - earlier['wave16A', 202]):.2f} at seeds 42 and 202  [c17-106]")
print(f"  difference at seed 101 against wave 16 control A: {gap:+.3f}; against wave 6 control A: {bleu['C', 101, 'test'] - earlier['wave6A', 101]:+.3f} (this job's control reproduces wave 6's at this seed)")
import re
line = next(l for l in open(HERE / "summary_179457.txt") if l.startswith("GATE:"))
dd = {int(k): float(v) for k, v in re.findall(r"(\d+): (-?[\d.]+)", line.split("{", 1)[1].split("}")[0])}
print(f"  read from the saved job output summary_179457.txt: differences from wave 5's control (seed: got - control) {dd}")
for s in SEEDS:
    mine = bleu["C", s, "test"] - earlier["wave16A", s]
    if abs(mine - dd[s]) > 1e-3:
        sys.exit(f"GATE FAILED: seed {s} difference {mine:+.3f} vs the read-out's {dd[s]}")
print("GATE: the differences from wave 16 control A (recomputed here) equal the read-out job's recorded differences from wave 5's control at all three seeds to 1e-3, "
      "so the wave 16 control carries wave 5's scores  [c17-106]")

print("\nTraining length and seeds (read from the saved job output progress.json of each run)")
for a in "CXY":
    for s in SEEDS:
        p = json.load(open(HERE / "runs" / f"{a}_s{s}_progress.json"))
        assert p["status"] == "done" and p["epochs"] == 40 and len(p["history"]) == 40, (a, s)
print("  all 9 runs (C, X, Y x seeds 42/101/202) completed 40 training epochs of the full loss (progress.json)")
print(f"  seeds per arm: {len(SEEDS)}  [c17-101]")
