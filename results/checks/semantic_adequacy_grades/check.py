"""Semantic adequacy grading (Qwen2.5-3B-Instruct judge, 0-4): means of four systems and the paired comparison.

Recomputes from the saved per-clip grades of the re-grading job (10288). Gate: each recomputed mean
equals the mean printed in that job's saved log.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "grades.jsonl")]
sysn = ["dwpose_ctc", "vlp", "rgbfusion", "s6"]
g = {s: {r["idx"]: r["adequacy"] for r in rows if r["system"] == s} for s in sysn}
for s in sysn:
    log = open(HERE / f"{s}.log").read()
    m = re.search(r"model=(\S+) field=\S+ graded=(\d+)/(\d+)\s*\nmean=([\d.]+)", log)
    model, k, n, rec = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    mean = sum(g[s].values()) / len(g[s])
    print(f"GATE {s}: recomputed mean {mean:.3f} vs recorded {rec:.3f} (read from the saved job log {s}.log)")
    if len(g[s]) != n or k != n or round(mean, 3) != rec:
        sys.exit("GATE FAILED")
    assert set(g[s].values()) <= {0, 1, 2, 3, 4}
    g[s + "_model"] = model

models = {g[s + "_model"] for s in sysn}
assert models == {"Qwen/Qwen2.5-3B-Instruct"}
print("\nJudge model (read from the saved job logs): Qwen2.5-3B-Instruct  [c20-008]")
print(f"Scale: grades take values {min(min(g[s].values()) for s in sysn)} to {max(max(g[s].values()) for s in sysn)}, i.e. 0-4  [c20-008]")
print(f"Systems graded: {len(sysn)}, each on {len(g['s6'])} official test clips  [c20-008]")

mean = {s: sum(g[s].values()) / len(g[s]) for s in sysn}
print("\nMean adequacy")
for s in sysn:
    print(f"  {s:11s} {mean[s]:.3f}")
print(f"Baseline model (DWPose+CTC) mean adequacy {mean['dwpose_ctc']:.3f}  [c20-009]")
print(f"Final ensemble (S6) mean adequacy {mean['s6']:.3f}  [c20-009]")

a, b = g["s6"], g["rgbfusion"]
assert a.keys() == b.keys()
hi = sum(a[i] > b[i] for i in a); lo = sum(a[i] < b[i] for i in a)
print(f"\nFinal ensemble vs RGB-fusion model, per clip: graded higher on {hi} clips and lower on {lo} (equal on {len(a)-hi-lo})  [c18-022]")
