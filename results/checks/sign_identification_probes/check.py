"""Sign-identification probes (dissertation: Results, Discussion, Appendix 'isolated-sign references').

Per-span saved outputs of the recorded probe (pred_dwpose.csv), the located table of Auslan-Daily's isolated clips
(located.csv), the MM-WLAuslan label counts (mmwl_gloss_counts.json) and the saved job logs of the probes
(log_dwpose.txt, job_probe_a2_a5_depth.out).  Nothing is trained; the probes themselves need pose arrays.
"""
import csv, json, re, sys, statistics
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG12F = HERE / "job_probe_a2_a5_depth.out"


def fail(m):
    sys.exit("GATE FAILED: " + m)


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(a, b) + 1)) / 2 ** n)


# ---- spans, glosses (per-item files)
spans = list(csv.DictReader(open(HERE / "spans45.csv")))
G45 = {r["Query_Gloss"].strip().lower() for r in spans}
pred = [r for r in csv.DictReader(open(HERE / "pred_dwpose.csv")) if r["variant"] == "hands"]
k = sum(int(r["correct"]) for r in pred); n = len(pred)

# gate: the recorded run (job log of the probe) scored hands 61/337
log = open(LOG12F).read()
m = re.search(r"A0 conversational spans.*?n\s+(\d+)\s+top-1\s+([\d.]+)% \((\d+)\)", log)
rec_n, rec_k = int(m.group(1)), int(m.group(3))
print(f"GATE per-span file hands correct {k}/{n}; recorded in the saved probe job output {rec_k}/{rec_n} (expected 61/337)")
if (k, n) != (rec_k, rec_n) or (k, n) != (61, 337):
    fail("per-span file does not reproduce the recorded run")

print("\nCONVERSATIONAL SPANS (the recorded probe, hands descriptor, clip-disjoint folds)")
chance = 100 / len(G45)
print(f"annotated spans {len(spans)} of {len(G45)} signs; chance 1/{len(G45)} = {chance:.1f}%  [c08-046 c10-051]")
print(f"signs {len(G45)}  [c08-046 c10-051 c14-071 c14-070 c10-041 c14-047]")
print(f"chance {chance:.1f}%  [c01-018 c02-021 c08-046 c10-040 c14-071]")
print(f"spans {len(spans)}  [c08-046 c10-051]")
print(f"scored spans {n}  [c14-070]")
print(f"hands top-1 {100*k/n:.1f}% ({k}/{n})  [c01-018 c02-021 c10-040 c14-072 c18-077]")

# ---- isolated clips of Auslan-Daily
L = list(csv.DictReader(open(HERE / "located.csv")))
print(f"\nAUSLAN-DAILY ISOLATED CLIPS: {len(L):,} segments over {len({r['Query_Gloss'].lower() for r in L})} glosses  [c04-039]")
refs = [r for r in L if r["Query_Gloss"].lower() in G45 and r["status"] == "located"
        and r["comm_split"] == "train" and int(r["frames"]) >= 3]
print(f"isolated clips of the {len(G45)} signs on a TRAIN clip with at least 3 frames: {len(refs)}  [c14-070]")
cnt = []
for r in pred:
    c = r["span"].split(":")[0]
    cnt.append(sum(1 for x in refs if x["Query_Gloss"].lower() == r["gloss"] and x["Video_Name_Clip"] != c))
print(f"same-sign references per query (own clip excluded): mean {statistics.mean(cnt):.1f}, min {min(cnt)}, max {max(cnt)}  [c14-070]")

# ---- MM-WLAuslan citation recordings
mm = json.load(open(HERE / "mmwl_gloss_counts.json"))
spot = {r["Query_Gloss"].strip().lower() for r in csv.DictReader(open(HERE / "spans_safe.csv"))}
shared = spot & set(mm)
print(f"\nSpotting glosses {len(spot)}; also MM-WLAuslan glosses: {len(shared)} shared signs  [c10-041 c14-047]")
if shared != G45:
    fail("shared glosses differ from the probe's 45")
tot = sum(mm[g] for g in G45); per = sorted({mm[g] for g in G45})
print(f"MM-WLAuslan citation recordings of the {len(G45)} signs: {tot}  [c14-070]")
print(f"recordings per sign: {per[0]} (min) to {per[-1]} (max), i.e. {tot/len(G45):.0f} per sign  [c14-070]")

# ---- read from the saved probe job output (12f)
blocks = log.split("\n=== ")
hands = next(b for b in blocks if b.startswith("hands:") and "A2 isolated" in b)
def pct(label, text):
    mm_ = re.search(re.escape(label) + r".*?top-1\s+([\d.]+)% \((\d+)\)", text)
    return float(mm_.group(1)), int(mm_.group(2))
a2, a2k = pct("A2 isolated, own clip excluded [PRIMARY]", hands)
a5, a5k = pct("A5 MM-WLAuslan citation recordings", hands)
print("\nISOLATED CLIPS AS REFERENCES (read from the saved job output job_probe_a2_a5_depth.out)")
print(f"isolated clips as references, top-1 {a2:.1f}% ({a2k}/{n}) against conversational spans {100*k/n:.1f}%  [c14-052 c14-072 c18-077]")
print(f"citation-form recordings as references, top-1 {a5:.1f}% ({a5k}/{n})  [c14-052 c14-072 c18-078]")
for lab, key in (("isolated clips (A2) minus spans", "A2 against A0"),):
    mc = re.search(key + r".*?only first (\d+), only second (\d+), exact McNemar p = ([\d.e-]+)", hands)
    a, b = int(mc.group(1)), int(mc.group(2)); p = mcnemar(a, b)
    print(f"{lab}: only first {a}, only second {b}; exact McNemar p recomputed {p:.2f} (log {float(mc.group(3)):.2f})  [c18-077]")
    if abs(p - float(mc.group(3))) > 0.001: fail("McNemar p differs")
dep = next(b for b in log.split("\n\n") if b.startswith("45 glosses of the recorded run | hands"))
zx = pct("x/y, 3-D model", dep); zxyz = pct("x/y/z, 3-D model", dep); zz = pct("z only, 3-D model", dep)
print("\nESTIMATED HAND DEPTH (read from the saved job output)")
print(f"x/y/z top-1 {zxyz[0]:.1f}% against x/y {zx[0]:.1f}% (same 3-D model)  [c18-079]")
mc = re.search(r"x/y/z against x/y.*?only first (\d+), only second (\d+), exact McNemar p = ([\d.e-]+)", dep)
a, b = int(mc.group(1)), int(mc.group(2)); p = mcnemar(a, b)
print(f"only first {a}, only second {b}; exact McNemar p recomputed {p:.3f} (log {float(mc.group(3)):.3f})  [c18-079]")
if abs(p - float(mc.group(3))) > 0.0005: fail("McNemar p differs")
nl = re.search(r"z only: top-1 ([\d.]+)%; label-permutation null.*?95th percentile ([\d.]+)%", dep)
print(f"depth alone {zz[0]:.1f}% above a permutation null 95th percentile {float(nl.group(2)):.1f}%  [c18-080]")
