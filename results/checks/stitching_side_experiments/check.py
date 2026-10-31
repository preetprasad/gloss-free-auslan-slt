"""PoseStitch v2 side experiments: shuffled sign order (wave 10), wider-vocabulary corpus (wave 6), dropping function-word signs
(wave 7), fingerspelled-letter signs (wave 2), wrong-sense signs (wave 12b corpus facts).

Recomputes beam-5 BLEU-4 of every run from its saved hypotheses and the official validation/test references, checks them
against the job summaries, then forms the paired differences. Per-run values of wave 2 are read from the saved job summaries."""
import json, re, ast, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

D = HERE / "data"
J = lambda n: json.load(open(D / n))
refs = {sp: J(f"refs_{sp}.json")["refs"] for sp in ("val", "test")}
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
    assert sorted(m.values()) == sorted(refs["test"]), "saved test references differ from the test manifest"
    print("saved test references = comm_run/manifests/test.tsv (as a set of 800)")


def bleu(h, sp):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in refs[sp]])


# ---- GATE: every run's recomputed beam-5 BLEU must equal the value in the job summary ----
sets = {"wave10": J("hyps_wave10.json"), "wave7": J("hyps_wave7.json"), "wave7s2": J("hyps_wave7s2.json"), "wave6": J("hyps_wave6.json")}
bad = 0
summ_w10, summ_w7, summ_w6 = {}, {}, {}
for s in (42, 101, 202):
    for a, d in J(f"wave10_summary_s{s}.json").items(): summ_w10[f"{a}_s{s}"] = d
for s in (42, 101, 202, 303, 404):
    for a, d in J(f"wave7_summary_s{s}.json").items(): summ_w7[f"{a}_s{s}"] = d
for s_ in (42, 101, 202, 303, 404):
    for a, d in J(f"wave6_summary_s{s_}.json").items(): summ_w6[f"{a}_s{s_}"] = d
rec = {}  # (wave, run) -> (val, test)
n = 0
for w, tab in (("wave10", summ_w10), ("wave7", summ_w7), ("wave7s2", summ_w7), ("wave6", summ_w6)):
    for run, v in sets[w].items():
        rv, rt = bleu(v["val"], "val"), bleu(v["test"], "test")
        rec[(w, run)] = (rv, rt)
        n += 1
        if abs(rv - tab[run]["beam5_val"]) > 1e-6 or abs(rt - tab[run]["beam5_test"]) > 1e-6:
            bad += 1
            print("MISMATCH", w, run, rv, tab[run]["beam5_val"], rt, tab[run]["beam5_test"])
print(f"GATE {n} runs: BLEU-4 recomputed from the saved beam-5 hypotheses equals the job summary (expected 0 mismatches, found {bad})")
if bad:
    sys.exit("GATE FAILED")

def get(w, run):
    return rec[(w, run)]

def paired(wave_of, arm, ctl, seeds, which):
    """per-seed arm - control beam-5 differences, wave_of(seed) -> wave name"""
    i = 0 if which == "val" else 1
    return np.array([get(wave_of(s), f"{arm}_s{s}")[i] - get(wave_of(s), f"{ctl}_s{s}")[i] for s in seeds])

def sgn(x, d=2):
    return f"{x:+.{d}f}"

def tint(x):
    mu, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return mu - h, mu + h

# ---------------- wave 10: shuffled sign order ----------------
S3 = (42, 101, 202)
w10 = lambda s: "wave10"
print("\nWAVE 10 shuffled sign order (3 seeds, beam-5 best.pt; difference to the English-order control A)")
for arm, ids in (("S", {"val": "c17-063", "test": "c17-063"}), ("SC", {"val": "c17-064", "test": "c17-064"})):
    for which in ("val", "test"):
        d = paired(w10, arm, "A", S3, which)
        name = "validation" if which == "val" else "test"
        print(f"{arm} - A {name} per seed {' '.join(sgn(x, 3) for x in d)}; mean {sgn(d.mean())}  [{ids[which]}]")

# ---------------- wave 6: wider-vocabulary corpus ----------------
S5 = (42, 101, 202, 303, 404)
w6f = lambda s: "wave6"
print("\nWAVE 6 wider-vocabulary stitched corpus (B: 1.1:1; C: twice the amount)")
dv = paired(w6f, "B", "A", S5, "val"); dt = paired(w6f, "B", "A", S5, "test")
lo, hi = tint(dt)
print(f"B - A validation {sgn(dv.mean())} ({int((dv > 0).sum())}/5 positive)  [c16-032]")
print(f"B - A test {sgn(dt.mean())} [{sgn(lo)}, {sgn(hi)}] (95% t-interval over 5 seeds)  [c16-032]")
cv = paired(w6f, "C", "A", S3, "val")
print(f"C - A validation {sgn(cv.mean())} ({int((cv > 0).sum())}/3 positive; stage 1 uses the first three seeds)  [c16-032]")
print(f"seeds used for B vs A: {len(S5)}; for C vs A: {len(S3)} (counted from the saved job summaries)")

# ---------------- wave 7: dropping function-word signs ----------------
print("\nWAVE 7 function-word signs dropped (D vs A, 5 seeds; DL = length-matched arm)")
w7f = lambda s: "wave7" if s in S3 else "wave7s2"
dv = paired(w7f, "D", "A", S5, "val"); dt = paired(w7f, "D", "A", S5, "test")
lo, hi = tint(dt)
print(f"D - A validation {sgn(dv.mean())} ({int((dv > 0).sum())}/5 positive)  [c16-034]")
print(f"D - A test {sgn(dt.mean())} [{sgn(lo)}, {sgn(hi)}] ({int((dt > 0).sum())}/5 positive; fresh seeds 303 and 404: {sgn(dt[3], 3)}, {sgn(dt[4], 3)})  [c16-034]")
dl_seeds = [s for s in (303, 404) if ("wave7s2", f"DL_s{s}") in rec]
print(f"DL (length-matched) arm run at {len(dl_seeds)} seeds {dl_seeds}  [c16-034]")
dlt = np.array([get("wave7s2", f"DL_s{s}")[1] - get("wave7s2", f"A_s{s}")[1] for s in dl_seeds])
print(f"DL - A test per seed {' '.join(sgn(x, 3) for x in dlt)} (descriptive)")
ro = (D / "wave7_readout_output.txt").read_text()
mm = re.search(r"content_precision\s+(\S+ \S+ \S+ \S+ \S+) \| (\d)/5 \+ \| mean ([+-][\d.]+)", ro.split("-- test")[1])
print(f"content-word precision, D - A, test: mean {float(mm.group(3)):.2f} ({mm.group(2)}/5 positive) "
      f"(read from the saved job output wave7_readout_output.txt; the tool that wrote it is src/tools/arm_readouts.py)  [c16-034]")
prep = (D / "wave7_prep_66643.out").read_text()
mp = re.search(r"removed (\d+) of (\d+) spans \(([\d.]+)%\)", prep)
print(f"function-word signs dropped: {mp.group(1)} of {mp.group(2)} spans = {100*int(mp.group(1))/int(mp.group(2)):.0f}% "
      f"(read from the saved job output wave7_prep_66643.out)  [c16-033]")
print(f"seeds compared: {len(S5)}  [c16-033]")

# ---------------- constrained sentence generation ----------------
au = json.load(open(ROOT / "results/stitched_corpus_audit/stitch_corpus_audit.json"))["syn_20608_pool"]
print(f"constrained generation pool: {au['sentences']} sentences, {au['unique']} unique "
      f"(read from the saved corpus-audit output stitch_corpus_audit.json; the 6,880 unconstrained sentences are not saved locally)")

# ---------------- adopted corpus: letters and wrong-sense signs ----------------
g = J("adopted_corpus_gloss_counts.json")
tot = sum(g.values())
letters = sum(c for k, c in g.items() if len(k) == 1)
facts = (D / "wave12b_corpus_facts_output.txt").read_text()
cf = ast.literal_eval(re.search(r"counts: (\{.*\})", facts.splitlines()[0]).group(1))
wrong = sum(g.get(k, 0) for k in cf["per_gloss"])
print("\nADOPTED STITCHED CORPUS (13,760 clips; segment list of the corpus audit)")
print(f"GATE signs {tot} (recorded in the corpus-facts output: {cf['signs']}); wrong-sense signs {wrong} (recorded: {cf['wrong']})")
if tot != cf["signs"] or wrong != cf["wrong"]:
    sys.exit("GATE FAILED")
print(f"single-letter (fingerspelled alphabet) signs: {letters} of {tot} = {100*letters/tot:.1f}% of stitched signs  [c17-079]")
print(f"wrong-sense signs (the 11 sense-disambiguated glosses): {wrong} of {tot} = {100*wrong/tot:.2f}% of the signs  [c17-048]")

# ---------------- wave 12b: wrong-sense signs removed (W) vs the adopted corpus (A), job 138380 / summary 138381 ----------------
print("\nWAVE 12b wrong-sense signs removed (W - A, 3 seeds, beam-5 best.pt; per-run values read from the saved job output wave12b_summary_138381.out)")
log = (D / "wave12b_summary_138381.out").read_text()
runs = {}
for arm, js in re.findall(r"^([AW]) (\{.*\})$", log, re.M):
    runs.setdefault(arm, []).append(json.loads(js))
S12 = (42, 101, 202)
assert len(runs["A"]) == len(runs["W"]) == len(S12), "expected three A and three W runs in the job output"
# GATE: the per-seed differences formed here equal the 'W - A (seed s)' lines the job printed
printed = re.findall(r"W - A \(seed (\d+)\): beam-5 val ([+-][\d.]+) \| beam-5 test ([+-][\d.]+)", log)
dv12 = np.array([w["beam5_val"] - a["beam5_val"] for a, w in zip(runs["A"], runs["W"])])
dt12 = np.array([w["beam5_test"] - a["beam5_test"] for a, w in zip(runs["A"], runs["W"])])
for (s, pv, pt), v, t in zip(printed, dv12, dt12):
    assert int(s) in S12 and abs(float(pv) - v) < 6e-4 and abs(float(pt) - t) < 6e-4, ("GATE FAILED", s, pv, v, pt, t)
print(f"GATE per-seed W - A differences recomputed from the six run records equal the job's printed lines ({len(printed)} seeds)")
lo_v, hi_v = tint(dv12); lo_t, hi_t = tint(dt12)
print(f"W - A validation per seed {' '.join(sgn(x, 3) for x in dv12)}; mean {sgn(dv12.mean(), 4)} = {sgn(dv12.mean(), 3)} to 3 decimals, which the dissertation rounds again to {sgn(round(round(dv12.mean(), 3) + 1e-9, 2))} (a single rounding gives {sgn(dv12.mean())}); 95% t-interval [{sgn(lo_v)}, {sgn(hi_v)}] over {len(S12)} seeds  [c17-049 c17-048]")
print(f"W - A test per seed {' '.join(sgn(x, 3) for x in dt12)}; mean {sgn(dt12.mean())} [{sgn(lo_t)}, {sgn(hi_t)}] (95% t-interval over {len(S12)} seeds)  [c17-049 c17-048]")
mv = re.search(r"beam5_val .*mean ([+-][\d.]+) \| 95% t \[([+-][\d.]+), ([+-][\d.]+)\]", log)
mt = re.search(r"beam5_test .*mean ([+-][\d.]+) \| 95% t \[([+-][\d.]+), ([+-][\d.]+)\]", log)
for nm, m, mean, lo, hi in (("validation", mv, dv12.mean(), lo_v, hi_v), ("test", mt, dt12.mean(), lo_t, hi_t)):
    assert all(abs(float(a) - b) < 6e-4 for a, b in zip(m.groups(), (mean, lo, hi))), ("GATE FAILED", nm)
print("GATE mean and 95% t-interval equal the job's own seed_decision.py lines (validation and test)")

# ---------------- wave 2: letter fix ----------------
print("\nWAVE 2 letter-fixed corpus (w2LF vs w2A, beam-5 best.pt, read from the saved job summaries)")
d2 = {s: J(f"wave2_summary_s{s}.json") for s in S3}
v = np.array([d2[s]["w2LF"]["beam5_val"] - d2[s]["w2A"]["beam5_val"] for s in S3])
t = np.array([d2[s]["w2LF"]["beam5_test"] - d2[s]["w2A"]["beam5_test"] for s in S3])
print(f"w2LF - w2A validation per seed {' '.join(sgn(x, 3) for x in v)}; mean {sgn(v.mean())}  [c17-080]")
print(f"w2LF - w2A test per seed {' '.join(sgn(x, 3) for x in t)}; mean {sgn(t.mean())}  [c17-080]")
print(f"seeds: {len(S3)}  [c17-079]")
