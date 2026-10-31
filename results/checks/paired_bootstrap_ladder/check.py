"""Paired-bootstrap comparisons, development ladder and S6 against the published systems.

Recomputes from saved per-sentence decodes of the 800 official test clips: BLEU-4 of the baseline, the 40-epoch
pre-trained model, the RGB-fusion model, the six ensemble members and the final ensemble; the 2,000-resample paired
bootstrap (tools/paired_bootstrap.py convention: seed 0, one shared draw matrix, percentiles 2.5 / 97.5); the
ensemble's own interval; the judge-adequacy bootstrap (tools/adequacy_bootstrap.py convention).
"""
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4, corpus_wer
from sacrebleu.metrics import BLEU

def load(p):
    return sorted((json.loads(l) for l in open(p, encoding="utf-8")), key=lambda r: r["idx"])

SG = ROOT / "results/semantic_adequacy_grading"
S6P = ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl"
MEM = ROOT / "results/final_ensemble_member_test_decodes"
sysrows = {"CTC": load(SG / "hyps_dwpose_ctc.jsonl"), "VLP": load(SG / "hyps_vlp.jsonl"),
           "RGB": load(SG / "hyps_rgbfusion.jsonl"), "S6": load(S6P)}
refs_raw = [r["ref"] for r in sysrows["S6"]]
for k, v in sysrows.items():
    assert [r["ref"] for r in v] == refs_raw, k + " references differ"
R = [detokenize(r) for r in refs_raw]
n = len(R)
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
    assert sorted(m.values()) == sorted(r.strip() for r in refs_raw), "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")

# stitched-data ensembles (Table res_bootstrap rows 5-6, Figure res_bootstrap rows 4-7)
for k, d, lp in (("F", "ensemble_three_duration_two_fusion_decodes", "0.7"), ("G1", "ensemble_six_duration_two_fusion_decodes", "0.7"), ("Fp", "ensemble_one_to_one_duration_members_decodes", "0.6")):
    sysrows[k] = load(ROOT / "results" / d / f"hyps_LP={lp}.jsonl")
    assert [r["ref"] for r in sysrows[k]] == refs_raw, k + " references differ"
H = {k: [detokenize(r["hyp_beam"]) for r in v] for k, v in sysrows.items()}
full = {k: corpus_bleu4(h, R) for k, h in H.items()}
print(f"GATE final ensemble re-scores to {full['S6']:.3f} (recorded 11.802)")
if abs(full["S6"] - 11.802) > 0.0015:
    sys.exit("GATE FAILED")
print(f"GATE baseline {full['CTC']:.3f} (recorded 9.522), 40-epoch VLP {full['VLP']:.3f} (recorded 10.272), RGB fusion {full['RGB']:.3f} (recorded 10.335)")
if any(abs(full[k] - v) > 0.0015 for k, v in (("CTC", 9.522), ("VLP", 10.272), ("RGB", 10.335))):
    sys.exit("GATE FAILED")
print(f"GATE stitched-data ensembles {full['F']:.3f} (recorded 12.114), {full['G1']:.3f} (recorded 11.562), {full['Fp']:.3f} (recorded 12.836)")
if any(abs(full[k] - v) > 0.0015 for k, v in (("F", 12.114), ("G1", 11.562), ("Fp", 12.836))):
    sys.exit("GATE FAILED")

# ---- ladder (Table res_ladder) ----
g = {k: corpus_bleu4([detokenize(r["hyp_greedy"]) for r in sysrows[k]], R) for k in ("VLP", "RGB")}
print("\nDEVELOPMENT LADDER, official test, 800 clips")
print(f"+ VLP pretraining, 40 epochs: greedy {g['VLP']:.2f}, beam 5 {full['VLP']:.2f}  [c09-012]")
print(f"+ RGB fusion: greedy {g['RGB']:.2f}, beam 5 {full['RGB']:.2f}  [c09-013 c09-005 c14-031 c21-007]")
print(f"RGB fusion adds {full['RGB']-full['VLP']:+.2f} over the 40-epoch pre-trained model  [c09-013]")
s6 = full["S6"]
print(f"Final ensemble: beam 10, length penalty 0.3, BLEU-4 {s6:.2f} ({s6:.3f})  [c09-005 c14-005 c14-031 c21-007 c21-025 c11-028]")
wer = corpus_wer(H["S6"], R)
print(f"Final ensemble WER {wer:.3f} on the {n}-clip official test set  [c21-025]")
print(f"Official test set size: {n} clips  [c21-025 c12-001 c19-017 c12-012 c02-025 c12-003 c14-031 c21-007]")

# ---- members, decoded alone with the ensemble decoder settings ----
mem = {}
for d in sorted(MEM.iterdir()):
    rows = load(d / "hyps_LP=0.3.jsonl")
    assert [r["ref"] for r in rows] == refs_raw
    mem[d.name] = corpus_bleu4([detokenize(r["hyp_beam"]) for r in rows], R)
ck = re.search(r'--ckpts "\$S6"', open(ROOT / "jobs/slurm/mbr_wer_test2.sbatch").read())
s6line = [l for l in open(ROOT / "jobs/slurm/mbr_wer_test2.sbatch") if l.startswith("S6=")][0]
listed = [Path(p).parent.name for p in re.findall(r"\$G/([^,\"]+/best\.pt)", s6line)]
assert ck and sorted(listed) == sorted(mem), (listed, sorted(mem))
print(f"\nEnsemble member list, read from jobs/slurm/mbr_wer_test2.sbatch (the job whose --ckpts decoded the final test ensemble at length penalty 0.3): {len(listed)} translators averaged  [c11-088]")
print(f"the {len(listed)} listed checkpoints are exactly the {len(mem)} member dumps used below: {sorted(listed) == sorted(mem)}  [c11-088]")
print(f"Members decoded alone with the ensemble decoder settings ({len(mem)} dumps)  [c21-007]")
for k, v in mem.items():
    print(f"  member {k}: {v:.2f}")
best_name = max(mem, key=mem.get)
best = mem[best_name]
print(f"strongest member ({best_name}) {best:.2f}; ensemble {s6:.2f}; ensemble minus strongest member {s6-best:+.2f}  [c21-007]")

# ---- published comparison ----
PUB = 9.95
print(f"\nGASLT published score 9.95 (published value, Auslan-Daily benchmark paper); ensemble minus 9.95 = {s6-PUB:+.2f}  [c02-015 c11-004 c11-028 c13-053 c14-033 c21-007 c21-025]")
print(f"relative to GASLT published: {100*(s6-PUB)/PUB:.1f}% (published value 9.95)  [c11-004]")

# ---- bootstrap ----
# per-sentence n-gram statistics; corpus BLEU of a resample = BLEU from the summed statistics
bl = BLEU(tokenize="13a")
def stats(h):
    return np.array(bl._extract_corpus_statistics(h, [R]), dtype=np.float64)  # n x 10
ST = {k: stats(h) for k, h in H.items()}
def score(st):
    return bl._compute_score_from_stats(list(st.sum(0))).score
for k in ST:  # fast scorer must equal the project scorer on the full set
    assert abs(score(ST[k]) - full[k]) < 1e-9, k
NRES, SEED = 2000, 0
tool = (ROOT / "src/tools/paired_bootstrap.py").read_text()
t_n = int(re.search(r'"--n", type=int, default=(\d+)', tool).group(1))
t_seed = int(re.search(r'"--seed", type=int, default=(\d+)', tool).group(1))
t_ci = float(re.search(r'"--ci", type=float, default=([\d.]+)', tool).group(1))
assert "np.random.default_rng(a.seed)" in tool and "rng.integers(0, len(pool), size=(a.n, len(pool)))" in tool
assert (t_n, t_seed, t_ci) == (NRES, SEED, 95.0), (t_n, t_seed, t_ci)
print(f"\nSettings read from the source of src/tools/paired_bootstrap.py (the tool that produced Table res_bootstrap): "
      f"default {t_n:,} resamples, seed {t_seed}, {t_ci:.0f}% interval, one shared draw matrix of sentence indices over the {n} sentences; "
      f"this check uses the same convention  [c07-064 c07-066 c02-025 c12-001 c12-003 c19-017]")
rng = np.random.default_rng(SEED)
draws = rng.integers(0, n, size=(NRES, n))
def boot(k):
    return np.array([score(ST[k][ii]) for ii in draws])
BS = {k: boot(k) for k in ST}
lo_p, hi_p = 2.5, 97.5
print(f"\nPAIRED BOOTSTRAP of BLEU-4 differences (BLEU-4, 4-gram): {NRES:,} resamples with replacement of the {n} test sentences, seed {SEED}; "
      f"interval = {lo_p}th and {hi_p}th percentiles of the difference = 95% interval  [c07-064 c07-066 c02-025 c12-001 c12-003 c12-012 c19-017]")
def pair(a, b, tag):
    d = BS[a] - BS[b]
    obs = full[a] - full[b]
    lo, hi = np.percentile(d, [lo_p, hi_p])
    print(f"{a} minus {b}: {obs:+.2f} (observed {obs:+.3f}), 95% interval [{lo:+.2f}, {hi:+.2f}], P(>0) {100*np.mean(d>0):.1f}%, "
          f"{'resolvable' if lo>0 or hi<0 else 'not resolvable'}, half-width {(hi-lo)/2:.2f}  [{tag}]")
    return (hi - lo) / 2, lo, hi
ST["M"] = stats([detokenize(r["hyp_beam"]) for r in load(MEM / best_name / "hyps_LP=0.3.jsonl")]); BS["M"] = boot("M"); full["M"] = best
# recorded intervals of the original tool run (a saved output of tools/paired_bootstrap.py, 2,000 resamples, seed 0)
REC = {}
for l in open(HERE / "recorded_bootstrap_tool_output.txt"):
    m = re.match(r"\[all n=800\] (\w+) vs (\w+): observed ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", l)
    if m:
        REC[(m.group(1), m.group(2))] = tuple(float(m.group(i)) for i in (3, 4, 5))
alias = {"RGB": "RGBF", "CTC": "CTC", "VLP": "VLP", "S6": "S6", "F": "F", "G1": "G1", "Fp": "Fprime"}
res = {}
def pair(a, b, tag):
    d = BS[a] - BS[b]
    obs = full[a] - full[b]
    lo, hi = np.percentile(d, [lo_p, hi_p])
    print(f"{a} minus {b}: {obs:+.2f} (observed {obs:+.3f}), 95% interval [{lo:+.2f}, {hi:+.2f}], P(>0) {100*np.mean(d>0):.1f}%, "
          f"{'resolvable' if lo>0 or hi<0 else 'not resolvable'}, half-width {(hi-lo)/2:.2f}  [{tag}]")
    res[(a, b)] = ((hi - lo) / 2, lo, hi)
    key = (alias.get(a), alias.get(b))
    if key in REC:
        ro, rl, rh = REC[key]
        if max(abs(obs - ro), abs(lo - rl), abs(hi - rh)) > 0.0015:
            sys.exit(f"GATE FAILED: {a}-{b} differs from the recorded tool output {REC[key]}")
print("GATE every interval below equals the recorded output of tools/paired_bootstrap.py (saved in recorded_bootstrap_tool_output.txt) to 0.0015")
pair("S6", "RGB", "c09-005 c14-005 c14-031 c21-007 c12-006 c07-058")
pair("S6", "M", "c21-007")
pair("RGB", "CTC", "c12-006")
pair("VLP", "CTC", "c12-007")
pair("F", "S6", "c12-012")
pair("Fp", "S6", "c12-012")
print(f"(M = {best_name}, same decoder settings, {best:.2f}; table row 'its strongest member')  [c21-007]")
print(f"RGB fusion ({full['RGB']:.2f}) vs baseline ({full['CTC']:.2f}): P(>0) {100*np.mean(BS['RGB']-BS['CTC']>0):.1f}%  [c12-006]")
print(f"40-epoch pre-trained model ({full['VLP']:.2f}) vs baseline ({full['CTC']:.2f}): P(>0) {100*np.mean(BS['VLP']-BS['CTC']>0):.1f}%  [c12-007]")
print("Figure res_bootstrap rows 5 and 7, which are not in Table res_bootstrap (stitched-data ensembles):  [c12-012]")
pair("G1", "S6", "c12-012")
pair("Fp", "F", "c12-012")
print(f"Stitched-data ensemble scores: 0.55:1 {full['F']:.2f}, 1.1:1 {full['Fp']:.2f}, six-model {full['G1']:.2f}; final ensemble {full['S6']:.2f}  [c12-012]")
print(f"RGB-fusion checkpoint decoded alone with the ensemble decoder settings {mem['comm_dwpose_rgb_fusion']:.2f}; its standalone beam-5 decode (the ladder row) {full['RGB']:.2f}")
tab = [("S6", "RGB"), ("S6", "M"), ("RGB", "CTC"), ("VLP", "CTC"), ("F", "S6"), ("Fp", "S6")]
allp = tab + [("G1", "S6"), ("Fp", "F")]
h6 = [res[k][0] for k in tab]; h8 = [res[k][0] for k in allp]
print(f"95% half-widths, the {len(tab)} rows of Table res_bootstrap: " + ", ".join(f"{h:.2f}" for h in h6) +
      f"; range {min(h6):.2f} to {max(h6):.2f}, mean {np.mean(h6):.2f}")
print(f"95% half-widths, those {len(tab)} rows plus the two extra rows of Figure res_bootstrap ({len(allp)} comparisons): range {min(h8):.2f} to {max(h8):.2f}, mean {np.mean(h8):.2f}; "
      f"largest {max(h8):.2f}, median {np.median(h8):.2f}")
print(f"(single-system interval, not a paired difference) half-width of the {n}-sentence interval on the ensemble alone: {(np.percentile(BS['S6'],97.5)-np.percentile(BS['S6'],2.5))/2:.1f}")
only = [f"{a}-{b}" for (a, b), (h, lo, hi) in res.items() if lo > 0 or hi < 0]
print(f"intervals excluding zero among the {len(res)} compared pairs: {only}  [c07-058 c12-012]")

lo, hi = np.percentile(BS["S6"], [lo_p, hi_p])
print(f"\nFinal ensemble's own 95% bootstrap interval over test sentences: {lo:.2f} to {hi:.2f}  [c11-029 c21-023]")
print(f"interval includes GASLT's published 9.95: {lo < PUB < hi}  [c21-023]")

# ---- judge adequacy ----
ad = json.load(open(HERE / "adequacy_grades.json"))
ga, gb = np.array(ad["s6"], float), np.array(ad["rgbfusion"], float)
d = ga - gb
r2 = np.random.default_rng(0)
bt = np.array([d[r2.integers(0, len(d), len(d))].mean() for _ in range(2000)])
l2, h2 = np.percentile(bt, [2.5, 97.5])
print(f"\nJudge adequacy (0-4), read from the saved per-clip grades of job 10288: final ensemble {ga.mean():.3f}, RGB-fusion member {gb.mean():.3f}  [c20-010]")
print(f"difference {d.mean():+.3f}, 95% interval {l2:+.2f} to {h2:+.2f} (paired bootstrap over {len(d)} clips, 2,000 resamples, seed 0)  [c20-010]")

# ---- protocol statements ----
import importlib.util
from scipy import stats as sps
spec = importlib.util.spec_from_file_location("seed_decision", ROOT / "src/tools/seed_decision.py")
sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
bad = [k for k, v in sd.T975.items() if abs(v - sps.t.ppf(0.975, k)) > 5e-4]
assert not bad, f"seed_decision.T975 differs from Student's t at df {bad}"
print(f"\nsrc/tools/seed_decision.py: T975[df] (two-sided 95%) matches Student's t quantile for every df 1..{max(sd.T975)}, df = n - 1 [GATE]")
for nn in (3, 5):
    d = list(np.linspace(0.0, 1.0, nn))     # n paired differences
    m, lo_, hi_ = sd.interval(d)
    want = sps.t.ppf(0.975, nn - 1) * np.std(d, ddof=1) / np.sqrt(nn)
    assert abs((hi_ - lo_) / 2 - want) < 1e-3, nn
    print(f"seed_decision.interval on {nn} paired differences uses T975[{nn-1}] = {sd.T975[nn-1]} (Student's t, n - 1 = {nn-1} degrees of freedom): "
          f"half-width {(hi_-lo_)/2:.3f} equals t(0.975, {nn-1}) x s / sqrt({nn}) = {want:.3f}  [c07-057]")
def seeds(path, pat):
    for i, l in enumerate(open(path), 1):
        if re.search(pat, l):
            return i, l.strip()
i, l = seeds(ROOT / "jobs/slurm/wave1_r2.sbatch", r"^SEEDS=")
print(f"Stage 1 seeds, read from jobs/slurm/wave1_r2.sbatch line {i}: {l}  (three seeds: 42, 101, 202)  [c07-052 c07-054]")
i, l = seeds(ROOT / "jobs/slurm/w4_train.sbatch", r"^S=\(")
print(f"Stage 2 seeds, read from jobs/slurm/w4_train.sbatch line {i}: {l}  (303 and 404 follow 42, 101, 202)  [c07-055]")
