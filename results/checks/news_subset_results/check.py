"""Claims check: News wave 4 (nA vs nSA, five seeds), validation-LP decoding, signable-word recall, ensemble gate.
Run: cd src && python3 ../results/checks/news_subset_results/check.py"""
import json, re, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
from sacrebleu.metrics import BLEU

D = json.load(open(HERE / "data/stats.json"))
SEEDS, LPS, T975 = (42, 101, 202, 303, 404), ("0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "2.0"), 2.776  # df 4
bl = BLEU()
sc = lambda t: bl._compute_score_from_stats(t)

# GATE: every beam-5 BLEU-4 recomputed from n-gram counts equals the value recorded by its job; every LP-sweep value equals the job log
for k, r in D["runs"].items():
    if abs(sc(r["tot"]).score - r["recorded"]) > 1e-6: sys.exit(f"GATE FAILED {k}")
for k, sw in D["sweeps"].items():
    for lp, v in sw.items():
        if abs(sc(v["tot"]).score - v["log"]) > 0.0006: sys.exit(f"GATE FAILED sweep {k} LP {lp}")
print("gate: 20 beam-5 BLEU-4 values (nA, nSA x 5 seeds x val/test) and all LP-sweep values recomputed from n-gram counts equal the values recorded by the jobs")

def ci(v):
    m = st.mean(v); h = T975 * st.stdev(v) / len(v) ** 0.5
    return m, m - h, m + h
B5 = {(a, s, sp): D["runs"][f"{a}_s{s}_{sp}"]["recorded"] for a in ("nA", "nSA") for s in SEEDS for sp in ("val", "test")}
fx = lambda v: " ".join(f"{x:+.2f}" for x in v)
nA_t, nSA_t = [B5["nA", s, "test"] for s in SEEDS], [B5["nSA", s, "test"] for s in SEEDS]
m_nA, m_nSA = st.mean(nA_t), st.mean(nSA_t)
print(f"seeds per arm: {len(SEEDS)} (five); beam-5 BLEU-4 on the official News test, best checkpoint  [c11-017 c11-039 c18-072]")
print(f"nA  (News real only) 5-seed mean test {m_nA:.2f} ({m_nA:.3f})  [c11-038 c14-104]")
print(f"nSA (stitched + pose augmentation) 5-seed mean test {m_nSA:.2f} ({m_nSA:.3f})  [c11-017 c11-038 c14-104 c11-040]")
d = [a - b for a, b in zip(nSA_t, nA_t)]; m, lo, hi = ci(d)
print(f"nSA - nA test per seed {fx(d)}  mean {m:+.2f} ({m:+.3f}) from {m_nA:.2f} to {m_nSA:.2f}  [c11-038]")
print(f"nSA - nA test 95% t-interval over the 5 seeds {lo:+.2f} to {hi:+.2f}; higher in {sum(x > 0 for x in d)}/5 seeds  [c11-039]")
PUB = {"SL-Transformer": 2.11, "GASLT": 2.72, "SL-Luong": 2.81}  # published values (cited literature, not this project's measurement)
print(f"nSA mean {m_nSA:.2f} is {m_nSA - PUB['SL-Transformer']:.2f} above published SL-Transformer {PUB['SL-Transformer']}  [c11-040 c14-104]")
print("published News baselines (cited): SL-Transformer 2.11, GASLT 2.72, SL-Luong 2.81; nSA mean " + f"{m_nSA:.2f} is {'below' if m_nSA < PUB['GASLT'] else 'above'} GASLT 2.72 and {'below' if m_nSA < PUB['SL-Luong'] else 'above'} SL-Luong 2.81  [c14-104]")

# recipe settings read from the two job scripts that trained the five seeds (Stage 1 job 7950 seeds 42/101/202; Stage 2 job 16398 seeds 303/404)
scripts = {"w4_news.sbatch (job 7950, seeds 42/101/202)": (ROOT / "jobs/slurm/w4_news.sbatch").read_text(),
           "w4_news_s2.sbatch (job 16398, seeds 303/404)": (ROOT / "jobs/slurm/w4_news_s2.sbatch").read_text()}
real, syn = None, None
for name, sb in scripts.items():
    n512 = {a: bool(re.search(rf"w4run {a} .*--max_src_frames 512", sb)) for a in ("nA", "nSA")}
    assert all(n512.values()), name
    print(f"read from jobs/slurm/{name}: --max_src_frames 512 in arm nA {n512['nA']} and arm nSA {n512['nSA']}  [c11-035]")
    m_ = re.search(r'"\$\(wc -l < "\$MF/news_sa_train.tsv"\)" = (\d+) \].*is not ([\d,]+) \+ ([\d,]+)"', sb)
    assert m_ and int(m_[1]) == int(m_[2].replace(",", "")) + int(m_[3].replace(",", "")), name
    assert re.search(r"nSA = News \+ first %s synthetic rows \+ pose-aug" % m_[3], sb) and re.search(r"w4run nSA .*\$AUG_DEF", sb), name
    assert not re.search(r"w4run nA .*AUG_DEF", sb), name
    r_, y_ = int(m_[2].replace(",", "")), int(m_[3].replace(",", ""))
    assert real in (None, r_) and syn in (None, y_); real, syn = r_, y_
    print(f"read from jobs/slurm/{name}: the nSA arm trains on news_sa_train.tsv, required to have {m_[1]} rows = {r_} real News rows + {y_:,} stitched rows, with pose augmentation (nA: real only)  [c11-037]")
print(f"both scripts agree: 5-seed comparison used --max_src_frames 512 in both arms (two scripts, seeds 42/101/202 and 303/404)  [c11-035]")
print(f"the recipe: {syn:,} stitched clips added to {real} real rows, ratio {syn / real:.1f}:1 ({syn / real:.3f}), plus pose augmentation, in all five nSA runs  [c11-037]")

# validation-selected length penalty (beam 10): LP chosen on validation per run, test scored at that LP
def at(name, lp): return sc(D["sweeps"][name][lp]["tot"])
tuned, bp0, bp1 = {}, {}, {}
for a in ("nA", "nSA"):
    for s in SEEDS:
        lp = max(LPS, key=lambda x: at(f"{a}_s{s}_val", x).score)
        tuned[a, s] = at(f"{a}_s{s}_test", lp).score
        bp0[a, s] = sc(D["runs"][f"{a}_s{s}_test"]["tot"]).bp; bp1[a, s] = at(f"{a}_s{s}_test", lp).bp
dec = [tuned["nSA", s] - B5["nSA", s, "test"] for s in SEEDS]; m, lo, hi = ci(dec)
mt = st.mean(tuned["nSA", s] for s in SEEDS)
print(f"nSA with beam 10 and the length penalty chosen on validation (five seeds): test mean {m_nSA:.2f} to {mt:.2f} ({mt:.3f}); difference {m:+.2f} ({m:+.3f}), 95% t-interval {lo:+.2f} to {hi:+.2f}; higher in {sum(x > 0 for x in dec)}/5  [c18-072 c18-073]")
print(f"nSA brevity penalty (13a), beam 5 / LP 0.6 to validation-selected beam 10: {st.mean(bp0['nSA', s] for s in SEEDS):.2f} to {st.mean(bp1['nSA', s] for s in SEEDS):.2f}  [c18-074]")

# signable-word recall, parsed from the saved job output
rec = {}
for l in open(HERE / "data/signable_recall.txt"):
    m_ = re.match(r"(nA|nSA)\s+s(\d+): signable ([\d.]+)% \| unsignable ([\d.]+)%", l)
    if m_: rec[m_[1], int(m_[2])] = (float(m_[3]), float(m_[4]))
assert len(rec) == 10
for i, lab in ((0, "words that the synthetic data can sign (signable)"), (1, "words that it cannot (unsignable)")):
    dd = [rec["nSA", s][i] - rec["nA", s][i] for s in SEEDS]; m, lo, hi = ci(dd)
    print(f"read from the saved job output signable_recall.txt (job 16398 secondary): nSA - nA recall, {lab}: {m:+.2f} percentage points [{lo:+.2f}, {hi:+.2f}], same sign as the mean in {sum((x > 0) == (m > 0) for x in dd)}/5 seeds  [c18-075 c18-076]")

# ensemble futility gate (job 21744)
ens = {lp: sc(v["tot"]).score for lp, v in D["sweeps"]["ens5_val"].items()}
lpb = max(ens, key=ens.get); eb = ens[lpb]
single = max(sc(D["sweeps"][f"nSA_s{s}_val"][lp]["tot"]).score for s in SEEDS for lp in LPS)
gate_txt = open(HERE / "data/ens_phase1.txt").read()
thr = re.search(r"best single \+ ([\d.]+)", gate_txt)[1]
logs = open(HERE / "data/ens5_val.log").read()
nmem = int(re.search(r"\[mbr\] (\d+) members", logs)[1]); nchk = int(re.search(r"matches all (\d+) checkpoints", logs)[1])
print(f"read from the saved job output ens5_val.log (job 21744): the ensemble decode loaded {nmem} members and checked {nchk} checkpoints; its five single-member validation decodes are the {len(SEEDS)} nSA seeds {SEEDS}  [c18-098]")
assert nmem == nchk == len(SEEDS) == 5
print(f"read from the saved job output ens_phase1.txt: pre-registered validation gate = best single +{thr}  [c18-098]")
print(f"ensemble validation {eb:.3f} (LP {lpb}) against {single:.3f} for the best single model: difference {eb - single:+.3f} ({'below' if eb - single < float(thr) else 'above'} the {thr} gate)  [c18-099]")

# c04-024: share of News clips above the 256-frame input cap, from the per-clip frame lengths of the News alignment table
import csv
fr = [int(r["frames"]) for r in csv.DictReader(open(HERE / "data/news_clip_frames.csv"))]
cap = int(re.search(r"max_src_frames: int = (\d+)", (ROOT / "src/config.py").read_text())[1])
over = sum(f > cap for f in fr)
print(f"News clips: {len(fr):,}; input cap max_src_frames (config default) {cap}; clips longer than {cap} frames {over:,} = {100 * over / len(fr):.1f}% (about a third); median {st.median(fr):.0f} frames  [c04-024]")

# c20-001: training-set size of the joint Communication + News run (job 1140711), reconstructed from its two component counts
comm = json.load(open(HERE / "data/real_train_count.json"))["clips"]
print(f"joint run training clips: {comm:,} Communication (lines of the saved real training list of job 21734) + {real:,} News (checked by the job scripts above) = {comm + real:,}  [c20-001]")
