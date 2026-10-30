"""Recompute the 'published systems run with their own code' numbers (wave 12l3 + the five-seed extension).

Run: cd src && python3 ../results/checks/published_systems_own_code/check.py
Needs the reference sentences (make data): they sit in the saved outputs under "ref"/"refs".
"""
import json, math, sys, warnings
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats
import sacrebleu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

warnings.filterwarnings("ignore")
D = HERE / "data"
S3 = (42, 101, 202)
PUBLISHED = {"slt": 9.20, "gaslt": 9.95}  # literature values (Auslan-Daily benchmark paper, Table 3): inputs, not outputs
TOL = 1.0  # pre-registered reproduction tolerance
NAME = {"slt": "SL-Transformer", "gaslt": "GASLT"}
bad = []


def project(h, r):  # this project's scoring: detokenise + sacreBLEU 13a
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r])


def authors(h, r):  # the authors' script: sacreBLEU on the raw tokenised text, tokenize='none'
    return sacrebleu.corpus_bleu(h, [r], tokenize="none", force=True).score


def load_run(folder):
    rows = [json.loads(l) for l in open(folder / "dump.jsonl")]
    assert len(rows) == 800
    h, r = [x["hyp_beam"] for x in rows], [x["ref"] for x in rows]
    sc = json.load(open(folder / "scores.json"))
    rec = sc["splits"]["test"]
    p, a = project(h, r), authors(h, r)
    if abs(p - rec["project"]["bleu4"]) > 1e-6 or abs(a - rec["authors"]["bleu4"]) > 1e-6:
        bad.append(f"{folder.name}: recomputed {p:.6f}/{a:.6f} != recorded {rec['project']['bleu4']:.6f}/{rec['authors']['bleu4']:.6f}")
    return p, a, r, sc


def load_edw(f):
    j = json.load(open(f))
    p = project(j["beam_hyps"], j["refs"])
    if abs(p - j["recorded_bleu4_13a"]) > 1e-6:
        bad.append(f"{f.name}: recomputed {p:.6f} != recorded {j['recorded_bleu4_13a']:.6f}")
    return p, j["refs"]


def mean_ci(x):  # 95% t-interval of the mean over seeds
    x = np.asarray(x); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - h, m + h


def diff_ci(a, b):  # difference of means, pooled-variance t-interval
    a, b = np.asarray(a), np.asarray(b); n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = a.mean() - b.mean(); h = stats.t.ppf(.975, n1 + n2 - 2) * sp * math.sqrt(1 / n1 + 1 / n2)
    return d, d - h, d + h


def sg(x): return f"{x:+.2f}"
def iv(lo, hi): return f"[{lo:.2f}, {hi:.2f}]"
def siv(lo, hi): return f"[{sg(lo)}, {sg(hi)}]"


cells = [f"{m}__{d}" for m in ("slt", "gaslt") for d in ("ap56_raw", "ap56_frame", "ap53_frame", "dw53_frame")]
T, A, REFS, EP = {}, {}, None, set()
for c in cells:
    T[c], A[c] = [], []
    for s in S3:
        p, a, r, sc = load_run(D / "l3" / f"{c}_s{s}")
        REFS = REFS or r
        assert r == REFS
        T[c].append(p); A[c].append(a); EP.add((sc["epochs_started"], sc["stop"]))
E3 = []
for s in S3:
    p, r = load_edw(D / "l3" / f"Edw_s{s}.json"); assert r == REFS; E3.append(p)
# five-seed extension: seeds 42/101/202 from the earlier job, 303/404 from the extension job
T5 = {m: [load_run(D / "l2" / f"{m}__dw53_frame_s{s}")[0] for s in S3]
         + [load_run(D / "ext" / f"{m}__dw53_frame_s{s}")[0] for s in (303, 404)] for m in ("slt", "gaslt")}
E5 = [load_edw(D / "l2" / f"Edw_s{s}.json")[0] for s in S3] + [load_edw(D / "ext" / f"Edw_s{s}.json")[0] for s in (303, 404)]

if bad:
    print("GATE FAILED (recomputed scores differ from the values recorded by the jobs):")
    print("\n".join(bad)); sys.exit(1)
print("Gate passed: all 24 authors'-code runs (and the 10 further DWPose runs of the five-seed comparison) re-score to the")
print("  recorded project and authors' BLEU-4 (1e-6); the 3 + 3 + 2 translator control runs re-score to their recorded values;")
print("  references identical across the 24 runs (800 clips).")
print(f"Training length of every authors'-code run (8 cells x 3 seeds): {sorted(EP)}")
print("Seeds per cell: 3 (42, 101, 202)")

print("\nBlock 1: the authors' own code on the signer's AlphaPose keypoints (test BLEU-4, authors' script, 3-seed mean, 95% t-interval)")
for m in ("slt", "gaslt"):
    for c in (f"{m}__ap56_raw", f"{m}__ap56_frame"):
        mu, lo, hi = mean_ci(A[c])
        pub = PUBLISHED[m]
        inside = (lo >= pub - TOL) and (hi <= pub + TOL)
        print(f"  {NAME[m]:15s} {c.split('__')[1]:11s} {mu:.2f}  {iv(lo, hi)}  published {pub:.2f}; |diff| {abs(mu-pub):.2f} <= {TOL}: {abs(mu-pub) <= TOL}; interval inside the band: {inside}"
              + ("" if inside else "  (interval crosses the tolerance edge: consistent with, not shown)"))
for m in ("slt", "gaslt"):
    print(f"  project scoring function, paper-literal cell (ap56_raw), {NAME[m]}: {np.mean(T[m+'__ap56_raw']):.2f}")

print("\nBlock 2: authors' code on 53 DWPose points minus this project's translator (40 epochs, beam 5), test BLEU-4, project function")
for m in ("slt", "gaslt"):
    arm = T[f"{m}__dw53_frame"]
    d, lo, hi = diff_ci(arm, E3)
    print(f"  {NAME[m]:15s} 3 seeds ({np.mean(arm):.2f} against {np.mean(E3):.2f}): {sg(d)}  {siv(lo, hi)}")
for m in ("slt", "gaslt"):
    arm = T5[m]
    d, lo, hi = diff_ci(arm, E5)
    print(f"  {NAME[m]:15s} 5 seeds ({np.mean(arm):.2f} against {np.mean(E5):.2f}): {sg(d)}  {siv(lo, hi)}")
cs = [diff_ci(T[f"{m}__dw53_frame"], T[f"{m}__ap53_frame"]) for m in ("slt", "gaslt")]
print("  Authors' code only: DWPose minus the signer's AlphaPose cut to the same 53 points (3 seeds, project function):")
print(f"    SL-Transformer {sg(cs[0][0])} {siv(cs[0][1], cs[0][2])};  GASLT {sg(cs[1][0])} {siv(cs[1][1], cs[1][2])}")
