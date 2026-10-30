"""Claims check, wave 12a: News stitched data vs pose augmentation (nA, nS, nSA, nRA; seeds 42/101/202).
Run: cd src && python3 ../results/checks/news_stitching_vs_augmentation/check.py"""
import json, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from sacrebleu.metrics import BLEU

D, SEEDS, T975 = HERE / "data", (42, 101, 202), 4.303      # two-sided 95% t, df = 2 (as tools/seed_decision.py)
ARMS = ("nA", "nS", "nSA", "nRA")
bleu = BLEU()

def load(name):
    return json.load(open(D / f"{name}.json"))

def score(d):
    tot = [sum(s[i] for s in d["stats"]) for i in range(10)]
    return bleu._compute_score_from_stats(tot).score

X, JOB = {}, {}
for a in ARMS:
    for s in SEEDS:
        for sp in ("val", "test"):
            d = load(f"{a}_s{s}_{sp}")
            b = score(d)
            if abs(b - d["recorded_bleu"]) > 1e-6:                        # gate: the job's own recorded beam-5 BLEU-4
                sys.exit(f"GATE FAILED {a} s{s} {sp}: {b} vs recorded {d['recorded_bleu']}")
            X[a, s, sp] = b; JOB[a, s] = d["job"]
print("gate: all 24 beam-5 BLEU-4 (13a, detokenised) recomputed from per-sentence n-gram counts equal the values recorded by the jobs")

# controls from the earlier job (138377) vs the augmentation-alone arm (172449): is the control reproducible?
for sp in ("val", "test"):
    p, c = load(f"probe_nA_s42_{sp}"), load(f"nA_s42_{sp}")
    same = sum(x == y for x, y in zip(p["hyps"], c["hyps"]))
    print(f"control nA seed 42 re-run in job {p['job']}: {same}/{len(c['hyps'])} identical hypotheses on {sp}, BLEU {p['recorded_bleu']:.5f} vs {c['recorded_bleu']:.5f}")
    assert same == len(c["hyps"]) and len(p["hyps"]) == len(c["hyps"])
print("nRA jobs:", ", ".join(f"s{s} {JOB['nRA', s]}" for s in SEEDS), "| controls nA jobs:", ", ".join(f"s{s} {JOB['nA', s]}" for s in SEEDS))
print("nRA seeds 101 and 202 are compared with controls from an earlier job submission (138377); seed 42's control was reproduced above")
print("seeds per arm:", {a: len([s for s in SEEDS if (a, s, 'test') in X]) for a in ARMS})

for a in ARMS:
    print(f"{a:4s} beam-5 BLEU-4 mean  val {st.mean(X[a,s,'val'] for s in SEEDS):.2f}  test {st.mean(X[a,s,'test'] for s in SEEDS):.2f}")

def contrast(name, plus, minus):
    for sp in ("test", "val"):
        d = [X[plus, s, sp] - X[minus, s, sp] for s in SEEDS]
        m = st.mean(d); h = T975 * st.stdev(d) / len(d) ** 0.5
        print(f"{name} = {plus} - {minus}  {sp}: per seed " + " ".join(f"{x:+.2f}" for x in d)
              + f"  positive {sum(x > 0 for x in d)}/3  mean {m:+.2f} ({m:+.3f}) [{m-h:+.2f}, {m+h:+.2f}]")
contrast("both together (B)", "nSA", "nA")
contrast("stitched alone (S)", "nS", "nA")
contrast("augmentation added (A)", "nSA", "nS")
contrast("augmentation alone (R)", "nRA", "nA")
