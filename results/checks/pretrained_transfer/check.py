#!/usr/bin/env python3
"""Recompute the scores of the fine-tuned pre-trained sign-language model (How2Sign checkpoint) on Auslan-Daily from saved decodes.

Run:  cd src && python3 ../results/checks/<this folder>/check.py

Inputs: data/{arm}_s{seed}_{val,test}.json = beam-5 hypotheses of 3 arms x 5 seeds x 2 splits (hypotheses only; references come from
comm_run/manifests/{train,val,test}.tsv, written by `make data`).  Arms: ctrl_small (our small translator from scratch, real clips only),
u_scratch (the same network as u_asl with random pose-side weights), u_asl (the fine-tuned checkpoint).  Seeds 42/101/202 = first job,
303/404 = second-stage job.
Gate: every decode must re-score (sacreBLEU 13a on detokenised text, and slt.metrics) to the BLEU-4 and the distinct-output count recorded by
the job that wrote it; the check exits non-zero otherwise.
Statistics follow the original read-out: seed-level two-sided 95% t-interval (df = 4) of paired per-seed differences; length-adjusted
difference = u_asl BLEU x (ctrl brevity penalty / u_asl brevity penalty) - ctrl BLEU; between-video permutation test of the per-clip lift
(first three seeds, 2,000 permutations, python random seed 0).
"""
import json, math, random, re, statistics, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sacrebleu
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402
from arm_readouts import readouts  # noqa: E402
from stitch_lexicon import STOP  # noqa: E402

D = HERE / "data"
MAN = ROOT / "comm_run" / "manifests"
SEEDS = [42, 101, 202, 303, 404]
ARMS = ["ctrl_small", "u_scratch", "u_asl"]


def fail(msg):
    print("GATE FAILED:", msg)
    sys.exit(1)


def read_tsv(name):
    p = MAN / f"{name}.tsv"
    if not p.exists():
        fail(f"{p} missing: run `make data` first")
    return [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if "\t" in l]


ROWS = {sp: read_tsv(sp) for sp in ("train", "val", "test")}
REFS = {sp: [r[1] for r in ROWS[sp]] for sp in ROWS}
norm = lambda t: " ".join(t.lower().split())
TRAIN = {norm(t) for t in REFS["train"]}
N = {"val": 800, "test": 800}

R = {}
for sp in ("val", "test"):
    if len(REFS[sp]) != N[sp]:
        fail(f"{sp} manifest has {len(REFS[sp])} rows")
    refs = [detokenize(r) for r in REFS[sp]]
    for arm in ARMS:
        for s in SEEDS:
            j = json.load(open(D / f"{arm}_s{s}_{sp}.json"))
            h = j["hyps"]
            assert len(h) == N[sp]
            H = [detokenize(x) for x in h]
            b = sacrebleu.corpus_bleu(H, [refs], tokenize="13a")
            if abs(b.score - corpus_bleu4(H, refs)) > 1e-9 or abs(b.score - j["recorded_bleu4"]) > 1e-6 or len(set(h)) != j["recorded_unique"]:
                fail(f"{arm} s{s} {sp}: recomputed {b.score} / recorded {j['recorded_bleu4']}, distinct {len(set(h))} / {j['recorded_unique']}")
            ro = readouts(h, REFS[sp], STOP, train=TRAIN)
            R[sp, arm, s] = dict(bleu=b.score, bp=b.bp, distinct=len(set(h)), unseen=ro["bleu4_unseen"], unseen_n=ro["unseen_n"], hyps=h)
print("GATE: all 30 saved decodes re-score to the BLEU-4 and distinct-output count recorded by their jobs (tolerance 1e-6)")

T4 = stats.t.ppf(0.975, 4)


def tci(d):
    d = np.array(d, dtype=float)
    m = d.mean()
    h = T4 * d.std(ddof=1) / math.sqrt(len(d))
    return m, m - h, m + h


def mean(sp, arm, key):
    return float(np.mean([R[sp, arm, s][key] for s in SEEDS]))


def diff(sp, x, y, key="bleu"):
    return [R[sp, x, s][key] - R[sp, y, s][key] for s in SEEDS]


f2 = lambda v: f"{v:.2f}"
sg = lambda v: f"{v:+.2f}"
UN = R['test', 'u_asl', 42]['unseen_n']
PUB = 9.95  # published best pose baseline on Auslan-Daily Communication (cited value, not this project's measurement)

print(f"\n== Five-seed mean beam-5 BLEU-4 (seeds {SEEDS}); unseen = {UN} test clips whose reference is absent from training ==")
print(f"{'arm':12s} {'validation':>10s} {'test':>8s} {'unseen (test)':>14s}")
TAG = {"ctrl_small": "c11-046 c21-002 c21-012 c19-020 c11-048", "u_scratch": "c11-049 c19-021", "u_asl": "c01-022 c03-008 c11-014 c11-045 c21-006 c21-012 c21-005 c13-055 c19-019 c11-048"}
for arm in ARMS:
    print(f"{arm:12s} val {f2(mean('val', arm, 'bleu'))} test {f2(mean('test', arm, 'bleu'))} unseen {f2(mean('test', arm, 'unseen'))} over {len(SEEDS)} seeds  [{TAG[arm]}]")
print(f"{UN} unseen-target test clips  [c11-048 c19-047]")
print("per-seed test BLEU-4 of the fine-tuned arm u_asl:", " ".join(f2(R['test', 'u_asl', s]['bleu']) for s in SEEDS))
print(f"BLEU-4: fine-tuned test mean {f2(mean('test', 'u_asl', 'bleu'))} (five seeds) against published best {PUB} (published value, cited)  [c03-008 c11-045]")
print(f"from scratch test mean {f2(mean('test', 'ctrl_small', 'bleu'))}; random pose-side weights {f2(mean('test', 'u_scratch', 'bleu'))}  [c21-012 c11-046]")

print("\n== Paired comparisons, seed-level 95% t-interval (df = 4) ==")
print("95% t-interval over 5 seeds (df = 4)  [c03-010 c11-047]")
TG = {("test", "u_asl", "ctrl_small"): "c03-010 c11-047 c19-029", ("val", "u_asl", "ctrl_small"): "c19-029",
      ("test", "u_asl", "u_scratch"): "c19-031", ("val", "u_asl", "u_scratch"): "c19-031"}
for sp in ("test", "val"):
    for x, y, lab in (("u_asl", "ctrl_small", "fine-tuned minus from scratch"), ("u_asl", "u_scratch", "fine-tuned minus random pose-side weights")):
        d = diff(sp, x, y)
        m, lo, hi = tci(d)
        print(f"{sp:4s} {lab}: {sg(m)} [{sg(lo)}, {sg(hi)}]  positive in {sum(v > 0 for v in d)} of {len(d)} seeds  [{TG[sp, x, y]} c19-030 c11-047]")
d = [R["test", "u_asl", s]["bleu"] * R["test", "ctrl_small", s]["bp"] / R["test", "u_asl", s]["bp"] - R["test", "ctrl_small", s]["bleu"] for s in SEEDS]
m, lo, hi = tci(d)
print(f"test fine-tuned minus from scratch, adjusted for output length (brevity penalty): {sg(m)} [{sg(lo)}, {sg(hi)}] (unrounded {m:.4f} [{lo:.4f}, {hi:.4f}])  [c19-030]")

print("\n== Distinct outputs (of 800 clips) ==")
for sp in ("val", "test"):
    for arm in ("u_scratch", "u_asl"):
        print(f"{sp:4s} {arm:10s}", " ".join(f"s{s}:{R[sp, arm, s]['distinct']}" for s in SEEDS))
low = [(s, R["val", "u_scratch", s]["distinct"]) for s in SEEDS if R["val", "u_scratch", s]["distinct"] < 100]
print(f"random-weight runs with fewer than 100 distinct validation outputs (the collapse guard of 100): {len(low)} -> " + ", ".join(f"seed {s} has {n} distinct outputs" for s, n in low) + "  [c11-049 c19-032]")

# between-video permutation test of the per-clip lift (u_asl - ctrl_small), seeds 42/101/202, all 800 test clips
vid = [re.search(r"(video_\d+)_\d+", r[0]).group(1) for r in ROWS["test"]]
test = REFS["test"]
sb = lambda h, r: sacrebleu.sentence_bleu(detokenize(h), [detokenize(r)], tokenize="13a").score
s3 = [42, 101, 202]
lift = [statistics.mean(sb(R["test", "u_asl", k]["hyps"][i], test[i]) - sb(R["test", "ctrl_small", k]["hyps"][i], test[i]) for k in s3) for i in range(800)]


def stat(idx, labels):
    g = defaultdict(list)
    for i in idx:
        g[labels[i]].append(lift[i])
    ms = [statistics.mean(v) for v in g.values() if len(v) >= 3]
    return statistics.pvariance(ms), len(ms)


random.seed(0)
idx = list(range(800))
obs, nv = stat(idx, vid)
labs = [vid[i] for i in idx]
null = []
for _ in range(2000):
    random.shuffle(labs)
    null.append(stat(idx, dict(zip(idx, labs)))[0])
p = sum(x >= obs for x in null) / 2000
print(f"\n== Episode specificity of the per-clip gain (3-seed mean sentence-BLEU lift, {nv} videos with at least 3 clips) ==")
print(f"between-video permutation test, 2,000 permutations: p = {p:.3f}  [c19-049]")

# ensemble of the five fine-tuned models (clean-code decoding, beam 10, length penalty 0.8)
E = json.load(open(D / "ensemble_Estar_test.json"))
He = [detokenize(x) for x in E["hyps"]]
Rt = [detokenize(r) for r in REFS["test"]]
be = corpus_bleu4(He, Rt)
print(f"\nGATE: ensemble re-scores to the BLEU-4 recorded by its job ({E['recorded_bleu4']:.4f})")
if abs(be - E["recorded_bleu4"]) > 1e-6 or len(E["hyps"]) != 800:
    fail(f"ensemble {be} vs recorded {E['recorded_bleu4']}")
print(f"five-model ensemble (beam {E['beam']}, length penalty {E['length_penalty']}) test BLEU-4 {f2(be)} ({be:.3f})  [c13-055 c21-005 c21-006]")

# How2Sign port parity gate, read from the saved job output
G = json.load(open(D / "gate_how2sign_port.json"))
print(f"\nHow2Sign port parity (read from the saved job output gate_how2sign_port.json, {G['n']} clips, beam {G['beams']}): port {f2(G['bleu4_13a'])} ({G['bleu4_13a']:.3f}), released output {f2(G['released_bleu4_13a_same_clips'])}  [c19-009]")
print(f"GATE target {G['target']} +- {G['tol']}: " + ("passed" if abs(G['bleu4_13a'] - G['target']) <= G['tol'] else "failed"))
