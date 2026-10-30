#!/usr/bin/env python3
"""Wave 6 (jobs 16435 + 25863) decision under results/PREREG_WAVE6_CORPUS_2026-09-23.md section 4.

Run from src:  python ../results/wider_corpus_joint_training/analyse_wave6.py > ../results/wider_corpus_joint_training/decision.txt
Gates first (every recorded Stage-1 value, every stored BLEU re-scored through slt/metrics.py), then the decision
through the project's own tools (holm_decision.py --family, seed_decision.py, arm_readouts.py).
"""
import json, os, statistics as st, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PB = HERE.parents[1] / "src"
sys.path[:0] = [str(PB), str(PB / "tools")]
from slt.metrics import corpus_bleu4, detokenize          # noqa: E402
from arm_readouts import readouts                          # noqa: E402
from seed_decision import interval                         # noqa: E402  (df = n - 1 t table)
from stitch_lexicon import STOP                            # noqa: E402

RUNS, KEY = HERE / "runs", "tokenize=13a;detok=True"
TRAIN = HERE.parent / "real_data_learning_curve" / "train.tsv"
S1, S2 = [42, 101, 202], [303, 404]
ARMS = {"A": S1 + S2, "B": S1 + S2, "C": S1}


def load(arm, seed, split, ck="best"):
    p = RUNS / f"{arm}_s{seed}" / f"beam5_{split}_{ck}.json"
    d = json.load(open(p))
    assert d["beam"] == 5 and d["length_penalty"] == 0.6 and d["beam_result"]["n"] == 800, p
    assert d["manifest"].endswith(f"/{split}.tsv"), (p, d["manifest"])              # val/test mix-up guard
    assert d["checkpoint"].endswith(f"/{arm}_s{seed}/{'best' if ck == 'best' else 'soup_last5'}.pt"), p
    got = corpus_bleu4([detokenize(h) for h in d["beam_hyps"]], [detokenize(r) for r in d["refs"]], "13a")
    assert abs(got - d["beam_result"]["metrics"][KEY]) < 1e-6, (p, got)            # stored number = re-score
    return d


J = {(a, s, sp, ck): load(a, s, sp, ck) for a, ss in ARMS.items() for s in ss
     for sp in ("val", "test") for ck in ("best", "soup_last5")}
B5 = {k: v["beam_result"]["metrics"][KEY] for k, v in J.items()}
for sp in ("val", "test"):                                                          # same refs, same order
    refs = {tuple(v["refs"]) for k, v in J.items() if k[2] == sp}
    assert len(refs) == 1, sp
assert J[("A", 42, "val", "best")]["refs"] != J[("A", 42, "test", "best")]["refs"]
test_refs = J[("A", 42, "test", "best")]["refs"]
local_test = [l.rstrip("\n").split("\t", 1)[1] for l in open(TRAIN.parent / "test.tsv") if "\t" in l]
assert test_refs == local_test, "test refs differ from the local official test manifest"

# ---------------------------------------------------------------- GATE: recorded Stage-1 values (prereg + task)
rec_abs = {("A", 42): (11.307, 11.711), ("A", 101): (11.939, 11.366), ("A", 202): (11.291, 10.839),
           ("B", 42): (11.446, 11.672), ("B", 101): (11.314, 11.236)}
for (a, s), (v, t) in rec_abs.items():
    assert abs(B5[(a, s, "val", "best")] - v) < 5e-4 and abs(B5[(a, s, "test", "best")] - t) < 5e-4, (a, s)


def d(arm, seed, sp, ck="best"):
    return B5[(arm, seed, sp, ck)] - B5[("A", seed, sp, ck)]


rec_delta = {("B", "val"): [0.138, -0.625, 0.516], ("B", "test"): [-0.039, -0.131, 0.811],
             ("C", "val"): [-0.159, -1.325], ("C", "test"): [-1.588, -1.275]}
for (arm, sp), want in rec_delta.items():
    got = [d(arm, s, sp) for s in S1[:len(want)]]
    assert all(abs(g - w) < 1.5e-3 for g, w in zip(got, want)), (arm, sp, got, want)
assert abs(st.mean(d("B", s, "val") for s in S1) - 0.010) < 5e-4
assert abs(st.mean(d("B", s, "test") for s in S1) - 0.214) < 5e-4
print("GATE PASSED: all 26 criterion + 26 soup JSONs re-score exactly through slt/metrics.py (13a, detok);")
print("  val/test manifests, beam 5, LP 0.6, n 800, checkpoint paths as expected; test refs == local test.tsv;")
print("  every recorded Stage-1 value reproduces: A s42 11.307/11.711, s101 11.939/11.366, s202 11.291/10.839;")
print("  B s42 11.446/11.672, s101 11.314/11.236; B-A val +0.138/-0.625/+0.516 (mean +0.010),")
print("  test -0.039/-0.131/+0.811 (mean +0.214); C-A val -0.159/-1.325, test -1.588/-1.275.\n")

print("Absolute beam-5 of best.pt (val / test)")
for arm, ss in ARMS.items():
    print(f"  {arm}  " + "  ".join(f"s{s} {B5[(arm, s, 'val', 'best')]:.3f}/{B5[(arm, s, 'test', 'best')]:.3f}"
                                  for s in ss))

# ---------------------------------------------------------------- per-seed summaries + family for the tools
for s in S1 + S2:
    json.dump({a: {"beam5_val": B5[(a, s, "val", "best")], "beam5_test": B5[(a, s, "test", "best")]}
               for a in ARMS if s in ARMS[a]}, open(HERE / f"summary_s{s}.json", "w"), indent=1)
fam = {"alpha": 0.05, "arms": {a: {"val": [d(a, s, "val") for s in ARMS[a]],
                                   "test": [d(a, s, "test") for s in ARMS[a]], "n_stage1": 3} for a in ("B", "C")}}
json.dump(fam, open(HERE / "family.json", "w"), indent=1)

cvm = st.mean(fam["arms"]["C"]["val"])
print(f"\nC Stage-1 futility: C_s202 val delta {d('C', 202, 'val'):+.3f} (needed > +1.484); "
      f"3-seed mean {cvm:+.3f} -> {'STOP' if cvm <= 0 else 'CONTINUE'}")


def run(*args):
    out = subprocess.run([sys.executable, *args], cwd=PB, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    print(f"\n$ python {' '.join(args)}\n{out.stdout.rstrip()}")


sums = [str(HERE / f"summary_s{s}.json") for s in S1 + S2]
run("tools/seed_decision.py", "--control", "A", "--arm", "B", *sums)
run("tools/seed_decision.py", "--control", "A", "--arm", "C", *sums[:3])
run("tools/holm_decision.py", "--family", str(HERE / "family.json"))

# ---------------------------------------------------------------- mandated readouts (R2 rule 7), paired B - A
run("tools/arm_readouts.py", *[str(RUNS / f"{a}_s{s}") for a in ARMS for s in ARMS[a]],
    "--split", "test", "--ckpt", "best", "--train_manifest", str(TRAIN))
train = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split()) for l in open(TRAIN) if "\t" in l}
R = {(a, s): readouts(J[(a, s, "test", "best")]["beam_hyps"], test_refs, STOP, train)
     for a in ARMS for s in ARMS[a]}
assert R[("A", 42)]["unseen_n"] == 570, R[("A", 42)]["unseen_n"]
print("\nPaired readout deltas on beam-5 TEST of best.pt (seed-level 95% t, df = n-1; descriptive, not criteria)")
for arm in ("B", "C"):
    for k in ("bleu4_unseen", "content_precision", "content_recall", "wer", "len_ratio", "distinct"):
        dd = [R[(arm, s)][k] - R[("A", s)][k] for s in ARMS[arm]]
        m, lo, hi = interval(dd)
        print(f"  {arm}-A {k:18s} " + " ".join(f"{x:+8.3f}" for x in dd)
              + f" | +{sum(x > 0 for x in dd)}/{len(dd)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# ---------------------------------------------------------------- soups: recorded, NEVER a criterion
print("\nLast-5 soup, beam-5 (NOT a criterion in any case; recorded so it cannot be harvested later)")
for arm in ("B", "C"):
    for sp in ("val", "test"):
        dd = [d(arm, s, sp, "soup_last5") for s in ARMS[arm]]
        m, lo, hi = interval(dd)
        print(f"  {arm}-A soup {sp:4s} " + " ".join(f"{x:+.3f}" for x in dd)
              + f" | +{sum(x > 0 for x in dd)}/{len(dd)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# ---------------------------------------------------------------- prereg section 5 (hypothesis 1), descriptive
# The prereg names the hypothesis (the ~900 words absent from AD training text take mass from needed words) but
# specifies no test; its named follow-up is the AD-vocabulary-restricted corpus. This only asks whether B EMITS
# those words on test -- it cannot see mass diverted without being emitted.
vocab = {w for l in open(TRAIN) if "\t" in l for w in l.rstrip("\n").split("\t", 1)[1].lower().split()}
print(f"\nSection-5 descriptive check: hypothesis tokens outside the AD real-train vocabulary ({len(vocab)} types)")
for arm in ARMS:
    for s in ARMS[arm]:
        toks = [w for h in J[(arm, s, "test", "best")]["beam_hyps"] for w in h.lower().split()]
        oov = [w for w in toks if w not in vocab]
        print(f"  {arm}_s{s}: {len(oov)}/{len(toks)} tokens ({100 * len(oov) / len(toks):.2f}%), "
              f"{len(set(oov))} types {sorted(set(oov))[:8]}")

# ---------------------------------------------------------------- verdict, under the rule as written
bv, bt = fam["arms"]["B"]["val"], fam["arms"]["B"]["test"]
print(f"""
RULE (PREREG_WAVE6 section 4, quoted): "Confirmatory family {{B, C}}, Holm m = 2 at alpha = 0.05: adopt only if the
one-sided 5-seed paired t on beam-5 test deltas rejects at its Holm level AND the 5-seed mean beam-5 validation delta
> 0 AND both fresh seeds are positive on test. Stopped arms enter with p = 1."
VERDICT: C STOPPED at Stage 1 (3-seed val mean {cvm:+.3f} <= 0; p = 1). B NOT ADOPTED: fails all three conditions --
  one-sided p 0.331 > Holm level 0.025; 5-seed val mean {st.mean(bv):+.3f} <= 0; fresh seeds test {bt[3]:+.3f} / {bt[4]:+.3f}.
  The wave-5 full-dictionary corpus does not beat the adopted corpus in the joint 1.1:1 regime; doubling it to 2.2:1
  (C) is worse on every seed and every criterion metric. Nothing adopted; the adopted recipe stands.""")
