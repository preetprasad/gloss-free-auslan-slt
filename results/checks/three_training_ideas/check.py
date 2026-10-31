"""Recompute the numbers of the 'three training ideas' row (appendix B) from saved beam-5 decodes.

Run:  cd src && python3 ../results/checks/three_training_ideas/check.py
Data: data/<arm>_s<seed>_<split>.jsonl  ({"id", "hyp", "ref"}; refs restored by `make data`), data/runs_meta.json,
      data/g1_alignment_summary.json (the only number read from a saved job summary: needs checkpoints to recompute).
"""
import json, sys, warnings, logging
from pathlib import Path
from collections import Counter
from statistics import mean, stdev

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402
from stitch_lexicon import STOP  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

DATA = HERE / "data"
META = json.load(open(DATA / "runs_meta.json"))
S1, S2 = (42, 101, 202), (303, 404)
KEY_RECORDED = ("A_s42", "test", 11.715239196360335)  # value recorded by the original job's own read-out


def load(arm, seed, split):
    rows = [json.loads(l) for l in open(DATA / f"{arm}_s{seed}_{split}.jsonl")]
    return [r["hyp"] for r in rows], [r["ref"] for r in rows]


def cwords(s):
    return [w for w in s.lower().split() if w.isalpha() and w not in STOP]


def score(hyps, refs):
    bleu = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs], "13a")
    hit = nr = 0
    for h, r in zip(hyps, refs):
        ch, cr = Counter(cwords(h)), Counter(cwords(r))
        hit += sum(min(c, cr[w]) for w, c in ch.items())
        nr += sum(cr.values())
    return {"bleu": bleu, "recall": 100.0 * hit / nr,
            "lenr": sum(len(detokenize(h).split()) for h in hyps) / sum(len(detokenize(r).split()) for r in refs)}


def interval(d):
    m = mean(d)
    h = tdist.ppf(0.975, len(d) - 1) * stdev(d) / len(d) ** 0.5
    return m, m - h, m + h


def f2(x):
    return f"{x:+.2f}"


def fmt(d, nd=2):
    m, lo, hi = interval(d)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ---- score every saved decode, gate on the recorded values --------------------------------------------------
R = {}
bad = []
for name, m in META.items():
    arm, seed = name.split("_s")
    for sp in ("val", "test"):
        h, r = load(arm, int(seed), sp)
        sc = score(h, r)
        R[arm, int(seed), sp] = sc
        if sc["bleu"] != m[f"stored_{sp}"]:
            bad.append(f"{name} {sp}: {sc['bleu']!r} != stored {m[f'stored_{sp}']!r}")
g = R["A", 42, "test"]["bleu"]
if bad or g != KEY_RECORDED[2]:
    print("GATE FAILED (run `make data` first?):", bad[:3], g)
    sys.exit(1)
print(f"GATE: all {len(META) * 2} decodes re-score to the BLEU-4 stored by their jobs; "
      f"A seed 42 test = {g:.6f} (recorded {KEY_RECORDED[2]:.6f})")

# ---- set-up claims ---------------------------------------------------------------------------------------------
jobs1 = {n: m["job"] for n, m in META.items() if m["stage"] == 1}
print(f"\nSetup: {len(jobs1)} Stage-1 runs (5 arms x 3 seeds) all marked done by job {sorted(set(jobs1.values()))}; "
      f"E1/ST/P seeds {sorted({int(n.split('_s')[1]) for n in jobs1 if n.startswith(('E1', 'ST', 'P_'))})} -> three seeds each, one job")
print("Epochs: P (preference stage) " + "/".join(str(META[f'P_s{s}']['epochs']) for s in S1)
      + " ; AC (continuation control) " + "/".join(str(META[f'AC_s{s}']['epochs']) for s in S1)
      + " -> ten-epoch stage and ten-epoch continuation")


def contrast(arm, ctrl, seeds, split, key="bleu"):
    return [R[arm, s, split][key] - R[ctrl, s, split][key] for s in seeds]


# ---- Stage 1 (three seeds) -------------------------------------------------------------------------------------
print("\nStage 1, beam-5, seed-level 95% t-interval over seeds 42/101/202 (df 2), paired against the in-job control")
for arm, ctrl, nm in (("E1", "A", "Sign targets"), ("ST", "A", "Re-timing"), ("P", "AC", "Preference stage")):
    v, t = contrast(arm, ctrl, S1, "val"), contrast(arm, ctrl, S1, "test")
    rc = contrast(arm, ctrl, S1, "test", "recall")
    print(f"{nm:17s} validation {fmt(v)} ({sum(x > 0 for x in v)}/3 seeds higher) | test {fmt(t)} ({sum(x > 0 for x in t)}/3)"
          f" | test content-word recall {fmt(rc)} points | output length ratio val {mean(contrast(arm, ctrl, S1, 'val', 'lenr')):+.3f}"
          f" ({sum(x > 0 for x in contrast(arm, ctrl, S1, 'val', 'lenr'))}/3 longer)")
mv = mean(contrast("E1", "A", S1, "val"))
print(f"Sign targets: validation mean {f2(mv)} > 0 -> passes the futility screen (stopped only if <= 0)")
for arm, ctrl, nm in (("ST", "A", "Re-timing"), ("P", "AC", "Preference stage")):
    v, t, r = (mean(contrast(arm, ctrl, S1, sp, k)) for sp, k in (("val", "bleu"), ("test", "bleu"), ("test", "recall")))
    print(f"{nm}: exploratory rule (validation, test and test recall means all > 0): "
          + ("POSITIVE" if v > 0 and t > 0 and r > 0 else "NOT POSITIVE")
          + f"  (val {f2(v)}, test {f2(t)}, recall {f2(r)})")

# ---- Stage 2 (five seeds) --------------------------------------------------------------------------------------
S5 = S1 + S2
print("\nStage 2, five seeds 42/101/202/303/404 (df 4), paired against A")
tsum = {}
for arm, nm in (("E1", "Sign targets"), ("ST", "Re-timing")):
    v, t = contrast(arm, "A", S5, "val"), contrast(arm, "A", S5, "test")
    print(f"{nm:13s} with five seeds, validation {fmt(v)}, test {fmt(t)}; fresh seeds 303/404 test "
          + " ".join(f2(x) for x in t[3:]) + f" ({sum(x > 0 for x in t[3:])}/2 positive)")
    tstat = mean(t) / (stdev(t) / 5 ** 0.5)
    tsum[arm] = (tstat, 1 - tdist.cdf(tstat, 4))
for arm, level in (("E1", 0.01), ("ST", 0.05 / 6)):
    ts, p = tsum[arm]
    tc = tdist.ppf(1 - level, 4)
    print(f"{arm}: one-sided t(4) on the five test differences = {ts:.3f} (p = {p:.3f}); Holm level {level:.4f} needs t >= {tc:.3f} -> "
          + ("rejected" if ts >= tc else "not rejected, not adopted") + "  [levels from the pre-registration: E1 0.01, ST 0.05/6]")

# ---- alignment gate (read from the saved job summary) ----------------------------------------------------------
G = json.load(open(DATA / "g1_alignment_summary.json"))
a = G["mean_over_checkpoints"]["all"]
print(f"\nAlignment of the adopted CTC head (read from the saved job summary g1_alignment_summary.json; needs the checkpoints): "
      f"{G['n_all']} annotated sign spans: {100 * a['hit']:.1f}% hit against {100 * a['random']:.1f}% for random placement")
THR = 0.5  # threshold stated in the saved gate rule
print(f"Gate rule: {G['rule']}; hit >= {THR*100:.0f}%: {a['hit'] >= THR}; hit >= 2 x random: {a['hit'] >= 2 * a['random']} -> "
      + ("gate failed" if not (a['hit'] >= 0.5 and a['hit'] >= 2 * a['random']) else "gate passed")
      + f" (saved verdict {G['verdict']})")
