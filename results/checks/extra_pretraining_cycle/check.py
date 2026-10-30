"""Wave 12h (extra pre-training cycle, Stage 2): recompute the appendix row from the saved beam-5 decodes.

Re-scores 30 saved decodes (arms C / PJ / PS x seeds 505-909 x val/test, beam 5, length penalty 0.6, best.pt)
through slt.metrics (detokenise, sacreBLEU 13a), gates against the BLEU the original jobs recorded, then forms
paired differences, 95% t-intervals and the one-sided p of the pre-registered family {PJ - C}.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from slt.metrics import corpus_bleu4, detokenize

SEEDS = [505, 606, 707, 808, 909]
REF = {sp: json.load(open(HERE / f"refs_{sp}.json"))["refs"] for sp in ("val", "test")}
assert all(len(v) == 800 for v in REF.values())

def bleu(h, sp):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in REF[sp]], "13a")

B = {}  # B[(arm, seed, split)]
maxdev = 0.0
for s in SEEDS:
    hy = json.load(open(HERE / f"hyps_s{s}.json"))
    rec = json.load(open(HERE / f"recorded_s{s}.json"))
    for a in ("C", "PJ", "PS"):
        for sp in ("val", "test"):
            b = bleu(hy[a][sp], sp)
            B[(a, s, sp)] = b
            maxdev = max(maxdev, abs(b - rec[a][f"beam5_{sp}"]))
print(f"GATE: 30 decodes re-scored; max |recomputed - BLEU recorded by the jobs| = {maxdev:.2e}")
if maxdev > 1e-9:
    sys.exit("GATE FAILED")
g = B[("PJ", 808, "test")]
print(f"GATE: PJ seed 808 beam-5 test = {g:.6f} (recorded in the read-out: 10.080669)")
if abs(g - 10.08066901683454) > 1e-9:
    sys.exit("GATE FAILED")

# --- recipe (claims: 92-epoch first cycle, 40-epoch training, five new seeds, 13,760 synthetic clips)
ep = [json.load(open(HERE / f"epochs_s{s}.json")) for s in SEEDS]
pj = {e["PJ"]["pre_epochs"] for e in ep}; ps = {e["PS"]["pre_epochs"] for e in ep}
mn = {e[a]["main_epochs"] for e in ep for a in ("C", "PJ", "PS")}
print(f"first cycle on the adopted corpus (PJ): {sorted(pj)} epochs; then main training: {sorted(mn)} epochs; seeds: {len(SEEDS)} ({SEEDS})")
print(f"first cycle on the synthetic clips alone (PS): {sorted(ps)} epochs")
for l in open(HERE / "recipe_excerpt.txt"):
    if "13,760" in l or "N_SYN" in l:
        print("read from the job script header:", l.strip().lstrip("# "))
assert pj == {92} and mn == {40} and len(SEEDS) == 5

def diff(a, b, sp):
    d = np.array([B[(a, s, sp)] - B[(b, s, sp)] for s in SEEDS])
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return d, m, m - t * se, m + t * se, m / se

def show(name, a, b, sp):
    d, m, lo, hi, _ = diff(a, b, sp)
    print(f"{name}: {m:+.2f} ({lo:+.2f} to {hi:+.2f})  [{m:+.3f} [{lo:+.3f}, {hi:+.3f}]]  higher in {(d > 0).sum()} of {len(d)} seeds")
    return d, m, lo, hi

print()
d, m, lo, hi = show("PJ - C, beam-5 test (primary)", "PJ", "C", "test")
print("   per seed:", " / ".join(f"{x:+.3f}" for x in d))
show("PJ - C, beam-5 validation", "PJ", "C", "val")
t = diff("PJ", "C", "test")[4]
p = stats.t.sf(t, 4)
print(f"one-sided paired t on the five test differences: t = {t:.3f}, p = {p:.2f} ({p:.3f}); Holm level 0.05 (m = 1): {'reject' if p < 0.05 else 'retain'}")
print(f"both fresh seeds (808, 909) positive on test: {all(B[('PJ',s,'test')] > B[('C',s,'test')] for s in (808, 909))}")
print()
show("PS - C, beam-5 test (synthetic first cycle vs none)", "PS", "C", "test")
show("PS - PJ, beam-5 test (synthetic vs joint first cycle)", "PS", "PJ", "test")
print()
print("five-seed means, beam-5 val / test:",
      "; ".join(f"{a} {np.mean([B[(a,s,'val')] for s in SEEDS]):.3f} / {np.mean([B[(a,s,'test')] for s in SEEDS]):.3f}" for a in ("C", "PJ", "PS")))
