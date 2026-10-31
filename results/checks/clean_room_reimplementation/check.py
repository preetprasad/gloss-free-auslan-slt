#!/usr/bin/env python3
"""Clean-room code (parity, clean re-decode of the fine-tuned models) and own-code fine-tuning (K, KSA).

Run from src/:  python3 ../results/checks/clean_room_reimplementation/check.py
Re-scores saved decodes (hypotheses + references) with slt.metrics; the logit-parity number is read from
the saved summary of the parity job (no per-item data was kept for it).
"""
import hashlib, json, sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

D = HERE / "data"
L = lambda n: json.load(open(D / n))
SEEDS5, SEEDS3 = ["42", "101", "202", "303", "404"], ["42", "101", "202"]
SPL = ("val", "test")
fails = []


def gate(ok, msg):
    if not ok:
        fails.append(msg)
        print("GATE FAILED:", msg)


digests = set(open(ROOT / "results" / "train_target_hashes.txt").read().split())
refs, unseen = {}, {}
for sp in SPL:
    refs[sp] = [detokenize(r) for r in L(f"refs_{sp}.json")["refs"]]
    unseen[sp] = np.array([hashlib.sha1(" ".join(r.lower().split()).encode()).hexdigest() not in digests
                           for r in refs[sp]])
    gate(len(refs[sp]) == 800, f"{sp}: 800 references")
gate(int(unseen["test"].sum()) == 570, "test: 570 unseen references")


def score(hyps, sp):
    h = [detokenize(x) for x in hyps]
    ub = corpus_bleu4([x for x, m in zip(h, unseen[sp]) if m], [x for x, m in zip(refs[sp], unseen[sp]) if m])
    return corpus_bleu4(h, refs[sp]), ub


def load(name, seeds, sp):
    """-> {seed: (bleu, unseen_bleu, hyps)}; gate: every run reproduces the BLEU-4 its own job recorded."""
    out = {}
    for s, v in L(name).items():
        if s not in seeds:
            continue
        b, u = score(v["hyps"], sp)
        gate(abs(b - v["recorded_bleu4"]) < 1e-6, f"{name} seed {s}: {b:.6f} vs recorded {v['recorded_bleu4']:.6f}")
        if "job_gate_clean_bleu4" in v:
            gate(abs(b - v["job_gate_clean_bleu4"]) < 1e-6, f"{name} seed {s} vs job gate")
        out[s] = (b, u, v["hyps"])
    return out


port = {sp: load(f"port_u_asl_{sp}.json", SEEDS5, sp) for sp in SPL}
clean = {sp: load(f"clean_redecode_{sp}.json", SEEDS5, sp) for sp in SPL}
K = {sp: load(f"ownft_K_{sp}.json", SEEDS3, sp) for sp in SPL}
KSA = {sp: load(f"ownft_KSA_{sp}.json", SEEDS3, sp) for sp in SPL}
mean = lambda r, i=0: float(np.mean([v[i] for v in r.values()]))
print("gates passed: every run re-scores to the BLEU-4 recorded by its own job (tolerance 1e-6)"
      if not fails else "")

# ---- 1. logit parity of the independent implementation (parity job 186422, read from its saved summary)
gk = L("gk_job186422.json")
names = set(L("gk_clips_names.json")["clip_files"])
val_names = set()
mp = ROOT / "comm_run" / "manifests" / "val.tsv"
if mp.exists():
    val_names = {l.split("\t")[0].replace("\\", "/").split("/")[-1] for l in open(mp) if "\t" in l}
gate(gk["passed"] and gk["logits_dev"] < gk["tolerances"]["logits_abs"], "parity job passed")
print("\n[1] Independent implementation vs the private port, released checkpoint (read from the saved job summary gk_job186422.json)")
print(f"  clips compared: {len(names)}; all in the validation manifest: {names <= val_names if val_names else 'manifest not found'}")
print(f"  max |logit difference| = {gk['logits_dev']:.1e}  (rounds to {float('%.0e' % gk['logits_dev']):.0e}; tolerance {gk['tolerances']['logits_abs']:.0e})")
print(f"  exponent form: {gk['logits_dev']/10**int(np.floor(np.log10(gk['logits_dev']))):.1f} x 10^{int(np.floor(np.log10(gk['logits_dev'])))}; tolerance exponent {int(np.floor(np.log10(gk['tolerances']['logits_abs'])))}")
print(f"  max relative embedding difference = {gk['emb_dev']:.1e}; passed = {gk['passed']}")

# ---- 2. clean re-decode of the five port-fine-tuned models
print("\n[2] The five fine-tuned models decoded with our own code (beam 5, LP 1.0) vs the port's decodes")
print("  seed | split | port BLEU-4 | our code | identical hypotheses")
fr = []
for sp in SPL:
    for s in SEEDS5:
        n = sum(a == b for a, b in zip(port[sp][s][2], clean[sp][s][2]))
        fr.append(n / 8.0)
        print(f"  {s:>4} | {sp:<5} | {port[sp][s][0]:.2f} | {clean[sp][s][0]:.2f} | {n}/800 = {n/8:.1f}%")
for sp in SPL:
    print(f"  {sp}: port mean {mean(port[sp]):.2f} / our code {mean(clean[sp]):.2f}; "
          f"unseen-570 {mean(port[sp],1):.2f} / {mean(clean[sp],1):.2f}")
print(f"  identical hypotheses over the ten seed x split comparisons: {min(fr):.1f}% to {max(fr):.1f}%")
print(f"  rounded as in the dissertation: {min(fr):.0f} to {max(fr):.0f}% of outputs identical")

# ---- 3. own-code fine-tuning: K and KSA
print("\n[3] Fine-tuning with our own code, 3 seeds (42/101/202): K = plain, KSA = + stitched data and pose augmentation")
print("  system | split | mean BLEU-4 | unseen-570 (test only)")
for nm, R in (("K", K), ("KSA", KSA)):
    for sp in SPL:
        extra = f" | unseen {mean(R[sp],1):.2f}" if sp == "test" else ""
        print(f"  {nm:<3} | {sp:<5} | {mean(R[sp]):.2f}{extra}")
t95 = lambda x: (lambda m, se, t: (m, m - t * se, m + t * se))(
    np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x)), stats.t.ppf(0.975, len(x) - 1))
print("  KSA minus K, per seed (seed-level 95% t-interval):")
pos = {}
for sp in SPL:
    d = [KSA[sp][s][0] - K[sp][s][0] for s in SEEDS3]
    m, lo, hi = t95(d)
    pos[sp] = sum(x > 0 for x in d)
    print(f"  {sp}: {', '.join(f'{x:+.2f}' for x in d)}; mean {m:+.2f} ({lo:+.2f} to {hi:+.2f}); positive in {pos[sp]} of 3 seeds")
print(f"  futility rule (3-seed mean validation difference <= 0): "
      f"{'STOP' if np.mean([KSA['val'][s][0] - K['val'][s][0] for s in SEEDS3]) <= 0 else 'continue'}")

# ---- 4. own code minus port
print("\n[4] Our code minus the port, fine-tuned models (K 3 seeds minus port 5 seeds)")
for sp in SPL:
    print(f"  {sp}: {mean(K[sp]):.2f} - {mean(port[sp]):.2f} = {mean(K[sp]) - mean(port[sp]):+.2f}")
sys.exit(1 if fails else 0)
