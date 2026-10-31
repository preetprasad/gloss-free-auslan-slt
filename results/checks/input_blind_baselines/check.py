"""Input-blind constant baseline: official test vs unseen targets (Results 'null model', Introduction, Discussion).

One five-word training sentence (named by SHA-1 digest) is output for every clip. Recomputes its BLEU-4 on all 800
official test clips and on the unseen subset, counts the test references equal to it, and compares with the trained
systems' unseen-target scores (saved beam-5 test decodes).
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from slt.metrics import detokenize, corpus_bleu4
from restore_references import sentence

SG = ROOT / "results/semantic_adequacy_grading"
def load(p):
    return sorted((json.loads(l) for l in open(p)), key=lambda r: r["idx"])
dwc = load(SG / "hyps_dwpose_ctc.jsonl")
s6 = load(ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl")
rawrefs = [r["ref"] for r in dwc]
N = len(rawrefs)
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = [l.rstrip("\n").split("\t", 1)[1].strip() for l in open(man) if "\t" in l]
    assert m == rawrefs, "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")
R = [detokenize(r) for r in rawrefs]
digests = set(open(ROOT / "results/train_target_hashes.txt").read().split())
seen = [hashlib.sha1(r.encode("utf-8")).hexdigest() in digests for r in R]
unseen = [i for i, s in enumerate(seen) if not s]

# GATE: recorded S6 (11.802) and DWPose+CTC (9.52) beam-5 test scores
bs6 = corpus_bleu4([detokenize(r["hyp_beam"]) for r in s6], R)
bdw = corpus_bleu4([detokenize(r["hyp_beam"]) for r in dwc], R)
print(f"GATE S6 {bs6:.3f} (recorded 11.802), DWPose+CTC {bdw:.2f} (recorded 9.52)")
if round(bs6, 3) != 11.802 or round(bdw, 2) != 9.52:
    sys.exit("GATE FAILED")

const_raw = sentence("636ebd35c0929c80790e3b0c57919dfb05157d7f")
const = detokenize(const_raw)
print(f"\nSetting: {N} official test clips, one constant training sentence for every clip, BLEU-4 13a on detokenised text")
print(f"constant sentence has {len(const_raw.split())} whitespace tokens incl. final full stop = {len(const.split())} words")
print(f"The null model outputs one sentence of {len(const.split())} words for every clip  [c07-063 c11-079]")
C = [const] * N
b_all = corpus_bleu4(C, R)
b_un = corpus_bleu4([const] * len(unseen), [R[i] for i in unseen])
si = [i for i in range(N) if seen[i]]
b_se = corpus_bleu4([const] * len(si), [R[i] for i in si])
n_eq = sum(r == const for r in R)
print(f"Constant, all {N} test clips: BLEU-4 {b_all:.2f} ({b_all:.3f})  [c01-025 c02-024 c11-079 c14-011 c14-077]")
print(f"Test clips: {N} (800)  [c02-024 c11-079]")
print(f"Constant, {len(unseen)} unseen-target clips: BLEU-4 {b_un:.2f} ({b_un:.3f})  [c01-025 c02-023 c11-081 c14-011 c14-077]")
print(f"Unseen-target clips: {len(unseen)} (570)  [c02-023]")
print(f"Constant, {len(si)} seen-target clips: BLEU-4 {b_se:.2f}")
print(f"Test references exactly equal to the constant: {n_eq}  [c11-079]")
print(f"Constant: one sentence, the same for all clips ({len(set(C))} distinct output)  [c14-077 c02-023]")

def unseen_bleu(rows):
    return corpus_bleu4([detokenize(rows[i]["hyp_beam"]) for i in unseen], [R[i] for i in unseen])
u_dw, u_s6 = unseen_bleu(dwc), unseen_bleu(s6)
print(f"\nTrained systems, unseen-target BLEU-4 (beam 5; S6 = ensemble beam 10, LP 0.3):")
print(f"DWPose+CTC single model {u_dw:.2f}  [c11-081]")
print(f"S6 ensemble {u_s6:.2f}  [c11-081]")
print(f"Constant {b_un:.2f} vs trained {u_dw:.2f} to {u_s6:.2f}; ratio of the lowest trained score to the constant {u_dw/b_un:.1f}  [c11-081]")

# published baselines on Auslan-Daily Communication pose (cited values, not our measurement)
for name, pub in (("SL-Transformer", 9.20), ("GASLT", 9.95)):
    print(f"constant / published {name} ({pub} published) = {b_all/pub:.2f}  [c14-077]")
