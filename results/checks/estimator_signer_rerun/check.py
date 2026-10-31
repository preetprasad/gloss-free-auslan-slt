"""Estimator comparison on the signer's keypoints (job 196407) and the signer-keypoint probes."""
import csv, hashlib, json, re, statistics, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

D = {  # beam-5, length penalty 0.6, best checkpoint, official test (published dumps of job 196407)
    ("Eap", 42): ROOT / "results/estimator_signer_keypoints_rerun/runs/Eap_s42/beam5_test_best.json",
    ("Eap", 101): ROOT / "results/estimator_signer_keypoints_rerun/runs/Eap_s101/beam5_test_best.json",
    ("Eap", 202): ROOT / "results/estimator_signer_keypoints_rerun/runs/Eap_s202/beam5_test_best.json",
    ("Edw", 42): ROOT / "results/estimator_signer_keypoints_rerun/runs/Edw_s42/beam5_test_best.json",
    ("Edw", 101): ROOT / "results/estimator_signer_keypoints_rerun/runs/Edw_s101/beam5_test_best.json",
    ("Edw", 202): ROOT / "results/estimator_signer_keypoints_rerun/runs/Edw_s202/beam5_test_best.json",
}
SEEDS = (42, 101, 202)
T975_DF2 = 4.303   # two-sided 95% t, 2 degrees of freedom (tools/seed_decision.py)

def interval(d):
    m = statistics.mean(d); hw = T975_DF2 * statistics.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw

norm = lambda t: " ".join(t.lower().split())
digests = set(open(ROOT / "results/train_target_hashes.txt").read().split())
J = {k: json.load(open(p)) for k, p in D.items()}
refs0 = J[("Eap", 42)]["refs"]
assert all(j["refs"] == refs0 and len(j["beam_hyps"]) == 800 and j["beam"] == 5 for j in J.values())
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
    assert sorted(m.values()) == sorted(r.strip() for r in refs0), "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")
R = [detokenize(r) for r in refs0]
unseen = [i for i, r in enumerate(refs0) if hashlib.sha1(norm(r).encode()).hexdigest() not in digests]

bleu, ubleu = {}, {}
for k, j in J.items():
    H = [detokenize(h) for h in j["beam_hyps"]]
    bleu[k] = corpus_bleu4(H, R)
    ubleu[k] = corpus_bleu4([H[i] for i in unseen], [R[i] for i in unseen])
    # GATE: equals the BLEU recorded by the original job in the dump
    rec = j["beam_result"]["metrics"]["tokenize=13a;detok=True"]
    if abs(bleu[k] - rec) > 1e-6:
        sys.exit(f"GATE FAILED {k}: recomputed {bleu[k]} recorded {rec}")
print("GATE: all six recomputed beam-5 test BLEU-4 equal the values recorded in the job's dumps (to 1e-6)")

print("\nSetting: Auslan-Daily Communication official test, 800 clips, beam width 5, three seeds, AlphaPose and DWPose trained in one job (196407)  [c08-029 c08-038 c14-018]")
print("AlphaPose arm: HALPE-136 body+hands keypoints of the entry marked as the signer; DWPose arm: DWPose  [c08-034 c08-035]")
for a in ("Eap", "Edw"):
    print(f"{a} beam-5 test BLEU-4 per seed: " + " / ".join(f"{bleu[(a, s)]:.2f}" for s in SEEDS))
ma = statistics.mean(bleu[("Eap", s)] for s in SEEDS); md = statistics.mean(bleu[("Edw", s)] for s in SEEDS)
print(f"AlphaPose, signer keypoints (3 seeds) test mean {ma:.2f}  [c08-034 c08-043 c08-044 c14-018]")
print(f"DWPose (3 seeds, same job) test mean {md:.2f}  [c08-035 c08-043 c08-044 c14-018]")
dl = [bleu[("Edw", s)] - bleu[("Eap", s)] for s in SEEDS]
m, lo, hi = interval(dl)
print("DWPose minus AlphaPose per seed: " + " / ".join(f"{x:+.2f}" for x in dl))
print(f"difference {m:+.2f}, 95% t-interval {lo:+.2f} to {hi:+.2f}  [c08-043 c14-018]")
npos = sum(x > 0 for x in dl)
print(f"DWPose higher in {npos} of {len(dl)} seeds  [c08-043 c21-022]")
print(f"interval contains zero: {lo < 0 < hi}; contains +1.0 (about one point): {lo < 1.0 < hi}; contains -1.0: {lo < -1.0 < hi}  [c14-021]")
print(f"Figure bars: means of {len(SEEDS)} seeds with 95% t-intervals (t = {T975_DF2}, 2 degrees of freedom); 2 estimators compared  [c08-038 c14-021]")
print(f"two estimators, {len(SEEDS)} seeds: neither a ranking (interval contains zero) nor exclusion of a one-point difference (interval spans -1 to +1)  [c14-021]")

ud = [ubleu[("Edw", s)] - ubleu[("Eap", s)] for s in SEEDS]
um, ulo, uhi = interval(ud)
print(f"\nUnseen-reference test clips ({len(unseen)} of 800, reference absent from training, via results/train_target_hashes.txt)")
print(f"unseen BLEU-4 DWPose minus AlphaPose per seed: " + " / ".join(f"{x:+.2f}" for x in ud))
print(f"on test clips whose reference is absent from training: {um:+.2f} (95% t-interval {ulo:+.2f} to {uhi:+.2f})  [c08-031]")

# ---- hand-identification probe (continuous_id_probe, 45 signs, clip-disjoint folds), per-span outputs
def acc(path, var):
    rows = [r for r in csv.DictReader(open(HERE / path)) if r["variant"] == var]
    return rows, 100 * sum(int(r["correct"]) for r in rows) / len(rows)
def pairs(var):
    a = {r["span"]: int(r["correct"]) for r in csv.DictReader(open(HERE / "pred_dwpose.csv")) if r["variant"] == var}
    b = {r["span"]: int(r["correct"]) for r in csv.DictReader(open(HERE / "pred_ap_clips_signer.csv")) if r["variant"] == var}
    sh = sorted(set(a) & set(b)); oa = sum(a[s] and not b[s] for s in sh); ob = sum(b[s] and not a[s] for s in sh)
    return len(sh), oa, ob, binomtest(oa, oa + ob, 0.5).pvalue
log = {k: open(HERE / f).read() for k, f in (("dw", "log_dwpose.txt"), ("ap", "log_ap_clips_signer.txt"))}
def top5(txt, name):
    return float(re.search(rf"^\s*{re.escape(name)}\+vel .*?TOP-1\s+[\d.]+%\s+TOP-5\s+([\d.]+)%", txt, re.M).group(1))
_, h_dw = acc("pred_dwpose.csv", "hands"); _, h_ap = acc("pred_ap_clips_signer.csv", "hands")
_, b_dw = acc("pred_dwpose.csv", "body_hands"); _, b_ap = acc("pred_ap_clips_signer.csv", "body_hands")
print("\nHand sign-identification probe (nearest centroid, 45 signs, annotated spans of the Communication training clips)")
print(f"hand keypoints alone, top-1: DWPose {h_dw:.1f}%, signer's AlphaPose {h_ap:.1f}%  [c08-047]")
print(f"hand keypoints alone, top-5: DWPose {top5(log['dw'], 'hands'):.1f}%, signer's AlphaPose {top5(log['ap'], 'hands'):.1f}% (read from the saved probe logs)  [c08-047]")
print(f"body and hands, top-1: {b_dw:.1f}% and {b_ap:.1f}%  [c08-048]")
n, oa, ob, p = pairs("hands")
print(f"exact McNemar test, hands: {n} shared spans, DWPose only right {oa}, AlphaPose only right {ob}, p = {p:.3f}  [c08-048]")

# ---- mean hand confidence of the signer entry (all 14,041 clips): value saved by the original probe run
pj = json.load(open(HERE / "probe.json"))
hc = pj["hand_conf_alphapose_signer"]
print(f"\nmean confidence of the hand keypoints of the signer entry in the distributed files: {hc['mean']:.2f} ({hc['mean']:.4f}) over {hc['n_clips']:,} clips (read from the saved probe output probe.json)  [c08-018]")

# ---- which person: shoulder-normalised arm trajectories, DWPose tracked vs the entry marked as the signer
sc = json.load(open(HERE / "signer_check_results.json"))
g = sc["gate_stored_KPS"]
print(f"\nClips with several people: {sc['n_multi']} analysed + {sc['dw_missing']} without a DWPose file = {sc['n_multi'] + sc['dw_missing']:,} of {sc['n_multi'] + sc['n_single'] + sc['dw_missing']:,} clips  [c19-057 c06-020]")
print(f"the entry marked as the signer follows the tracked DWPose arm motion (r > 0.5) in {100 * g['share_r_gt_0p5_of_defined']:.1f}% of those clips with a defined r (read from the saved per-clip check signer_check_results.json)  [c06-020]")
