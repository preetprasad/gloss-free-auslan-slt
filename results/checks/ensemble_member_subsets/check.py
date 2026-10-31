"""Ensemble member-set ablation and member-set selection (dissertation Results, Discussion, Appendix B).

Recomputes from saved decodes: (1) the 63-subset ablation of the six final-ensemble members (validation, beam 10,
LP 0.3), its Shapley values and the 500-resample bootstrap of the kind difference; (2) validation-vs-test of four
ensemble member sets; (3) the rebuilt ensemble and its 1,000-halving test; (4) the video-disjoint ensemble.
"""
import hashlib, json, re, sys
from pathlib import Path
from itertools import combinations
from math import factorial
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
from slt.metrics import detokenize, corpus_bleu4, corpus_wer
from sacrebleu.metrics import BLEU
from holm_decision import simultaneous_band

D = HERE / "data"
BL = BLEU()


def stats(h, r):  # per-sentence sufficient statistics of sacreBLEU 13a (identical to corpus_bleu4 after summing)
    return np.array(BL._extract_corpus_statistics(h, [r]), dtype=float)


def sc(st):
    return BL._compute_score_from_stats(list(st.sum(0))).score


def load_dump(p):
    rows = sorted((json.loads(l) for l in open(p, encoding="utf-8")), key=lambda r: r["idx"])
    return [detokenize(r["hyp_beam"]) for r in rows], [detokenize(r["ref"]) for r in rows]


def bleu_dump(p):
    h, r = load_dump(p)
    return corpus_bleu4(h, r), h, r


def shapley(v, n):
    phi = [0.0] * n
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                m = sum(1 << j for j in S)
                phi[i] += w * (v[m | 1 << i] - v[m])
    return phi


def fail(msg):
    sys.exit("GATE FAILED: " + msg)


# ------------------------------------------------------------------ 1. the 63-subset ablation
saved = json.load(open(ROOT / "results/ensemble_member_subsets_ablation/subsets_scores.json"))
names = saved["names"]; n = len(names); full = (1 << n) - 1
hyps, refs = {}, None
for k in range(4):
    d = json.load(open(ROOT / f"results/ensemble_member_subsets_ablation/shard{k}of4.json"))
    refs = refs or d["refs"]
    assert refs == d["refs"]
    hyps.update({int(m): h for m, h in d["hyps"].items()})
assert len(refs) == 800 and sorted(hyps) == list(range(1, full + 1))
R = [detokenize(x) for x in refs]
H = {m: [detokenize(x) for x in h] for m, h in hyps.items()}
bf = corpus_bleu4(H[full], R); wf = corpus_wer(H[full], R)
print(f"GATE six-member set, validation beam 10 / LP 0.3: BLEU-4 {bf:.3f} WER {wf:.4f} (recorded 12.470 / 0.9164, job 120153)")
if abs(bf - 12.470) > 0.005 or abs(wf - 0.9164) > 0.0005:
    fail("full set does not reproduce")
ST = {m: stats(H[m], R) for m in H}
v = {0: 0.0, **{m: sc(ST[m]) for m in ST}}
assert max(abs(v[m] - corpus_bleu4(H[m], R)) < 1e-9 for m in (1, 7, full)) and all(abs(v[m] - corpus_bleu4(H[m], R)) < 1e-9 for m in (1, 7, 63))
print(f"\nDecoded and scored {len(ST)} non-empty subsets of the {n} members ({', '.join(names)}) on {len(R)} validation clips  [c12-017]")
print(f"Number of members {n}, number of subsets {len(ST)}  [c12-017]")
order = [(1 << (k + 1)) - 1 for k in range(n)]
print("Members added in the recorded order, validation BLEU-4: " + ", ".join(f"{v[m]:.2f}" for m in order) + "  [c18-058]")

phi = shapley(v, n); assert abs(sum(phi) - v[full]) < 1e-9
types = re.search(r"--types\s+(\S+)", open(ROOT / "jobs/slurm/ens_subsets.sbatch").read()).group(1).split(",")
assert len(types) == n
S_f = sum(p for p, t in zip(phi, types) if t == "fusion"); S_p = sum(p for p, t in zip(phi, types) if t == "pose")
# bootstrap of the kind difference: 500 sentence resamples, seed 0, one draw per resample (as the original job)
tm = [sum(1 << i for i in range(n) if types[i] == t) for t in ("fusion", "pose")]
rng = np.random.default_rng(0); td = []
for b in range(500):
    ix = rng.integers(0, len(R), len(R))
    vb = {0: 0.0, **{m: sc(ST[m][ix]) for m in ST}}
    pb = shapley(vb, n)
    td.append(sum(p for p, t in zip(pb, types) if t == "fusion") - sum(p for p, t in zip(pb, types) if t == "pose"))
lo, hi = np.percentile(td, [2.5, 97.5])
print(f"Shapley sums: fusion members {S_f:.2f}, pose members {S_p:.2f}  [c12-021]")
print(f"Fusion minus pose Shapley sum {S_f - S_p:+.2f}, 95% interval {lo:+.2f} to {hi:+.2f} (500 resamples)  [c12-021]")
print(f"read from the saved job output ensemble_member_subsets_ablation/subsets_scores.json for comparison: type_diff {saved['type_diff']['shapley_sum']}; recomputed matches: "
      f"{abs(lo - saved['type_diff']['shapley_sum'][0]) < 1e-9 and abs(hi - saved['type_diff']['shapley_sum'][1]) < 1e-9}")

# ------------------------------------------------------------------ 2. member sets: validation vs test
def section_line(path, header, pat, which=0):
    cur, out = None, []
    for l in open(path):
        if l.startswith("=== CANDIDATE"):
            cur = l.split()[2]
        m = re.search(pat, l)
        if m and cur == header:
            out.append(m)
    return out[which]

def lp_val(path, cand, lp):
    return float(section_line(path, cand, rf"LP={lp}\s+BLEU-4=\s*([\d.]+)").group(1))

val_E = lp_val(HERE / "ens-ext-1197079.out", "X2_DUR3", "0.8")
val_F = lp_val(HERE / "ens-search-1197719.out", "G5_DUR3_FUS2_eqF", "0.7")
val_G1 = lp_val(HERE / "ens-search-1197719.out", "G1_DUR6_FUS2", "0.7")
val_S6 = bf
t_S6, hS6, rS6 = bleu_dump(ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl")
t_E, _, _ = bleu_dump(ROOT / "results/ensemble_three_duration_members_decodes/hyps_LP=0.8.jsonl")
t_F, hF, _ = bleu_dump(ROOT / "results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl")
t_G1, _, _ = bleu_dump(D / "dumps_G1_test__hyps_LP=0.7.jsonl")
t_Fp, hFp, _ = bleu_dump(ROOT / "results/ensemble_one_to_one_duration_members_decodes/hyps_LP=0.6.jsonl")
t_Fa, hFa, rFa = bleu_dump(D / "dumps_Faug_test__hyps_LP=0.8.jsonl")
for nm, got, want in [("S6", t_S6, 11.802), ("E_dur_x3", t_E, 11.537), ("F", t_F, 12.114), ("G1", t_G1, 11.562), ("F-prime", t_Fp, 12.836), ("F-triple-prime", t_Fa, 12.727)]:
    print(f"GATE {nm} official test BLEU-4 {got:.3f} (recorded {want})")
    if abs(got - want) > 0.0015:
        fail(nm)
print("\nMember set | validation BLEU-4 (read from the saved job output for E, F, G1; S6 from the subset decode) | official test BLEU-4 (recomputed from the dump)")
for nm, a, b in [("S6", val_S6, t_S6), ("three stitched-data models", val_E, t_E), ("F", val_F, t_F), ("six stitched-data models plus two", val_G1, t_G1)]:
    print(f"  {nm:34s} {a:7.3f}  {b:7.3f}")
r = np.corrcoef([val_S6, val_E, val_F, val_G1], [t_S6, t_E, t_F, t_G1])[0, 1]
print(f"Four member sets scored on both splits  [c12-038]")
print(f"Pearson r between validation and test BLEU-4 {r:+.3f}  [c12-038]")
print(f"Larger variant (six stitched-data models plus two RGB-fusion): validation {val_G1:.2f}, test {t_G1:.2f}  [c12-037]")
print(f"Pose-only ensemble of three stitched-data models: validation {val_E:.3f} against {val_S6:.3f} for the final ensemble; test {t_E:.2f} against {t_S6:.2f}  [c18-097]")
print(f"Ensembles that scored above {t_S6:.2f} on test: F {t_F:.2f}, F-prime {t_Fp:.2f}, rebuilt {t_Fa:.2f}; count above = {sum(x > t_S6 for x in (t_F, t_Fp, t_Fa))}  [c14-084]")

# member counts from the saved decode logs
def members(path):
    m = re.search(r"(\d+) members fusion=\[([^\]]*)\]", open(path).read())
    fl = [x.strip() == "True" for x in m.group(2).split(",")]
    return len(fl), fl.count(False), fl.count(True)
nt, npz, nfu = members(D / "log_Faug_test.log")
print(f"Rebuilt ensemble (saved decode log): {nt} members, {npz} without RGB fusion, {nfu} RGB-fusion; test BLEU-4 {t_Fa:.2f}  [c12-040 c18-006]")
# settings of the member checkpoints, read from the cfg saved inside each checkpoint (data/member_cfgs.json, made by prepare_data.py)
MC = json.load(open(D / "member_cfgs.json"))
paug = ["pa_aug_s42", "pa_aug_s101", "pa_aug_s202"]; fus = ["rgb_fusion", "fus_s101"]; fprime = ["r111_s42", "r111_s101", "r111_s202"]
flags = [x.strip() == "True" for x in re.search(r"members fusion=\[([^\]]*)\]", open(D / "log_Faug_test.log").read()).group(1).split(",")]
kinds = [MC[k]["uses_rgb_features"] for k in paug + fus]
assert kinds == flags, (kinds, flags)
print(f"Member kinds in the order of the decode (3 pose-augmented checkpoints, then 2 RGB-fusion): from the checkpoints' saved settings {kinds}; from the decode log {flags}; equal: {kinds == flags}  [c12-040 c18-006]")
print(f"Pose augmentation switched on in the saved settings of {sum(MC[k]['pose_aug'] for k in paug)} of the 3 pose-augmented checkpoints (seeds {', '.join(str(MC[k]['seed']) for k in paug)})  [c18-006]")
print(f"Pose augmentation switched on in the {sum(MC[k]['pose_aug'] for k in fprime)} of 3 duration-stitch members of the earlier 1.1:1 ensemble (settings field present: {sum(MC[k]['pose_aug_field_present'] for k in fprime)})  [c18-006]")
adopted = {"train_manifest_name": "joint_13760.tsv", "num_keypoints": 133, "coord_norm": "frame", "epochs": 40, "lr": 0.0005, "lr_schedule": "cosine", "batch_size": 64, "uses_rgb_features": False, "pose_aug": True}
ok = [all(MC[k][f] == v for f, v in adopted.items()) for k in paug]
print(f"Settings of the three checkpoints: training manifest {MC[paug[0]]['train_manifest_name']} (13,760 stitched clips added to the real training clips), {MC[paug[0]]['num_keypoints']} keypoints, {MC[paug[0]]['coord_norm']} normalisation, "
      f"{MC[paug[0]]['epochs']} epochs, learning rate {MC[paug[0]]['lr']}, {MC[paug[0]]['lr_schedule']} schedule, batch {MC[paug[0]]['batch_size']}, pose augmentation on, no RGB input; all three agree: {all(ok)}  [c12-040]")
ratio = None
man = ROOT / "comm_run/manifests/train.tsv"
if man.exists():
    nreal = sum(1 for l in open(man, encoding="utf-8") if l.strip())
    ratio = 13760 / nreal
    print(f"Stitched to real training clips: 13,760 to {nreal:,} = {ratio:.1f}:1  [c12-040]")
print(f"The earlier 1.1:1 members of seeds 101 and 202 used the same training manifest and no pose augmentation: {all(MC[k]['train_manifest_name'] == 'joint_13760.tsv' and not MC[k]['pose_aug'] for k in fprime[1:])}  [c18-006]")
print("(The decode log does not name its checkpoint files; which checkpoints formed the rebuilt ensemble is recorded in the pre-registration, not in a saved output, so the settings above are those of the checkpoints it names. The log's member kinds and the remapped vocabulary of the last two members agree.)")
nt, npz, nfu = members(D / "vd_xmodal_31205__decode__VD5_test.log")
print(f"Video-disjoint ensemble (saved decode log): {nt} members, {npz} pose-only, {nfu} RGB-fusion  [c18-004]")

txt = open(HERE / "ens-ext-1197079.out").read().split("=== CANDIDATE X2_DUR3 ===")[1].split("=== CANDIDATE")[0]
m = re.search(r"(\d+) members fusion=\[([^\]]*)\]", txt); fl = [x.strip() == "True" for x in m.group(2).split(",")]
print(f"Pose-only duration ensemble (saved job output ens-ext 1197079, X2_DUR3): {len(fl)} members, {fl.count(True)} with RGB fusion  [c18-096]")

# ------------------------------------------------------------------ 3. test-tuned observations (job outputs)
def sections(path):
    """{(split, label): bleu4} from a job log laid out as '########## VAL|TEST ##########' / '### label ###' (swa-ens, ens-max)."""
    out, split, label = {}, None, None
    for l in open(path):
        m = re.match(r"#{10} (VAL|TEST) #{10}", l)
        if m: split = m.group(1)
        m = re.match(r"### (.+?) ###", l)
        if m and split: label = m.group(1)
        m = re.search(r"\[ENSEMBLE TEST\] beam\d+_bleu4=([\d.]+)", l)
        if m and split and label: out[(split, label)] = float(m.group(1))
    return out
def tier_sections(path):
    """{header: bleu4} for tier1: each '--- ... ---' header is followed by one score line."""
    out, hdr = {}, None
    for l in open(path):
        if l.startswith("--- "): hdr = l.strip()
        m = re.search(r"\[ENSEMBLE TEST\] beam\d+_bleu4=([\d.]+)", l)
        if m and hdr: out[hdr] = float(m.group(1))
    return out
T1 = tier_sections(HERE / "tier1-1082033.out")
t1_sel = max((h for h in T1 if h.startswith("--- VAL")), key=T1.get)
t1_sel_test = T1[t1_sel.replace("VAL", "TEST")]
tier = T1["--- TEST beam=10 lp=1.0 ---"]
SW = sections(HERE / "swa-ens-1094775.out"); MX = sections(HERE / "ens-max-1091645.out")
sw_sel = max((k for k in SW if k[0] == "VAL"), key=SW.get)[1]
mx_sel = max((k for k in MX if k[0] == "VAL"), key=MX.get)[1]
swa, mx = SW[("TEST", "D:SV+VL+R0+V0+C2+F1+F2 (D)")], MX[("TEST", "S9-weighted (beam=5 lp=1.0 w=1.3,1.3,1.0,0.9,0.9,0.9,0.8,0.8,0.9)")]
rows = [("beam/length-penalty re-tune of the two-member ensemble (tier1 1082033)", t1_sel.replace("--- VAL ", "").rstrip(" -"), t1_sel_test, "beam 10 / LP 1.0", tier),
        ("SWA ensemble candidates (swa-ens 1094775)", sw_sel, SW[("TEST", sw_sel)], "candidate D", swa),
        ("weighted ensembles (ens-max 1091645)", mx_sel.split(" (")[0], MX[("TEST", mx_sel)], "S9-weighted", mx)]
print("\nExploratory configurations chosen with test scores in view, against the validation-selected configuration of the same job (read from the saved job outputs):")
for nm, sel, st, tn, tv in rows:
    print(f"  {nm}: validation-selected {sel} scored {st:.2f} on test; {tn} scored {tv:.2f} ({tv - st:+.2f})")
print(f"Test-selected configurations scoring above the validation-selected one of their job: {sum(r[4] > r[2] for r in rows)} of {len(rows)}; their BLEU-4 {tier:.2f}, {swa:.2f} and {mx:.2f}  [c08-009]")

# ------------------------------------------------------------------ 4. paired bootstrap, rebuilt vs reported (test) and F vs S6
def paired(HA, HB, Rr, n=2000, seed=0):
    A, B = stats(HA, Rr), stats(HB, Rr); N = len(Rr)
    obs = sc(A) - sc(B)
    rng = np.random.default_rng(seed); draws = rng.integers(0, N, size=(n, N))
    d = np.array([sc(A[i]) - sc(B[i]) for i in draws])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, (d > 0).mean()
assert rFa == rS6
o, lo, hi, P = paired(hFa, hS6, rS6)
print(f"\nRebuilt ensemble minus final ensemble, official test, 2,000 resamples: {o:+.2f}, 95% interval {lo:+.2f} to {hi:+.2f}  [c18-007]")
print(f"Test BLEU-4 {t_Fa:.2f} against {t_S6:.2f}  [c18-007]")
oF, loF, hiF, _ = paired(hF, hS6, rS6)
print(f"F minus final ensemble, official test: {oF:+.2f}, 95% interval {loF:+.2f} to {hiF:+.2f}; includes zero: {loF < 0 < hiF}  [c01-028]")
print(f"F scored {t_F:.2f} on test against {t_S6:.2f}, having scored {val_F:.2f} against {val_S6:.2f} on validation (validation read from the saved job output ens-search / the subset decode): an apparent gain of {t_F - t_S6:+.2f} whose interval includes zero, the condition under which the dissertation withdrew it (the withdrawal itself is the author's decision, not a computed number)  [c01-028]")
# how many of the dissertation's paired comparisons exclude zero: the six rows of Table res_bootstrap, recomputed from the saved decodes (same convention: 2,000 resamples, seed 0, shared draws)
SG = ROOT / "results/semantic_adequacy_grading"
def ld(p):
    return sorted((json.loads(l) for l in open(p, encoding="utf-8")), key=lambda r: r["idx"])
sysd = {"CTC": ld(SG / "hyps_dwpose_ctc.jsonl"), "VLP": ld(SG / "hyps_vlp.jsonl"), "RGB": ld(SG / "hyps_rgbfusion.jsonl")}
MEMD = ROOT / "results/final_ensemble_member_test_decodes"
mem = {d.name: ld(d / "hyps_LP=0.3.jsonl") for d in sorted(MEMD.iterdir())}
Hs = {k: [detokenize(r["hyp_beam"]) for r in v] for k, v in sysd.items()}
for nm, rows_ in mem.items():
    Hs["m:" + nm] = [detokenize(r["hyp_beam"]) for r in rows_]
Hs["S6"], Hs["F"], Hs["Fp"] = hS6, hF, hFp
for nm, want in (("CTC", 9.522), ("VLP", 10.272), ("RGB", 10.335), ("Fp", 12.836)):
    got = corpus_bleu4(Hs[nm], rS6)
    print(f"GATE {nm} official test BLEU-4 {got:.3f} (recorded {want})")
    if abs(got - want) > 0.0015:
        fail(nm)
strongest = max((k for k in Hs if k.startswith("m:")), key=lambda k: corpus_bleu4(Hs[k], rS6))
SB = {k: stats(Hs[k], rS6) for k in ("CTC", "VLP", "RGB", "S6", "F", "Fp", strongest)}
draws = np.random.default_rng(0).integers(0, 800, size=(2000, 800))
BB = {k: np.array([sc(SB[k][i]) for i in draws]) for k in SB}
FULLB = {k: sc(SB[k]) for k in SB}
rowsB = [("S6", "RGB"), ("S6", strongest), ("RGB", "CTC"), ("VLP", "CTC"), ("F", "S6"), ("Fp", "S6")]
excl = 0
for a_, b_ in rowsB:
    d = BB[a_] - BB[b_]; lo_, hi_ = np.percentile(d, [2.5, 97.5]); ex = bool(lo_ > 0 or hi_ < 0); excl += ex
    print(f"  {a_} minus {b_.replace('m:comm_dwpose_', 'member ')}: {FULLB[a_] - FULLB[b_]:+.2f}, 95% interval {lo_:+.2f} to {hi_:+.2f}, excludes zero: {ex}")
print(f"Of the {len(rowsB)} paired comparisons of Table res_bootstrap, {len(rowsB) - excl} have an interval that includes zero and {excl} excludes it; the intervals' half-widths are about 1.2 BLEU-4  [c01-028]")

digests = set(open(ROOT / "results/train_target_hashes.txt").read().split())
norm = lambda s: " ".join(s.lower().split())
seen = np.array([hashlib.sha1(norm(x).encode()).hexdigest() in digests for x in rS6])
ub = corpus_bleu4([h for h, k in zip(hFa, seen) if not k], [x for x, k in zip(rS6, seen) if not k])
print(f"Rebuilt ensemble, unseen-target subset ({int((~seen).sum())} clips): BLEU-4 {ub:.2f}  [c18-008]")

# ------------------------------------------------------------------ 5. rebuild: validation halvings
lps = "0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0".split()
SV = {}
for lp in lps:
    h, rr = load_dump(D / f"dumps_Faug_val__hyps_LP={lp}.jsonl"); SV[lp] = stats(h, rr)
hs, rs = load_dump(ROOT / "results/final_ensemble_validation_decodes/hyps_LP=0.3.jsonl"); S6v = stats(hs, rs)
assert rr == rs
g1, g2 = sc(SV["0.8"]), sc(S6v)
print(f"\nGATE rebuilt ensemble validation BLEU-4 at LP 0.8 {g1:.3f} (recorded 13.472); reported ensemble at LP 0.3 {g2:.3f} (recorded 12.470)")
if abs(g1 - 13.472) > 0.0015 or abs(g2 - 12.470) > 0.0015:
    fail("rebuild gate")
rng = np.random.default_rng(0); dd = []
for _ in range(1000):
    p = rng.permutation(800); A, Bh = p[:400], p[400:]
    best = max(lps, key=lambda lp: sc(SV[lp][A]))
    dd.append(sc(SV[best][Bh]) - sc(S6v[Bh]))
dd = np.array(dd); lo, hi = np.percentile(dd, [2.5, 97.5])
print(f"Rebuilt minus reported on the held-out half over 1,000 halvings: mean {dd.mean():+.2f}, 2.5th to 97.5th percentile {lo:+.2f} to {hi:+.2f}  [c12-041]")
print(f"Rebuilt ahead on the other half in {100 * (dd > 0).mean():.1f}% of 1,000 halvings (fixed threshold 95%)  [c12-041]")

# ------------------------------------------------------------------ 6. video-disjoint ensemble, max-T over six looks
vd = {"vdB_s101": "vd_xmodal_31205__prior__seedens_vdB__s101_test__hyps_LP=0.9", "V6": "t2_vdens_10674__dumps_V6_test__hyps_LP=1.0",
      "V3A": "t2_vdens_10674__dumps_V3A_test__hyps_LP=0.9", "V3B": "t2_vdens_10674__dumps_V3B_test__hyps_LP=0.8",
      "vdB_ens5": "vd_xmodal_31205__prior__seedens_vdB__ens5_test__hyps_LP=0.8", "VD5": "vd_xmodal_31205__decode__VD5_test__hyps_LP=0.9"}
sysd, refv = {}, None
for k, f in vd.items():
    h, rr = load_dump(D / f"{f}.jsonl")
    refv = refv or rr; assert rr == refv
    sysd[k] = h
for k, want in [("vdB_s101", 7.287), ("VD5", 7.841)]:
    got = corpus_bleu4(sysd[k], refv)
    print(f"GATE {k} video-disjoint test BLEU-4 {got:.3f} (recorded {want})")
    if abs(got - want) > 0.0015:
        fail(k)
SD = {k: stats(h, refv) for k, h in sysd.items()}
N = len(refv); rng = np.random.default_rng(0); draws = rng.integers(0, N, size=(2000, N))
others = [k for k in SD if k != "vdB_s101"]
obs = np.array([sc(SD[k]) - sc(SD["vdB_s101"]) for k in others])
boot = np.array([[sc(SD[k][i]) - sc(SD["vdB_s101"][i]) for k in others] for i in draws])
se, per, sim, q = simultaneous_band(boot, obs)
j = others.index("VD5")
ro = open(D / "vd_xmodal_31205__g1_readout.txt").read()
tabrows = {m.group(1): (float(m.group(3)), float(m.group(4))) for m in re.finditer(r"^\s+(\w+)\s+(\d\.\d+)\s+(\d\.\d{3})\s+(\d\.\d{3})\s+single\s*$", ro, re.M)}
sel = max(tabrows, key=lambda k: tabrows[k][0])
bl = re.search(r"Baseline \(best-on-vd_val single among VD5's members\): (\w+)", ro).group(1)
best_test = max(tabrows, key=lambda k: tabrows[k][1])
print(f"\nSingle models among VD5's members, validation (video-disjoint) / test BLEU-4, read from the saved job output g1_readout.txt: " + "; ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in tabrows.items()))
print(f"Highest validation score among them: {sel}; the baseline named in the job output: {bl}; the baseline used here: vdB_s101; all equal: {sel == bl == 'vdB_s101'}  [c18-005]")
print(f"(Highest test score among them: {best_test} {tabrows[best_test][1]:.2f}; the baseline is the validation-selected single model, not the test-best one.)")
print(f"\nVideo-disjoint ensemble VD5 {corpus_bleu4(sysd['VD5'], refv):.2f} against {corpus_bleu4(sysd['vdB_s101'], refv):.2f} for the validation-selected single model {bl}  [c18-005]")
print(f"Difference {obs[j]:+.2f}; simultaneous 95% interval (max-T over {len(others)} looks, 2,000 resamples) {obs[j] - sim[j]:+.2f} to {obs[j] + sim[j]:+.2f}  [c18-005]")
