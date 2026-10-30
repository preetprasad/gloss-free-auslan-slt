"""Recomputes the dissertation's numbers on the 69-keypoint input, the graph front ends and the tracked-keypoint check.

Everything is re-scored from saved model hypotheses (data/) against the reference sentences restored by `make data`
(comm_run/manifests); only the parameter counts, the input widths and the clip counts of the pose check are read from saved job files.
Run from src/:  python3 ../results/checks/keypoints_69_input/check.py
"""
import json, math, statistics as st, sys, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(1, str(ROOT / "src" / "tools"))
warnings.filterwarnings("ignore")
import sacrebleu                                              # noqa: E402
from scipy.stats import t as Tdist                             # noqa: E402
from slt.metrics import corpus_bleu4, detokenize               # noqa: E402
from seed_decision import interval                              # noqa: E402  (seed-level 95% t interval, uncorrected)

DATA = HERE / "data"
MAN = ROOT / "comm_run" / "manifests"
if not (MAN / "val.tsv").exists():
    sys.exit("comm_run/manifests/*.tsv missing: run `make data` first")


def manifest(name):
    return [l.rstrip("\n").split("\t", 1)[1] for l in open(MAN / f"{name}.tsv", encoding="utf-8") if "\t" in l]


REF = {"val": manifest("val"), "test": manifest("test")}
TRAIN = {" ".join(t.lower().split()) for t in manifest("train")}
UNSEEN = {sp: [i for i, r in enumerate(REF[sp]) if " ".join(r.lower().split()) not in TRAIN] for sp in REF}
DREF = {sp: [detokenize(r) for r in REF[sp]] for sp in REF}


def load(name):
    return json.load(open(DATA / name))


def bleu(hyps, sp, idx=None):
    H = [detokenize(h) for h in hyps]
    R = DREF[sp]
    if idx is not None:
        H, R = [H[i] for i in idx], [R[i] for i in idx]
    return corpus_bleu4(H, R, "13a")


def bp_geo(hyps, sp):
    b = sacrebleu.corpus_bleu([detokenize(h) for h in hyps], [DREF[sp]], tokenize="13a")
    return b.bp, b.score / b.bp


# ---------------------------------------------------------------- the runs and the gate
S3 = (42, 101, 202)
R = {}                                                          # (arm, seed) -> {val, test, set}
bad = []
checks = 0
for arm, seeds in (("ctrl", S3), ("C133", S3), ("F69", S3), ("ctrlB", S3), ("F69B", S3), ("P69TC", S3), ("SG", S3), ("ctrl", (303, 404)), ("F69", (303, 404))):
    for s in seeds:
        fs = sorted(DATA.glob(f"hyps_{arm}_s{s}_*.json"))
        assert len(fs) == 1, (arm, s, fs)
        j = json.load(open(fs[0]))
        row = {"set": j["job_set"]}
        for sp in ("val", "test"):
            row[sp] = j[sp]
            row[f"b_{sp}"] = bleu(j[sp], sp)
            checks += 1
            if abs(row[f"b_{sp}"] - j[f"recorded_{sp}"]) > 1e-6:
                bad.append((arm, s, sp, row[f"b_{sp}"], j[f"recorded_{sp}"]))
        R[arm, s] = row
SW = {}                                                         # (arm, seed) -> {lp: val BLEU}
for f in sorted(DATA.glob("sweep_val_*.json")):
    j = json.load(open(f))
    key = (j["arm"], j["seed"])
    SW[key] = {}
    for lp in j["lps"]:
        k = str(lp)
        SW[key][lp] = bleu(j["hyps"][k], "val")
        checks += 1
        if abs(SW[key][lp] - j["recorded"][k]) > 1e-6:
            bad.append(("sweep", key, lp))
    checks += 1
    if j["hyps"]["0.6"] != R[key]["val"]:                       # CPU sweep at LP 0.6 = the GPU decode of the run
        bad.append(("sweep LP 0.6 differs from the run", key))
W24 = {}
for pose in ("c0", "v2"):
    for arm in ("ctrl", "F69"):
        for s in S3:
            j = load(f"trackpose_{pose}_{arm}_s{s}.json")
            W24[pose, arm, s] = bleu(j["val"], "val")
            checks += 1
            if abs(W24[pose, arm, s] - j["recorded_val"]) > 1e-6:
                bad.append(("w24", pose, arm, s))
# recorded facts of the jobs used as extra gates: F69B seed 42 is F69 seed 42 bit for bit, and its sweep gave the same BLEU
checks += 1
if R["F69B", 42]["val"] != R["F69", 42]["val"] or R["F69B", 42]["test"] != R["F69", 42]["test"]:
    bad.append("F69B seed 42 != F69 seed 42")
print(f"GATE {'PASSED' if not bad else 'FAILED'}: {checks - len(bad)} of {checks} values reproduced "
      f"(BLEU-4 of every saved decode = the value its job recorded, to 1e-6; sweep at LP 0.6 = GPU decode; F69B seed 42 = F69 seed 42)")
print(f"references: validation {len(REF['val'])} clips, test {len(REF['test'])}; unseen-target clips {len(UNSEEN['val'])} (validation), {len(UNSEEN['test'])} (test)")
if bad:
    print("FAILED:", bad[:5]); sys.exit(1)


# ---------------------------------------------------------------- helpers
def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def diffs(arm, ctrl, sp, seeds=S3, key=None):
    k = key or f"b_{sp}"
    return [R[arm, s][k] - R[ctrl, s][k] for s in seeds]


def unseen_val(arm, ctrl, seeds=S3, sp="val"):
    return [bleu(R[arm, s][sp], sp, UNSEEN[sp]) - bleu(R[ctrl, s][sp], sp, UNSEEN[sp]) for s in seeds]


def line(label, d, nd=2):
    m, lo, hi = interval(d)
    pos = sum(x > 0 for x in d)
    seeds_txt = " ".join(sg(x, 3) for x in d)
    iv = f"95% t [{sg(lo, nd)}, {sg(hi, nd)}]" if len(d) > 1 else ""
    print(f"  {label:34s} per seed {seeds_txt} | mean {sg(m, nd)} {iv} | higher in {pos} of {len(d)}")
    return m, lo, hi


def lp_star(arm, ctrl_arm=None):
    """Arm-level validation-chosen length penalty: highest 3-seed mean validation BLEU-4 (ties: closest to 0.6, then the lower)."""
    lps = sorted(SW[arm, 42])
    m = {lp: st.mean(SW[arm, s][lp] for s in S3) for lp in lps}
    return min(lps, key=lambda lp: (-round(m[lp], 9), abs(lp - 0.6), lp))


def length_matched(arm, ctrl):
    la, lc = lp_star(arm), lp_star(ctrl)
    d = [SW[arm, s][la] - SW[ctrl, s][lc] for s in S3]
    return la, lc, d


def contrast(arm, ctrl, tag):
    print(f"-- {arm} - {ctrl}  (beam 5, best.pt, length penalty 0.6; job set {R[arm, 42]['set']})")
    v = line("validation", diffs(arm, ctrl, "val"))
    t = line("test", diffs(arm, ctrl, "test"))
    return v, t


print("\n=== Settings ===")
fd = load("feat_dim.json")
pm = load("params.json")
print(f"input width per frame (x, y, confidence): 69-keypoint arms {fd['F69_s42']} = 3 x {fd['F69_s42'] // 3} keypoints; 133-keypoint arms {fd['ctrl_s42']} = 3 x {fd['ctrl_s42'] // 3}")
print(f"  every 69-keypoint run has width {sorted({v for k, v in fd.items() if k.startswith(('F69', 'P69TC', 'SG'))})[0]}; every 133-keypoint run {sorted({v for k, v in fd.items() if k.startswith(('ctrl', 'C133'))})[0]}")
sets_a = sorted({R[a, s]["set"] for a in ("ctrl", "C133", "F69") for s in S3})
sets_b = sorted({R[a, s]["set"] for a in ("ctrlB", "F69B", "P69TC", "SG") for s in S3})
print(f"first submission: job set {', '.join(sets_a)}: arms ctrl, C133, F69, seeds {', '.join(map(str, S3))} ({len(S3)} seeds each)")
print(f"second submission: job set {', '.join(sets_b)}: arms ctrlB, F69B, P69TC, SG, seeds {', '.join(map(str, S3))} ({len(S3)} seeds each, with its own control ctrlB)")
print(f"parameters: linear front end on 69 keypoints (F69) {pm['F69']:.2f} M, control {pm['ctrl']:.2f} M; graph front ends (P69TC, SG) {pm['P69TC']:.2f} and {pm['SG']:.2f} M "
      f"-> {pm['P69TC']:.1f} M against {pm['F69']:.1f} M")

print("\n=== First submission (array A): 69-keypoint input and 133-keypoint median scaling against the control ===")
for arm in ("ctrl", "C133", "F69"):
    print(f"  {arm:5s} beam-5 val/test " + "  ".join(f"s{s} {R[arm, s]['b_val']:.3f}/{R[arm, s]['b_test']:.3f}" for s in S3)
          + f" | means {st.mean(R[arm, s]['b_val'] for s in S3):.3f}/{st.mean(R[arm, s]['b_test'] for s in S3):.3f}")
contrast("F69", "ctrl", "A")
line("unseen-target validation", unseen_val("F69", "ctrl"))
la, lc, d = length_matched("F69", "ctrl")
print(f"  length matching (each arm at its own validation-best length penalty: F69 {la}, ctrl {lc}):")
line("validation, F69* - ctrl*", d)
contrast("C133", "ctrl", "A")

print("\n=== Second submission (array B): the same 69-keypoint configuration against its own control ===")
for arm in ("ctrlB", "F69B", "P69TC", "SG"):
    print(f"  {arm:5s} beam-5 val/test " + "  ".join(f"s{s} {R[arm, s]['b_val']:.3f}/{R[arm, s]['b_test']:.3f}" for s in S3)
          + f" | means {st.mean(R[arm, s]['b_val'] for s in S3):.3f}/{st.mean(R[arm, s]['b_test'] for s in S3):.3f}")
contrast("F69B", "ctrlB", "B")
line("unseen-target validation", unseen_val("F69B", "ctrlB"))
la, lc, d = length_matched("F69B", "ctrlB")
print(f"  length matching (F69B at {la}, ctrlB at {lc}):")
line("validation, F69B* - ctrlB*", d)
gap = [R["F69B", s]["b_test"] - R["F69", s]["b_test"] for s in S3]
print("-- same configuration in the two submissions (F69B - F69, raw, not a contrast)")
print("  test per seed " + " ".join(sg(x, 3) for x in gap) + f" | largest absolute difference {max(abs(x) for x in gap):.2f}"
      + " | validation per seed " + " ".join(sg(R["F69B", s]["b_val"] - R["F69", s]["b_val"], 3) for s in S3))

print("\n=== Graph-convolution front ends on the 69 keypoints (array B) ===")
for arm in ("P69TC", "SG"):
    print(f"-- {arm} against the plain linear input layer on the same input (F69B)")
    line("validation", diffs(arm, "F69B", "val"))
    line("test", diffs(arm, "F69B", "test"))
    print(f"-- {arm} against the 133-keypoint control (ctrlB)")
    line("validation", diffs(arm, "ctrlB", "val"))
    line("test", diffs(arm, "ctrlB", "test"))
print("-- skeleton graph over identity graph (SG - P69TC)")
line("validation", diffs("SG", "P69TC", "val"))

print("\n=== Five seeds of the 69-keypoint input against the control ===")
S5 = (42, 101, 202, 303, 404)
sets5 = {s: (R["F69", s]["set"], R["ctrl", s]["set"]) for s in S5}
print("seeds " + ", ".join(map(str, S5)) + "; job sets " + ", ".join(f"{s}: {a}" for s, (a, b) in sets5.items()) + " (the first three from the first submission, the fresh seeds 303 and 404 from the later one)")
dv = diffs("F69", "ctrl", "val", S5)
dt = diffs("F69", "ctrl", "test", S5)
line("validation", dv)
line("test", dt)
print(f"  fresh seeds only (303, 404): validation {sg(dv[3])} {sg(dv[4])}, test {sg(dt[3])} {sg(dt[4])}: positive on both splits in {sum(x > 0 for x in dv[3:]) + sum(x > 0 for x in dt[3:])} of 4")
uv = unseen_val("F69", "ctrl", S5, "val")
ut = unseen_val("F69", "ctrl", S5, "test")
print(f"  unseen-target validation mean {sg(st.mean(uv))}, test mean {sg(st.mean(ut))}")
tt = st.mean(dt) / (st.stdev(dt) / math.sqrt(len(dt)))
print(f"  one-sided t test on test: t = {tt:.3f}, p = {Tdist.sf(tt, len(dt) - 1):.3f}")
# brevity: the arm's n-gram precision at the control's brevity penalty (arithmetic, as in the pre-registered column)
d2b = []
for s in S5:
    bp_c, _ = bp_geo(R["ctrl", s]["test"], "test")
    _, geo_x = bp_geo(R["F69", s]["test"], "test")
    d2b.append(geo_x * bp_c - R["ctrl", s]["b_test"])
print(f"  test gap at the control's brevity penalty {sg(st.mean(d2b), 3)} of the raw {sg(st.mean(dt), 3)} -> longer output accounts for {st.mean(dt) - st.mean(d2b):.2f} of the {st.mean(dt):.2f} test gain")

print("\n=== Tracked keypoints (second tracker) against first-detection keypoints, 69-keypoint input ===")
E = {p: [W24[p, "F69", s] - W24[p, "ctrl", s] for s in S3] for p in ("c0", "v2")}
did = [b - a for a, b in zip(E["c0"], E["v2"])]
line("gain on first-detection (validation)", E["c0"])
line("gain on tracked (validation)", E["v2"])
line("difference of the two gains", did)
g = load("trackpose_pose_gate.json")
print(f"  body scale differs by more than 10% between the two keypoint sets in {g['clips_scale_differs_gt10pct']} of {g['n']} sampled clips "
      f"(read from the saved job summary trackpose_pose_gate.json: the keypoint arrays are not shipped)")
