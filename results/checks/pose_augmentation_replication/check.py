"""Pose augmentation replication (job 1215231): 3 paired seeds, aug vs same-seed control, beam-5 official test.
Recomputes BLEU-4 from the saved hypotheses, validation/greedy from saved job outputs, seed-level t-interval,
content precision (stop list = train_ctc.STOP, the training script's own content-word list), and the recipe settings from config.py / the job script."""
import ast, json, re, sys
from pathlib import Path
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
r2 = lambda x: str(Decimal(repr(float(x))).quantize(Decimal("0.01"), ROUND_HALF_UP))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/tools"))
from slt.metrics import detokenize, corpus_bleu4
# the training script's own stop list (read from its source so that torch-heavy train_ctc.py need not be imported)
_tree = ast.parse((ROOT / "src/train_ctc.py").read_text())
_node = next(n.value for n in _tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "STOP")
STOP = set(eval(compile(ast.Expression(_node), "STOP", "eval"), {"set": set}))
assert len(STOP) > 60 and "the" in STOP
STOP75 = STOP  # train_ctc.STOP (75 words), used only for the labelled second line
from stitch_lexicon import STOP  # the 71-word list the dissertation's method chapter defines as content words
assert len(STOP) == 71
from collections import Counter

D = ROOT / "results/pose_augmentation_decodes"
SEEDS = (42, 101, 202)
def load(arm):
    rows = sorted((json.loads(l) for l in open(D / arm / "hyps_beam5.jsonl")), key=lambda r: r["idx"])
    return [r["hyp_beam"] for r in rows], [r["ref"] for r in rows]
def prog(arm): return json.load(open(D / arm / "progress.json"))["best"]["bleu4"]
def greedy(arm): return json.load(open(D / arm / "beam5_test.json"))["greedy"]["metrics"]["tokenize=13a;detok=True"]
def cprec(h, r, STOP=STOP):
    hit = nh = 0
    for a, b in zip(h, r):
        ca = [w for w in a.lower().split() if w.isalpha() and w not in STOP]
        cb = [w for w in b.lower().split() if w.isalpha() and w not in STOP]
        hit += sum((Counter(ca) & Counter(cb)).values()); nh += len(ca)
    return 100 * hit / max(nh, 1)

man = ROOT / "comm_run/manifests/test.tsv"
B, V, G, P, P75 = {}, {}, {}, {}, {}
for s in SEEDS:
    for k in ("ctrl", "aug"):
        h, r = load(f"pa_{k}_s{s}")
        if man.exists():
            m = [l.rstrip("\n").split("\t")[1].strip() for l in open(man) if "\t" in l]
            assert m == r, "references differ from the test manifest"
        B[k, s] = corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")
        V[k, s] = prog(f"pa_{k}_s{s}"); G[k, s] = greedy(f"pa_{k}_s{s}"); P[k, s] = cprec(h, r); P75[k, s] = cprec(h, r, STOP75)
# GATE: recorded beam-5 test scores in the job output (pose-aug-1215231.out)
rec = json.load(open(D / "pa_aug_s42/beam5_test.json"))["beam_result"]["metrics"]["tokenize=13a;detok=True"]
print(f"GATE recomputed aug s42 {B['aug',42]:.4f} vs recorded {rec:.4f} (read from the saved job output beam5_test.json)")
if abs(B["aug", 42] - rec) > 1e-6: sys.exit("GATE FAILED")

print("\nBeam-5 official test BLEU-4, aug vs its same-seed control (recomputed from the saved hypotheses, 800 clips)")
dl = []
for s in SEEDS:
    d = B["aug", s] - B["ctrl", s]; dl.append(d)
    print(f"  seed {s}: {B['ctrl',s]:.2f} -> {B['aug',s]:.2f}  ({d:+.2f})")
print(f"Test BLEU-4 at beam width 5 rose in all {sum(x>0 for x in dl)} (three) seeds, aug above its same-seed control in each  [c09-074]")
print(f"seed 42 from {r2(B['ctrl',42])} to {r2(B['aug',42])}; seed 202 from {r2(B['ctrl',202])} to {r2(B['aug',202])}; seed 101 from {r2(B['ctrl',101])} to {r2(B['aug',101])} (unrounded {B['aug',101]:.4f}; the dissertation says 11.38: DISAGREES, a rounding difference of 0.01)  [c09-075]")
mc = np.mean([B["ctrl", s] for s in SEEDS]); ma = np.mean([B["aug", s] for s in SEEDS])
dm = round(ma, 2) - round(mc, 2)
print(f"means {mc:.2f} to {ma:.2f}, {dm:+.2f} (unrounded {np.mean(dl):+.3f}); control mean {mc:.2f}, augmented mean {ma:.2f}  [c09-018 c09-075 c09-076]")
print(f"+ pose augmentation (3 seeds) {ma:.2f}, {dm:+.2f} over its concurrent controls ({mc:.2f}), {sum(x>0 for x in dl)} of 3 seeds  [c09-018]")
from scipy.stats import t as tdist
sd = np.std(dl, ddof=1); tc = tdist.ppf(0.975, 2); h = tc * sd / np.sqrt(3)
print(f"seed-level 95% t-interval of the paired differences: {np.mean(dl)-h:+.2f} to {np.mean(dl)+h:+.2f}  (t={tc:.3f}, df=2; includes zero: {np.mean(dl)-h<0<np.mean(dl)+h})  [c09-077 c14-026]")
print(f"95% t-interval over the seeds {np.mean(dl)-h:+.2f} to {np.mean(dl)+h:+.2f}  [c14-026]")
vd = [V["aug", s] - V["ctrl", s] for s in SEEDS]; gd = [G["aug", s] - G["ctrl", s] for s in SEEDS]
print(f"validation BLEU-4 aug minus ctrl per seed (read from the saved job output progress.json): {' '.join(f'{x:+.3f}' for x in vd)}; rose in {sum(x>0 for x in vd)} of 3  [c09-076]")
print(f"greedy test BLEU-4 aug minus ctrl per seed (read from the saved job output beam5_test.json): {' '.join(f'{x:+.3f}' for x in gd)}; rose in {sum(x>0 for x in gd)} of 3  [c09-076]")
print(f"beam-5 test rose in {sum(x>0 for x in dl)} of 3 seeds: the pre-registered three-seed rule is met: {all(x>0 for x in dl) and all(x>0 for x in vd)}  [c09-074 c09-076 c09-077]")
pd = [P["aug", s] - P["ctrl", s] for s in SEEDS]
print(f"Pooled content-word precision of the beam-5 test outputs (stitch_lexicon.STOP, {len(STOP)} words): " + "; ".join(f"seed {s} {P['ctrl',s]:.2f} -> {P['aug',s]:.2f}" for s in SEEDS))
print(f"Content-word precision rise per seed (points): {' '.join(f'{x:+.2f}' for x in pd)}; in every seed: {all(x>0 for x in pd)}")
print(f"range of the rise {min(pd):+.2f} to {max(pd):+.2f} points under the 71-word list; the dissertation says 1.6 to 2.3: DISAGREES (c09-078 not covered)")
pd75 = [P75["aug", s] - P75["ctrl", s] for s in SEEDS]
print(f"second line, labelled: under the 75-word train_ctc.STOP list the rise per seed is {' '.join(f'{x:+.2f}' for x in pd75)} points")

print("\nRecipe (read from src/config.py defaults and the job script)")
cfg = (ROOT / "src/config.py").read_text()
def dflt(n): return float(re.search(rf"{n}: float = ([0-9.]+)", cfg).group(1))
sp, rot, sc = dflt("aug_speed"), dflt("aug_rotate_deg"), dflt("aug_scale")
job = (ROOT / "jobs/slurm/pose_aug.sbatch").read_text()
m = re.search(r"--aug_speed ([0-9.]+) --aug_rotate_deg ([0-9.]+) --aug_scale ([0-9.]+)", job)
assert tuple(map(float, m.groups())) == (sp, rot, sc), "job flags differ from config defaults"
print(f"temporal resampling factor from [{1-sp:.1f}, {1+sp:.1f}] (aug_speed {sp}; job 1215231 passed the same value)  [c06-034]")
print(f"in-plane rotation from [-{rot:.0f}, +{rot:.0f}] degrees (aug_rotate_deg {rot})  [c06-035]")
print(f"x and y scaling from [{1-sc:.1f}, {1+sc:.1f}] (aug_scale {sc})  [c06-036]")

# 1.1:1 stitch ratio without augmentation: beam-5 test BLEU-4 of sc_r111 in the two job logs
print("\nSynthetic-to-real ratio 1.1:1 without augmentation, beam-5 test BLEU-4 (read from the saved job outputs stitch_ratio_job_logs.txt: stitch-scale 1196567 seed 42, stitch-scl2 1199393 seeds 101, 202)")
lines = open(Path(__file__).parent / "stitch_ratio_job_logs.txt").read().splitlines()
job = None; arm = None; grab = False; r111 = {}
for ln in lines:
    if ln.startswith("#####"): job = ln.split()[-1]
    m = re.search(r"=== ARM (sc_r111_s\d+)", ln)
    if m: arm = m.group(1); continue
    if arm and "[beam5]" in ln: grab = True; continue
    if arm and grab and "BLEU-4=" in ln:
        r111[arm] = (job, float(ln.split("BLEU-4=")[1])); arm = None; grab = False
assert sorted(r111) == ["sc_r111_s101", "sc_r111_s202", "sc_r111_s42"], r111
for a, (j, v) in sorted(r111.items()): print(f"  {a} ({j}): {v:.2f}")
m111 = np.mean([v for _, v in r111.values()])
print(f"mean of the three logged aug-free 1.1:1 scores {m111:.2f}  [c09-052]")
print(f"the higher means at 1.1:1 ({m111:.2f}, and {ma:.2f} with pose augmentation)  [c09-052]")

# recipe, read from the augmentation code itself (slt/data.py) and exercised on synthetic keypoints
print("\nRecipe checked against the augmentation code slt/data.py (augment_pose), run on synthetic keypoints")
from slt.data import augment_pose
from config import Config
src = (ROOT / "src/slt/data.py").read_text()
assert "np.linspace(0, T - 1, new_T).round().astype(int)" in src
print("temporal resampling picks frame indices with linspace(0, T-1, new_T).round(), i.e. nearest-index frame selection (source line checked)  [c06-034]")
def run(**kw):
    c = Config(); c.pose_aug = True; c.aug_speed = c.aug_rotate_deg = c.aug_scale = 0.0
    for k, v in kw.items(): setattr(c, k, v)
    return c
T0 = 100; rng = np.random.default_rng(0)
base = rng.normal(size=(T0, 133, 3)).astype(np.float32); base[..., 2] = 1.0; bb = np.ones((T0, 4), np.float32)
ratios = []
for i in range(2000):
    k, _ = augment_pose(base, bb, run(aug_speed=sp), np.random.default_rng(i)); ratios.append(k.shape[0] / T0)
print(f"temporal factor over 2000 draws: min {min(ratios):.2f}, max {max(ratios):.2f}, inside [{1-sp:.1f}, {1+sp:.1f}]: {min(ratios)>=1-sp-0.01 and max(ratios)<=1+sp+0.01}  [c06-034]")
assert min(ratios) >= 1 - sp - 0.01 and max(ratios) <= 1 + sp + 0.01 and max(ratios) > 1 + sp - 0.05 and min(ratios) < 1 - sp + 0.05
ang = []
for i in range(2000):
    k, _ = augment_pose(base, bb, run(aug_rotate_deg=rot), np.random.default_rng(i))
    c0 = base[..., :2].reshape(-1, 2).mean(0); d0 = (base[..., :2] - c0).reshape(-1, 2); d1 = (k[..., :2] - c0).reshape(-1, 2)
    # best-fit rotation angle from the cross and dot products of all point vectors
    ang.append(np.degrees(np.arctan2((d0[:, 0]*d1[:, 1]-d0[:, 1]*d1[:, 0]).sum(), (d0*d1).sum())))
print(f"in-plane rotation over 2000 draws: min {min(ang):+.1f}, max {max(ang):+.1f} degrees (limit +-{rot:.0f})  [c06-035]")
assert max(abs(a) for a in ang) <= rot + 0.1 and max(abs(a) for a in ang) > rot - 1
sx = []
for i in range(2000):
    k, _ = augment_pose(base, bb, run(aug_scale=sc), np.random.default_rng(i))
    c0 = base[..., :2].reshape(-1, 2).mean(0)
    for ax in (0, 1):
        d0 = base[..., ax].ravel() - c0[ax]; d1 = k[..., ax].ravel() - c0[ax]; sx.append((d0*d1).sum() / (d0*d0).sum())
print(f"x and y scale factors over 4000 draws: min {min(sx):.2f}, max {max(sx):.2f} (limit [{1-sc:.1f}, {1+sc:.1f}])  [c06-036]")
assert min(sx) >= 1 - sc - 0.01 and max(sx) <= 1 + sc + 0.01 and min(sx) < 1 - sc + 0.03
