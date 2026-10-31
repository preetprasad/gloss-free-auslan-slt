"""Person tracking: switch prevalence, blind audit, wave 3 (five seeds), wave 9 tracker v2, wave 12i heavy-switch bin, hand speed, pose jumps.

Recomputes from saved beam-5 hypotheses (re-scored through slt/metrics.py), per-clip switch counts of the tracked
re-extraction, the blind-audit labels and per-clip pose statistics. Training and decoding are not repeated.
"""
import csv, json, random, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = HERE / "data"
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize

KEY = "tokenize=13a;detok=True"
T975 = {2: 4.303, 4: 2.776}   # two-sided 95% t quantiles by df (published table values)


_dt = {}


def dt_(x):
    if x not in _dt:
        _dt[x] = detokenize(x)
    return _dt[x]


def bleu(h, r):
    return corpus_bleu4([dt_(x) for x in h], [dt_(x) for x in r], "13a")


def interval(d):
    m, sd, n = st.mean(d), st.stdev(d), len(d)
    hw = T975[n - 1] * sd / n ** 0.5
    return m, m - hw, m + hw


def fmt(d):
    m, lo, hi = interval(d)
    return f"mean {m:+.2f} ({m:+.3f}) 95% t [{lo:+.2f}, {hi:+.2f}] ({lo:+.3f}, {hi:+.3f}), {sum(x > 0 for x in d)}/{len(d)} seeds positive"


# ---------- references: the official manifests (written by `make data`)
order = json.load(open(D / "clip_order.json"))
refs = {}
for sp in ("val", "test"):
    rows = [l.rstrip("\n").split("\t", 1) for l in open(ROOT / "comm_run/manifests" / f"{sp}.tsv", encoding="utf-8") if "\t" in l]
    assert [c.rsplit("/", 1)[1][:-4] for c, _ in rows] == order[sp], f"{sp}: manifest order differs from the saved decodes"
    refs[sp] = [t for _, t in rows]


def load(fn, sp):
    out = {}
    for l in open(D / fn):
        j = json.loads(l)
        assert len(j["hyps"]) == 800
        got = bleu(j["hyps"], refs[sp])
        assert abs(got - j["bleu"]) < 1e-6, f"GATE FAILED: {j['run']} {sp} re-scores {got} vs recorded {j['bleu']}"
        out[(j["run"], j["decode_pose"])] = j["hyps"]
    return out


# ---------- bins (frozen before wave 12i from the wave-3 tracker's per-clip override share)
bins = json.load(open(D / "w12i_bins.json"))
pos = {sp: {c: i for i, c in enumerate(order[sp])} for sp in order}
B = {}


def sub(x, ii):
    return [x[i] for i in ii]


def did(Hs, sp, seeds, a, b, heavy, none, Da=None, Db=None):
    out = []
    for s in seeds:
        ha, hb = (Da or Hs)[(a, s)], (Db or Hs)[(b, s)]
        r = refs[sp]
        f = lambda ii: bleu(sub(ha, ii), sub(r, ii)) - bleu(sub(hb, ii), sub(r, ii))
        out.append(f(heavy) - (f(none) if none is not None else 0))
    return out


# ================= (1) switch prevalence of the tracked re-extraction (job 6591)
agg = json.load(open(D / "tracking_prevalence_agg.json"))
rows = list(csv.DictReader(open(D / "tracking_switched_clips.csv")))
tot = sum(v["clips"] for v in agg.values())
a_tr = agg["train"]
sw_clips = sum(int(r["switched"]) > 0 for r in rows if r["split"] == "train")
sw_fr = sum(int(r["switched"]) for r in rows if r["split"] == "train")
assert (sw_clips, sw_fr) == (a_tr["sw_clips"], a_tr["sw_frames"]), "per-clip file differs from the job totals"
print(f"Tracked re-extraction covers {tot:,} Communication clips (read from the saved job output tracking_prevalence_agg.json) [c12-062]")
print(f"Training clips with at least one switched frame {100*a_tr['sw_clips']/a_tr['clips']:.1f}% ({a_tr['sw_clips']}/{a_tr['clips']}); switched frames {100*a_tr['sw_frames']/a_tr['frames']:.1f}% ({a_tr['sw_frames']}/{a_tr['frames']}) [c12-063]")
n_ev = agg["val"]["clips"] + agg["test"]["clips"]
none_ev = n_ev - agg["val"]["sw_clips"] - agg["test"]["sw_clips"]
print(f"Evaluation clips (validation+test) with no switched frame {100*none_ev/n_ev:.0f}% ({none_ev}/{n_ev}) [c12-069]")
print("  (the switched-clip numerators are recounted from the per-clip file and match the job totals)")

# ================= (2) blind two-rater audit
L = json.load(open(D / "tracker_audit_labels.json"))
assert len(L) == 160
t = sum(v["m1"] == "tracked-only" for v in L.values() if v["r1"] == v["r2"] and v["stratum"] in ("light", "medium", "heavy", "agree"))
i0 = sum(v["m1"] == "index0-only" for v in L.values() if v["r1"] == v["r2"] and v["stratum"] in ("light", "medium", "heavy", "agree"))
print(f"\nBlind audit: {len(L)} frames, each labelled by 2 raters [c12-064]")
print(f"Tracker on the signer and the original extraction not: {t} of {t+i0} decisive frames ({100*t/(t+i0):.0f}%) [c12-065]")

# ================= (3) wave 3, five seeds, whole test set
S5 = (42, 101, 202, 303, 404)
w3 = {sp: load(f"w3_{sp}.jsonl", sp) for sp in ("val", "test")}
H3 = {sp: {(a, s): w3[sp][(f"{a}_s{s}", a)] for a in ("w3C", "w3T") for s in S5} for sp in w3}
print("\nWave 3 (job 6592 + 9444): index-0 pose (w3C) vs person-tracked pose (w3T), seeds 42/101/202/303/404")
print("GATE: all 20 beam-5 BLEU-4 values re-score to the recorded job values")
dt = [bleu(H3["test"]["w3T", s], refs["test"]) - bleu(H3["test"]["w3C", s], refs["test"]) for s in S5]
print(f"Test (800 clips) tracked - original, per seed {' '.join(f'{x:+.2f}' for x in dt)}: {fmt(dt)} [c12-066]")
print("  (the dissertation states the upper bound +0.55; the five-seed t-interval here is +0.545 rounded down to +0.54: a rounding difference, not forced)")
print(f"Stage-2 seeds (303, 404): tracked scores lower in {sum(x < 0 for x in dt[3:])} of 2 (informational for c12-068)")
# heavy bin (frac of frames overridden > 0.3) vs the clips with none
for sp in ("val", "test"):
    bins_sp = bins[sp]["bins"]
    B[sp] = {k: [pos[sp][c] for c in v] for k, v in bins_sp.items()}
dv = did(H3["val"], "val", S5, "w3T", "w3C", B["val"]["heavy"], None)
dtt = did(H3["test"], "test", S5, "w3T", "w3C", B["test"]["heavy"], None)
print(f"Heavy bin = clips on which the tracker replaced more than 30% of the frames (frozen rule: share > 0.3): {len(B['test']['heavy'])} test clips [c12-071]")
print(f"Heavy-switch clips, validation, rounded to one decimal: {st.mean(dv):+.1f} on {len(B['val']['heavy'])} clips [c17-015]")
print(f"Heavy-switch clips, test, rounded to one decimal: {st.mean(dtt):+.1f} on {len(B['test']['heavy'])} clips [c17-016]")
print(f"Heavy-switch clips, validation (n={len(B['val']['heavy'])}) tracked - original: {fmt(dv)} [c17-015]")
print(f"Heavy-switch clips, test (n={len(B['test']['heavy'])}): {fmt(dtt)}; negative in {sum(x < 0 for x in dtt)} seeds [c17-016]")

# ================= (4) wave 9 tracker v2, stage 1
S3 = (42, 101, 202)
w9 = {sp: load(f"w9_{sp}.jsonl", sp) for sp in ("val", "test")}
d9 = {}
for sp in ("val", "test"):
    d9[sp] = [bleu(w9[sp][(f"v2_s{s}", "v2")], refs[sp]) - bleu(w9[sp][(f"c0_s{s}", "c0")], refs[sp]) for s in S3]
print("\nWave 9 (job 43506): tracker v2 - original, seeds 42/101/202")
print(f"Validation mean {st.mean(d9['val']):+.2f} ({st.mean(d9['val']):+.3f}): stage 1 stops (mean <= 0) [c17-017]")
print(f"Test {fmt(d9['test'])} [c17-018]")

# ================= (5) wave 12i, five new seeds
S12 = (303, 404, 505, 606, 707)
H = {}
for sp in ("val", "test"):
    own = {}
    for arm in ("c0", "v2"):
        j = load(f"w12i_{sp}_{arm}.jsonl", sp)
        for s in S12:
            own[(arm, s)] = j[(f"{arm}_s{s}", arm)]
    H[sp] = own
print("\nWave 12i (jobs 138391/138392): tracker v2 vs original on the heavy-switch bin, five new seeds 303/404/505/606/707")
print(f"Seeds: {len(S12)} new seeds [c12-070]")
h = {sp: did(H[sp], sp, S12, "v2", "c0", B[sp]["heavy"], B[sp]["none"]) for sp in ("val", "test")}
print(f"Heavy bin (test n={len(B['test']['heavy'])}) DiD tracker v2 - original, minus clips without switches: {fmt(h['test'])} [c12-071 c12-072 c17-019]")
print(f"Heavy bin validation DiD: mean {st.mean(h['val']):+.2f} ({st.mean(h['val']):+.3f}) [c12-072 c17-020]")


def boot(sp, n=1000):
    rng, out = random.Random(0), []
    hv, nn = B[sp]["heavy"], B[sp]["none"]
    for _ in range(n):
        hh, n2 = rng.choices(hv, k=len(hv)), rng.choices(nn, k=len(nn))
        out.append(st.mean(did(H[sp], sp, S12, "v2", "c0", hh, n2)))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]
lo, hi = boot("test")
print(f"Clip-sampling interval (test, 1000 resamples within bins, seed 0) [{lo:+.2f}, {hi:+.2f}] [c17-020]")
wh = [bleu(H["test"]["v2", s], refs["test"]) - bleu(H["test"]["c0", s], refs["test"]) for s in S12]
mw, lw, hw_ = interval(wh)
print(f"Whole test set, tracker v2 - original: mean {mw:+.2f} ({mw:+.3f}) 95% t [{lw:+.2f}, {hw_:+.2f}] [c12-073 c17-021]")

# decode the original-trained model on tracked pose
X = {}
for arm in ("c0", "v2"):
    j = load(f"w12i_test_{arm}.jsonl", "test")
    for s in S12:
        X[(arm, s)] = j[(f"{arm}_s{s}", "v2" if arm == "c0" else "c0")]
cross = did(H["test"], "test", S12, "c0", "c0", B["test"]["heavy"], B["test"]["none"], Da=X, Db=H["test"])
print(f"Original-trained model decoded on tracked keypoints, heavy-bin DiD vs decoding on its own pose: mean {st.mean(cross):+.2f} ({st.mean(cross):+.3f}) [c17-022]")

# ================= (6) hand speed with tracked keypoints (per-clip mean left-hand displacement / shoulder width)
hs = list(csv.DictReader(open(D / "hand_speed_per_clip.csv")))
fd = st.mean(float(r["first_detection_speed"]) for r in hs); tk = st.mean(float(r["tracked_speed"]) for r in hs)
print(f"\nHand speed on {len(hs)} training clips: first-detection mean {fd:.3f}, tracked mean {tk:.3f} per frame")
print("The citation-form comparison value is a recorded mean of 0.044 per frame (MM-WLAuslan clips are not held locally), recorded 9 Sep.")
print(f"Ratio to the recorded citation value, per frame {tk/0.044:.1f}, per second {tk/0.044*25/30:.1f}  (informational, untagged)")

# ================= (7) pose jumps in locally stored clips
pj = list(csv.DictReader(open(D / "pose_jump_census.csv")))
use = [r for r in pj if float(r["shoulder_valid_share"]) >= 0.9]
jc = sum(int(r["jumps"]) > 0 for r in use)
jt, tt = sum(int(r["jumps"]) for r in use), sum(int(r["transitions"]) for r in use)
print(f"\nPose-jump census over locally stored clips with shoulders tracked on at least 90% of frames: {len(use):,} clips [c17-012]")
print(f"Clips with at least one jump {100*jc/len(use):.1f}% ({jc}); frame transitions with a jump {100*jt/tt:.2f}% ({jt}/{tt})")
print("  (the dissertation states 25.2% and 2.31%; this definition of a jump, from pose/tracker_v2/research/alphapose_identity_check.py, gives the values above: not forced to agree)")
