"""Wave 16 (rest-cut stitching): recompute the dissertation's rest-cut numbers from saved outputs.
Run: cd src && python3 ../results/checks/rest_cut_stitching/check.py
"""
import csv, json, math, random, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import sacrebleu
from scipy import stats
from slt.metrics import corpus_bleu4, detokenize

SEEDS = [42, 101, 202, 303, 404]
ARMS = ["A", "R08", "R10"]


def r2(x, sign=False):
    s = str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))   # round half up, as the text does
    if s in ("-0.00",): s = "0.00"
    return ("+" + s if sign and not s.startswith("-") else s)


def fail(msg):
    print("GATE FAILED:", msg); sys.exit(1)


# ---- references: the official manifests that `make data` writes (the saved decodes' own references equal them, checked on the original files)
refs = {}
for sp in ("val", "test"):
    p = ROOT / "comm_run" / "manifests" / f"{sp}.tsv"
    if not p.exists():
        fail(f"{p} missing: run `make data` first")
    refs[sp] = [detokenize(l.rstrip("\n").split("\t")[1]) for l in open(p, encoding="utf-8")]
    if len(refs[sp]) != 800 or refs[sp][0].startswith("sha1:"):
        fail(f"{sp} references are not restored")

# ---- Part 1: BLEU-4 from the saved beam-5 decodes (best checkpoint, 13a, detokenised)
bleu, bp = {}, {}
for arm in ARMS:
    for sp in ("val", "test"):
        d = json.load(open(HERE / f"hyps_{arm}_{sp}.json"))
        for s in SEEDS:
            h = [detokenize(x) for x in d[f"s{s}"]["beam_hyps"]]
            b = corpus_bleu4(h, refs[sp])
            if abs(b - d[f"s{s}"]["recorded_bleu"]) > 1e-6:
                fail(f"{arm} s{s} {sp}: re-scored {b} != recorded {d[f's{s}']['recorded_bleu']}")
            bleu[arm, s, sp] = b
            bp[arm, s, sp] = sacrebleu.corpus_bleu(h, [refs[sp]], tokenize="13a").bp
print("gate 1 OK: all 30 beam-5 decodes (3 arms x 5 seeds x val/test) re-score to the BLEU-4 recorded by the training jobs")
recA = [round(bleu["A", s, "test"], 3) for s in SEEDS]
if recA != [11.711, 10.745, 10.411, 11.305, 10.869]:
    fail(f"control test scores {recA} differ from the wave-5 control values recorded in the job log")
print("gate 2 OK: control (A) test BLEU-4 per seed =", recA, "= wave-5 control")


def tint(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(.975, n - 1)
    return m, m - q * se, m + q * se, m / se


print("\n=== Five-seed paired differences against the control A (beam 5, best checkpoint) ===")
P = {}
for arm in ("R08", "R10"):
    dt = [bleu[arm, s, "test"] - bleu["A", s, "test"] for s in SEEDS]
    dv = [bleu[arm, s, "val"] - bleu["A", s, "val"] for s in SEEDS]
    adj = [bleu[arm, s, "test"] * bp["A", s, "test"] / bp[arm, s, "test"] - bleu["A", s, "test"] for s in SEEDS]
    m, lo, hi, t = tint(dt); p = stats.t.sf(t, len(SEEDS) - 1); P[arm] = p
    mv, lv, hv, _ = tint(dv)
    print(f"{arm} - A  test      mean {r2(m, 1)} ({r2(lo, 1)} to {r2(hi, 1)}), higher in {sum(x > 0 for x in dt)} of 5 seeds, one-sided t {t:.3f}, p {p:.4f}")
    print(f"{arm} - A  validation mean {r2(mv, 1)} ({r2(lv, 1)} to {r2(hv, 1)}), higher in {sum(x > 0 for x in dv)} of 5 seeds; Stage-1 (seeds 42/101/202) mean {r2(np.mean(dv[:3]), 1)}")
    print(f"{arm} - A  length-adjusted test gain {r2(np.mean(adj), 1)} (per seed {', '.join(r2(x, 1) for x in adj)}); fresh seeds 303/404 test {r2(dt[3], 1)}, {r2(dt[4], 1)}")
    print(f"{arm} mean test/val {r2(np.mean([bleu[arm, s, 'test'] for s in SEEDS]))}/{r2(np.mean([bleu[arm, s, 'val'] for s in SEEDS]))}"
          f"  A {r2(np.mean([bleu['A', s, 'test'] for s in SEEDS]))}/{r2(np.mean([bleu['A', s, 'val'] for s in SEEDS]))}")
order = sorted(P, key=P.get)
rej0 = P[order[0]] < 0.025; rej1 = rej0 and P[order[1]] < 0.05
print(f"Holm (m = 2): {order[0]} p {P[order[0]]:.4f} vs 0.025 -> {'reject' if rej0 else 'retain'}; {order[1]} p {P[order[1]]:.4f} vs 0.05 -> {'reject' if rej1 else 'retain'}")

# ---- Part 2: how much of the stitched clips' rest-pause defect there is (per-clip audit, 2,000 clips per group)
rows = list(csv.DictReader(open(HERE / "audit_per_clip.csv")))
grp = {}
for r in rows: grp.setdefault(r["group"], {})[int(r["index"])] = float(r["interior_runs"])
ok = json.load(open(HERE / "audit_ids_runs.json"))
if [grp["stitched"][i] for i in ok["stitched_ids"] if i in grp["stitched"]] != [v for i, v in zip(ok["stitched_ids"], ok["stitched_runs"]) if v is not None]:
    fail("audit table and audit marker disagree on the stitched clips")
rng = np.random.default_rng(16); x = np.array(list(grp["stitched"].values()))
bm = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)])
ci = (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))
if not (abs(ci[0] - ok["ci95"][0]) < 5e-4 and abs(ci[1] - ok["ci95"][1]) < 5e-4):
    fail(f"bootstrap interval {ci} != audit's {ok['ci95']}")
print("\ngate 3 OK: the audit's bootstrap interval of the stitched mean is reproduced:", f"[{ci[0]:.3f}, {ci[1]:.3f}]")
m = {g: np.mean(list(v.values())) for g, v in grp.items()}
print(f"=== Interior rest runs (both hands down between signs) per clip, 2,000 clips per group ===")
print(f"stitched clips {r2(m['stitched'])} (exact {m['stitched']:.4f}); real clips, tracked keypoints {r2(m['real_tracked'])} (exact {m['real_tracked']:.4f}); real clips as extracted {r2(m['real_as_trained'])} (exact {m['real_as_trained']:.4f})")

# ---- Part 3: depth of the cut, from the per-segment cut table of each arm (94,428 stitched signs)
print("\n=== Depth of the cut (per-segment table of the cut arms; 94,428 stitched signs) ===")
for arm in ("R08", "R10"):
    n = fr = fr_true = lead = trail = 0
    for r in csv.DictReader(open(HERE / f"cut_segments_{arm}.csv")):
        l, t, k, c = int(r["lead"]), int(r["trail"]), int(r["trimmed_len"]), int(r["count"])
        n += c; lead += l * c; trail += t * c
        fr += c * (l + t) / max(l + t + k, 1)       # the job's statistic: cut / (cut + trimmed length)
        fr_true += c * (l + t) / max(k, 1)          # cut / trimmed length
    rec = json.load(open(HERE / f"PREP_OK_{arm}.json"))
    assert n == rec["segments"] and abs(fr / n - rec["mean_removed_frac"]) < 1e-12, (arm, n, fr / n, rec["mean_removed_frac"])
    print(f"{arm}: frames removed {100 * fr / n:.1f}% as recorded by the build check (cut / (cut + trimmed length); matches its saved value exactly); "
          f"cut / trimmed length alone: {100 * fr_true / n:.1f}%; mean frames cut per sign: start {lead / n:.2f}, end {trail / n:.2f}")
    print(f"{arm}: interior rest runs left per clip (500 clips; read from the saved job summary PREP_OK_{arm}.json) {r2(rec['manip_R_mean'])}")
# gate: the same 500 clips for the uncut arm reproduce the audit
ids = ok["stitched_ids"]; runs = dict(zip(ids, ok["stitched_runs"]))
pick = sorted(random.Random(162).sample(ids, 500)); vals = [runs[i] for i in pick if runs[i] is not None]
rec = json.load(open(HERE / "PREP_OK_R08.json"))
if abs(np.mean(vals) - rec["manip_A_mean"]) > 5e-4:
    fail(f"uncut-arm manipulation-check mean {np.mean(vals)} != saved {rec['manip_A_mean']}")
print(f"gate 4 OK: the uncut arm on the same 500 clips gives {np.mean(vals):.3f} rest runs per clip (the saved build check: {rec['manip_A_mean']})")
