"""Video-disjoint split and VD-CONFIRM (dissertation Methodology, Results 'Video-disjoint', Discussion, Appendix).

Recomputes from saved outputs: split sizes and video counts, reference overlap (29.3%, 886 unseen-reference clips),
the five-seed beam-5 comparison from the per-seed job summaries, and the six-model ensemble against the best single
model from their saved test decodes (BLEU-4 and a paired bootstrap, 2,000 resamples, seed 0).
"""
import csv, json, re, statistics, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize

D = HERE / "data"
ids = list(csv.DictReader(open(D / "vd_split_ids.csv")))
by = {sp: [r["clip"] for r in ids if r["split"] == sp] for sp in ("train", "val", "test")}
vids = {sp: {re.match(r"(video_\d+)_", c).group(1) for c in by[sp]} for sp in by}
assert not (vids["train"] & vids["test"]) and not (vids["train"] & vids["val"]) and not (vids["val"] & vids["test"])
print(f"video-disjoint split: train {len(by['train']):,} clips from {len(vids['train'])} videos, "
      f"val {len(by['val']):,} from {len(vids['val'])}, test {len(by['test']):,} from {len(vids['test'])}; no video shared  [c07-061 c11-070]")

# ---- GATE: seed-level comparison reproduces the recorded decision file
T975 = {4: 2.776}
rows = [json.load(open(D / f"summary_s{s}.json")) for s in (42, 101, 202, 303, 404)]
def arm(a, k): return [r[a][k] for r in rows]
d = [b - a for a, b in zip(arm("vdA", "beam5_test"), arm("vdB", "beam5_test"))]
m = statistics.mean(d); hw = T975[4] * statistics.stdev(d) / 5 ** .5
rec = open(D / "seed_decision_5seeds.txt").read()
want = f"mean {m:+.3f} | 95% t [{m-hw:+.3f}, {m+hw:+.3f}]"
print(f"GATE recorded job read-out (seed_decision_5seeds.txt) contains '{want}': {want in rec}")
if want not in rec:
    sys.exit("GATE FAILED")

n = len(rows)
print(f"\nVD-CONFIRM: {n} seeds, read from the saved job outputs summary_s*.json (original job 6596 + 7965)  [c18-018 c07-062 c11-073]")
mA, mB = statistics.mean(arm("vdA", "beam5_test")), statistics.mean(arm("vdB", "beam5_test"))
print(f"beam-5 test BLEU-4, mean over {n} seeds: plain baseline {mA:.2f} ({mA:.3f}), adopted recipe {mB:.2f} ({mB:.3f})  [c11-073]")
print(f"difference {m:+.2f} ({m:+.3f}); positive in {sum(x > 0 for x in d)} of {n} seeds  [c11-073]")
print(f"seed-level 95% t-interval {m-hw:+.2f} to {m+hw:+.2f}  [c11-073]")

# ---- references and overlap (needs make data)
man = {}
for sp in ("train", "val", "test"):
    for l in open(ROOT / "comm_run" / "manifests" / f"{sp}.tsv"):
        a, b = l.rstrip("\n").split("\t"); man[Path(a).stem] = b.strip()
norm = lambda s: " ".join(s.lower().split())
train_txt = {norm(man[c]) for c in by["train"] if c in man}
V6 = sorted((json.loads(l) for l in open(D / "V6_test_LP1.0.jsonl")), key=lambda r: r["idx"])
B = sorted((json.loads(l) for l in open(D / "vdB_s101_test_LP0.9.jsonl")), key=lambda r: r["idx"])
refs = [r["ref"] for r in V6]
assert refs == [r["ref"] for r in B] and len(refs) == len(by["test"])
assert all(norm(man[c]) == norm(r) for c, r in zip(by["test"], refs)), "decode references differ from the split manifest"
print("decode references equal the split's test manifest in order: 1254/1254")
seen = np.array([norm(r) in train_txt for r in refs])
print(f"\nreferences occurring verbatim in the video-disjoint training text: {int(seen.sum())} of {len(refs)} test clips = {100*seen.mean():.1f}%  [c11-075]")
print(f"test clips whose reference never occurs in that training text: {int((~seen).sum())}  [c11-076]")

# ---- six-model ensemble vs best single
hv = [detokenize(r["hyp_beam"]) for r in V6]; hb = [detokenize(r["hyp_beam"]) for r in B]; R = [detokenize(r) for r in refs]
bv, bb = corpus_bleu4(hv, R, "13a"), corpus_bleu4(hb, R, "13a")
print(f"GATE recorded 7.123 (ensemble) / 7.287 (best single): {bv:.3f} / {bb:.3f}")
if round(bv, 3) != 7.123 or round(bb, 3) != 7.287:
    sys.exit("GATE FAILED")
mem = re.search(r"V6\s+\S+\s+\S+\s+\S+\s+(.*?)\s*\(t2", open(D / "v6_members_line.txt").read())
nmem = sum(len(g.split("/")) for g in re.findall(r"vd[AB] s(\d+(?:/\d+)*)", mem.group(1))) if mem else 0
words = {6: "six"}
print(f"ensemble members (read from the saved read-out line v6_members_line.txt): {nmem} ({words.get(nmem)})  [c14-035 c11-078]")
print(f"ensemble test BLEU-4 {bv:.2f} ({bv:.3f}); best single model {bb:.2f} ({bb:.3f})  [c14-035 c11-078]")
n_ = len(R); rng = np.random.default_rng(0)
draws = rng.integers(0, n_, size=(2000, n_)); dl = np.empty(2000)
for k, ii in enumerate(draws):
    Rk = [R[i] for i in ii]
    dl[k] = corpus_bleu4([hv[i] for i in ii], Rk, "13a") - corpus_bleu4([hb[i] for i in ii], Rk, "13a")
lo, hi = np.percentile(dl, [2.5, 97.5])
from decimal import Decimal, ROUND_HALF_UP
h2 = lambda x: Decimal(f"{x:.3f}").quantize(Decimal("0.01"), ROUND_HALF_UP)  # round the 3-decimal value half up, as the dissertation does
print(f"difference {bv-bb:+.2f} ({bv-bb:+.3f}); 95% paired-bootstrap interval {lo:+.3f} to {hi:+.3f}, i.e. {h2(lo):+} to {h2(hi):+} (2,000 resamples, seed 0)  [c11-078]")

# ---- the "best single model" is chosen on validation, not on test
rec_singles = {}
for l in open(D / "g1_singles_lines.txt"):
    n, lp, v, t, _ = l.split()
    rec_singles[n] = (lp, float(v), float(t))
sc = {}
for n, (lp, rv, rt) in rec_singles.items():
    s = n.split("_s")[1]
    val = sorted((json.loads(x) for x in open(D / f"vdB_s{s}_val_LP{lp}.jsonl")), key=lambda r: r["idx"])
    tst = sorted((json.loads(x) for x in open(D / f"vdB_s{s}_test_LP{lp}.jsonl")), key=lambda r: r["idx"])
    assert len(val) == len(by["val"]) and len(tst) == len(by["test"])
    assert all(norm(man[c]) == norm(r["ref"]) for c, r in zip(by["val"], val)), "validation references differ from the split manifest"
    bvl = corpus_bleu4([detokenize(r["hyp_beam"]) for r in val], [detokenize(r["ref"]) for r in val], "13a")
    btt = corpus_bleu4([detokenize(r["hyp_beam"]) for r in tst], [detokenize(r["ref"]) for r in tst], "13a")
    sc[n] = (bvl, btt)
    print(f"GATE recorded {n} at LP {lp}: validation {rv:.3f}, test {rt:.3f}; recomputed {bvl:.3f} / {btt:.3f}")
    if round(bvl, 3) != rv or round(btt, 3) != rt:
        sys.exit("GATE FAILED")
bestv = max(sc, key=lambda k: sc[k][0]); bestt = max(sc, key=lambda k: sc[k][1])
print(f"the best single model is the one with the highest video-disjoint validation BLEU-4 among the three adopted-recipe models: "
      f"{bestv} ({sc[bestv][0]:.2f} validation, {sc[bestv][1]:.2f} test)  [c14-035 c11-078]")
print("scope of 'best single model': the highest-validation model among the three adopted-recipe (vdB) singles that make up the ensemble comparison; the plain-baseline (vdA) singles are not considered  [c14-035 c11-078]")
print(f"(not the best on test: {bestt} scores {sc[bestt][1]:.2f} on test; the single model is selected on validation, so its test score is not a test-selected maximum)  [c14-035 c11-078]")
assert round(bb, 3) == round(sc[bestv][1], 3) and bestv == "vdB_s101"
