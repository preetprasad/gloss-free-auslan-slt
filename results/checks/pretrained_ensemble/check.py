#!/usr/bin/env python3
"""Recompute the five-model and three-model ensemble readings of the pre-trained sign-language model from saved decodes.

Run:  cd src && python3 ../results/checks/<this folder>/check.py

Inputs (saved decodes, hypotheses only; references come from comm_run/manifests/{val,test}.tsv, written by `make data`):
  data/single5_s{seed}_{val,test}.json   five fine-tuned models decoded alone (beam 5, length penalty 1.0)
  data/ens5_grid_b{beam}_lp{lp}.json     validation grid of the five-model ensemble (10 settings)
  data/ens5_{E0,Estar}_{val,test}.json   the two ensembles on validation and on test
  data/single3_*, data/ens3_*            the three-model ensemble (members fine-tuned with our own code)
Statistics follow the original read-out: BLEU-4 from sacreBLEU 13a sufficient statistics of detokenised text (asserted equal to slt.metrics),
one set of B = 2000 sentence resamples (numpy default_rng(0)) shared by all systems, family of four comparisons, studentised max-T
(q = 0.95 quantile of max_j |boot_j - obs_j| / se_j), predicted power = Phi((validation effect - q se) / se).
"""
import json, math, re, sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

D = HERE / "data"
MAN = ROOT / "comm_run" / "manifests"
B_BOOT, ALPHA, N = 2000, 0.05, 800
BL = BLEU(tokenize="13a")


def fail(msg):
    print("GATE FAILED:", msg)
    sys.exit(1)


def read_tsv(p):
    if not p.exists():
        fail(f"{p} missing: run `make data` first")
    return [l.rstrip("\n").split("\t", 1)[1] for l in open(p, encoding="utf-8") if "\t" in l]


REFS = {sp: read_tsv(MAN / f"{sp}.tsv") for sp in ("val", "test")}
norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()
TRAIN = {norm(t) for t in read_tsv(MAN / "train.tsv")}
SEEN = np.array([norm(r) in TRAIN for r in REFS["test"]])


def load(name):
    return json.load(open(D / name))


def stats(hyps, refs):
    return np.array(BL._extract_corpus_statistics([detokenize(h) for h in hyps], [[detokenize(r) for r in refs]]), dtype=float)


def bleu(sums):
    return float(BL._compute_score_from_stats([float(x) for x in sums]).score)


GATES = []
SETTING = {}


def sysstats(name, split):
    """sufficient statistics of a saved decode; gate: equals the BLEU-4 the original job recorded and slt.metrics."""
    j = load(name)
    refs = REFS[split]
    assert len(j["hyps"]) == len(refs) == N, name
    st = stats(j["hyps"], refs)
    a = bleu(st.sum(0))
    b = corpus_bleu4([detokenize(h) for h in j["hyps"]], [detokenize(r) for r in refs], "13a")
    if abs(a - b) > 1e-9 or abs(a - j["recorded_bleu4"]) > 1e-6:
        fail(f"{name}: recomputed {a} / slt.metrics {b} / recorded {j['recorded_bleu4']}")
    GATES.append(name)
    SETTING[name] = (j["beam"], j["length_penalty"])
    return st, j["hyps"]


def family(S, E0, Es, bi, tag):
    """S: list of single-model stats, E0/Es ensemble stats, bi index of the best-on-validation single."""
    allst = np.stack(S + [E0, Es])                      # (k+2, N, 10)
    k = len(S)
    idx = np.random.default_rng(0).integers(0, N, size=(B_BOOT, N))
    obs_b = np.array([bleu(a.sum(0)) for a in allst])
    boot = np.empty((B_BOOT, k + 2))
    for r, ii in enumerate(idx):
        sums = allst[:, ii, :].sum(1)
        boot[r] = [bleu(x) for x in sums]

    def deltas(v):                                      # v: (..., k+2)
        mean_s = v[..., :k].mean(-1)
        best = v[..., bi]
        e0, es = v[..., k], v[..., k + 1]
        return np.stack([e0 - mean_s, e0 - best, es - mean_s, es - best], -1)

    obs, bt = deltas(obs_b), deltas(boot)
    se = bt.std(0, ddof=1)
    q = float(np.quantile((np.abs(bt - obs) / se).max(1), 1 - ALPHA))
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    out = {"obs": obs, "se": se, "q": q, "maxT": (obs - q * se, obs + q * se), "pb": np.percentile(bt, [2.5, 97.5], axis=0),
           "power": [Phi((o - q * s) / s) for o, s in zip(obs, se)], "E_minus_E0": (obs_b[k + 1] - obs_b[k], np.percentile(boot[:, k + 1] - boot[:, k], [2.5, 97.5])),
           "single": obs_b[:k], "ens": (obs_b[k], obs_b[k + 1])}
    return out


def fmt_iv(lo, hi):
    return f"[{lo:+.2f}, {hi:+.2f}]"


LAB = ("E0-meanS", "E0-bestS", "E*-meanS", "E*-bestS")


def report(title, res, nseed, meanlab, bestlab):
    print(f"max-T over the family of four (B = {B_BOOT}, alpha {ALPHA}), q = {res['q']:.3f}; observed on {title}:")
    for j, lab in enumerate(LAB):
        print(f"  {lab:9s} {res['obs'][j]:+.2f}  max-T 95% {fmt_iv(res['maxT'][0][j], res['maxT'][1][j])}  (paired-bootstrap 95% {fmt_iv(res['pb'][0][j], res['pb'][1][j])})")


POWER_MIN = 0.5


def claim_rule(res, power):
    out = {}
    for tag, js in (("E0", (0, 1)), ("E*", (2, 3))):
        ci = all(res["maxT"][0][j] > 0 for j in js)
        pw = min(power[j] for j in js)
        out[tag] = (ci, pw, ci and pw >= POWER_MIN)
    return out


def unseen_bleu(hyps):
    st = stats(hyps, REFS["test"])
    return bleu(st[~SEEN].sum(0)), bleu(st[SEEN].sum(0))


def desc(hyps):
    refs = REFS["test"]
    b = BL.corpus_score([detokenize(h) for h in hyps], [[detokenize(r) for r in refs]])
    ver = float(np.mean([norm(h) in TRAIN for h in hyps]))
    return b.sys_len / b.ref_len, b.bp, ver, len(set(hyps))


# =============================================================== five-model ensemble
print("=== Five fine-tuned models and their ensemble (n = 800 clips per split) ===")
SEEDS5 = (42, 101, 202, 303, 404)
grid_names = [(b, lp) for b in (5, 10) for lp in (0.6, 0.8, 1.0, 1.2, 1.5)]
gval = {k: bleu(sysstats(f"ens5_grid_b{k[0]}_lp{k[1]:.1f}.json", "val")[0].sum(0)) for k in grid_names}
top = max(round(v, 3) for v in gval.values())
tied = [k for k, v in gval.items() if round(v, 3) == top]
pick = sorted(tied, key=lambda k: (k[0], abs(k[1] - 1.0), k[1]))[0]
print("validation grid, five-model ensemble BLEU-4 (beam, length penalty):")
for k in grid_names:
    print(f"  beam {k[0]:2d} lp {k[1]:.1f}: {gval[k]:.2f}" + ("   <- selected on validation" if k == pick else ""))
print(f"E0 = beam 5, length penalty 1.0 (fixed in advance); E* = beam {pick[0]}, length penalty {pick[1]} (validation argmax)")
if pick != (10, 0.8):
    fail(f"selected {pick}")

ST = {sp: {} for sp in ("val", "test")}
HY = {sp: {} for sp in ("val", "test")}
for sp in ("val", "test"):
    for s in SEEDS5:
        ST[sp][f"s{s}"], HY[sp][f"s{s}"] = sysstats(f"single5_s{s}_{sp}.json", sp)
    for nm in ("E0", "Estar"):
        ST[sp][nm], HY[sp][nm] = sysstats(f"ens5_{nm}_{sp}.json", sp)
sc = lambda sp, k: bleu(ST[sp][k].sum(0))
best_val = max(SEEDS5, key=lambda s: (round(sc("val", f"s{s}"), 3), -s))
print("single models, beam 5 length penalty 1.0, BLEU-4 validation / test:")
for s in SEEDS5:
    print(f"  seed {s:3d}: {sc('val', f's{s}'):.2f} / {sc('test', f's{s}'):.2f}" + ("   <- best on validation" if s == best_val else ""))
vmean = np.mean([sc("val", f"s{s}") for s in SEEDS5])
tmean = np.mean([sc("test", f"s{s}") for s in SEEDS5])
print(f"member mean: validation {vmean:.2f}, test {tmean:.2f}; best-on-validation member: test {sc('test', f's{best_val}'):.2f}")
print(f"five-model ensemble E0: validation {sc('val', 'E0'):.2f}, test {sc('test', 'E0'):.2f}")
print(f"five-model ensemble E*: validation {sc('val', 'Estar'):.2f}, test {sc('test', 'Estar'):.2f}")
assert SETTING["ens5_E0_test.json"] == (5, 1.0) and SETTING["ens5_Estar_test.json"] == pick, SETTING
print("test decodes of the five-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam %d, length penalty %.1f)" % pick)

bi = SEEDS5.index(best_val)
mk = lambda sp: family([ST[sp][f"s{s}"] for s in SEEDS5], ST[sp]["E0"], ST[sp]["Estar"], bi, sp)
rv, rt = mk("val"), mk("test")
print("--- predicted power from the validation decodes (plug-in: test effect = validation effect) ---")
for j, lab in enumerate(LAB):
    print(f"  {lab:9s} validation effect {rv['obs'][j]:+.2f}, predicted power {rv['power'][j]:.2f}")
print("--- test reading ---")
report("test", rt, 5, "member mean", "member best on validation")
print(f"  member mean {tmean:.2f}; member best on validation {sc('test', f's{best_val}'):.2f}")
cl = claim_rule(rt, rv["power"])
for tag in ("E0", "E*"):
    ci, pw, ok = cl[tag]
    print(f"claim rule {tag}: both max-T intervals exclude 0 = {ci}; predicted power {pw:.2f} (>= {POWER_MIN}: {pw >= POWER_MIN}) -> {'CLAIM' if ok else 'NO CLAIM'}")
print(f"descriptive, outside the family: E* minus E0 {rt['E_minus_E0'][0]:+.2f} {fmt_iv(*rt['E_minus_E0'][1])}")

print("unseen-570 test BLEU-4 (references absent from training, n = %d):" % int((~SEEN).sum()))
ub = {k: unseen_bleu(HY["test"][k]) for k in ST["test"]}
print(f"  member mean unseen {np.mean([ub[f's{s}'][0] for s in SEEDS5]):.2f}")
print(f"  five-model E0 unseen {ub['E0'][0]:.2f}, E* unseen {ub['Estar'][0]:.2f}")
d = {k: desc(HY["test"][k]) for k in ("E0", "Estar")}
print(f"  E* length ratio {d['Estar'][0]:.3f}, brevity penalty {d['Estar'][1]:.3f}; verbatim-train outputs E* {d['Estar'][2]:.3f}")

# comparison with the 11.80 system (published dump, same test clips)
s6 = [json.loads(l) for l in open(ROOT / "results" / "final_ensemble_test_decodes" / "hyps_LP=0.3.jsonl", encoding="utf-8")]
s6h = [r["hyp_beam"] for r in s6]
if [r["ref"] for r in s6] != REFS["test"]:
    fail("S6 dump references differ from the test manifest")
s6_all = bleu(stats(s6h, REFS["test"]).sum(0))
s6_un, _ = unseen_bleu(s6h)
if abs(s6_all - 11.802) > 0.002:
    fail(f"S6 gate: {s6_all}")
print(f"  {s6_all:.2f}-system (S6, beam 10, lp 0.3): test {s6_all:.2f}, unseen {s6_un:.2f}")
print(f"  best published score on this subset (GASLT): 9.95 (literature value, not recomputed here)")

# =============================================================== three-model ensemble
print()
print("=== Three models fine-tuned with our own code and their ensemble ===")
SEEDS3 = (42, 101, 202)
S3 = {sp: {} for sp in ("val", "test")}
H3 = {}
for sp in ("val", "test"):
    for s in SEEDS3:
        S3[sp][f"s{s}"], h = sysstats(f"single3_s{s}_{sp}.json", sp)
        if sp == "test":
            H3[f"s{s}"] = h
for nm in ("E0", "Estar"):
    S3["test"][nm], H3[nm] = sysstats(f"ens3_{nm}_test.json", "test")
sc3 = lambda sp, k: bleu(S3[sp][k].sum(0))
best3 = max(SEEDS3, key=lambda s: (round(sc3("val", f"s{s}"), 3), -s))
t3 = {s: sc3("test", f"s{s}") for s in SEEDS3}
print("three single models, test BLEU-4: " + ", ".join(f"seed {s} {t3[s]:.2f}" for s in SEEDS3) + f"; mean {np.mean(list(t3.values())):.2f}")
print(f"best on validation: seed {best3} (validation {sc3('val', f's{best3}'):.2f}, test {t3[best3]:.2f})")
assert SETTING["ens3_E0_test.json"] == (5, 1.0) and SETTING["ens3_Estar_test.json"] == (10, 0.8), SETTING
print("test decodes of the three-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam 10, length penalty 0.8)")
print(f"three-model ensemble: E0 test {sc3('test', 'E0'):.2f}, E* test {sc3('test', 'Estar'):.2f}")
r3 = family([S3["test"][f"s{s}"] for s in SEEDS3], S3["test"]["E0"], S3["test"]["Estar"], SEEDS3.index(best3), "test3")
report("test", r3, 3, "member mean", "member best on validation")
ci3 = all(r3["maxT"][0][j] > 0 for j in (2, 3))
print(f"E*: both max-T intervals exclude 0 = {ci3} -> {'CLAIM' if ci3 else 'NO CLAIM: bounded estimate only'}")
for k, lab in (("E0", "E0"), ("Estar", "E*")):
    u, _ = unseen_bleu(H3[k])
    print(f"  three-model {lab} unseen-570 test BLEU-4 {u:.2f}")
u3 = np.mean([unseen_bleu(H3[f"s{s}"])[0] for s in SEEDS3])
print(f"  three-model member mean unseen {u3:.2f}")
print(f"gates passed: {len(GATES)} saved decodes re-scored equal to the BLEU-4 recorded by the original jobs and to slt.metrics")
print("CHECK_DONE")
