"""Wave 12i read-out: heavy-switch-bin difference-in-differences, V2 - C0 (confirmatory test of a wave-9 post hoc contrast).

Pre-registration: results/PREREG_WAVE12I_TRACKER_HEAVY_2026-10-13.md. The method is part (2) of
results/tracker_v2_translation_test/w9_secondaries.py, unchanged:

  B(arm, seed, bin) = corpus BLEU-4 (detokenised, sacreBLEU 13a) of the beam-5 decode of that run's best.pt on the
                      clips of the bin, each arm decoded on its OWN pose;
  DiD(seed)         = [B(v2, heavy) - B(c0, heavy)] - [B(v2, none) - B(c0, none)].

Bins are FROZEN in tools/w12i_bins.json (made once by `freeze`, before any wave-12i run existed) from
results/ndet_per_clip.csv: frac_chosen_nonzero = share of frames on which the wave-3 tracker (v1, job 6590) kept a
detection other than detection 0. none: == 0 | light: (0, 0.1] | medium: (0.1, 0.3] | heavy: > 0.3.

  python tools/w12i_heavy_did.py selftest
  python tools/w12i_heavy_did.py freeze NDET_CSV MANIFEST_DIR ARM_STATS_JSON OUT_JSON
  python tools/w12i_heavy_did.py w9gate --runs W9_RUNS --manifests W9_MANIFESTS      # must reproduce the recorded DiD
  python tools/w12i_heavy_did.py w9gate --full --boot 200 --runs .. --manifests ..   # every read-out, on the wave-9 seeds
  python tools/w12i_heavy_did.py inputs ARM_STATS_JSON --manifests W9_MANIFESTS      # how the two poses differ per bin
  python tools/w12i_heavy_did.py report --runs RUNS --manifests W9_MANIFESTS [--readouts_dir D] [--boot 1000]

`report` prints no score unless every run of both arms and all five pre-registered seeds carries its DONE marker.
SciPy is absent on Kaya, so the one-sided p uses the closed-form Student t CDF (df 4; df 2 for the wave-9 gate).
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, statistics as st, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402
from seed_decision import T975  # noqa: E402  (two-sided 95% t quantiles, df 1-6)

KEY = "tokenize=13a;detok=True"
ARMS = ("c0", "v2")                                   # control first
SEEDS = (303, 404, 505, 606, 707)                     # pre-registered; none was used in wave 9
W9_SEEDS = (42, 101, 202)
ALPHA = 0.05
NDET_MD5 = "9c45dae4aa1e2bdbe04b0cdc0e8ca806"         # results/ndet_per_clip.csv = Kaya wave3_track/ndet_per_clip.csv
BINS = {"none": lambda f: f == 0, "light": lambda f: 0 < f <= 0.1,
        "medium": lambda f: 0.1 < f <= 0.3, "heavy": lambda f: f > 0.3}       # as w9_secondaries.py
N_BIN = {"val": {"none": 620, "light": 72, "medium": 53, "heavy": 55},       # results/tracker_v2_translation_test/secondaries_output.txt
         "test": {"none": 607, "light": 80, "medium": 49, "heavy": 64}}
N_DEGEN = {"val": 13, "test": 9}                      # V2 keeps < half the frames (prereg amendment of 26 Sep)
W9_DID = {("val", "light"): (0.605, -1.525, 0.276), ("val", "medium"): (0.198, 0.815, -1.044),
          ("val", "heavy"): (5.289, 5.750, 4.199), ("test", "light"): (2.026, -1.638, 1.620),
          ("test", "medium"): (2.495, -2.453, 3.641), ("test", "heavy"): (2.080, 3.562, 1.194)}   # secondaries_output.txt


# ------------------------------------------------------------------ statistics

def t_cdf(t, df):
    """Student t CDF in closed form for df 2 and 4 (the only two this read-out needs)."""
    if df == 2:
        return 0.5 + t / (2 * math.sqrt(2 + t * t))
    if df == 4:
        u = t / math.sqrt(4 + t * t)
        return 0.5 + 0.75 * u * (1 - u * u / 3)
    return float("nan")


def tstats(d):
    """mean, sd, t, one-sided p (H1: mean > 0), two-sided 95% t-interval."""
    n, m = len(d), st.mean(d)
    sd = st.stdev(d)
    t = m / (sd / n ** 0.5) if sd > 0 else (math.copysign(math.inf, m) if m else 0.0)
    p = 1 - t_cdf(t, n - 1) if sd > 0 else float(m <= 0)        # zero spread: decided by the sign alone
    hw = T975[n - 1] * sd / n ** 0.5
    return {"mean": m, "sd": sd, "t": t, "p_one_sided": p, "lo": m - hw, "hi": m + hw,
            "positive": sum(x > 0 for x in d), "n": n}


def line(d):
    s = tstats(d)
    return (f"{' '.join(f'{x:+.3f}' for x in d)} | {s['positive']}/{s['n']} | mean {s['mean']:+.3f} "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | t {s['t']:+.3f}, one-sided p {s['p_one_sided']:.4f}")


def verdict(test_did, val_did):
    """The pre-registered rule: one-sided paired t over the five test DiDs at alpha 0.05 AND validation mean > 0."""
    p, vm = tstats(test_did)["p_one_sided"], st.mean(val_did)
    return {"p_test": p, "val_mean": vm, "test_rejects": p < ALPHA, "val_positive": vm > 0,
            "confirmed": p < ALPHA and vm > 0}


# ------------------------------------------------------------------ data

def bleu(h, r):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")


def rows(p):
    out = [l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8") if "\t" in l]
    return [c.rsplit("/", 1)[1][:-4] for c, _ in out], [t for _, t in out]


def sha(order):
    return hashlib.sha256("\n".join(order).encode()).hexdigest()


def freeze(ndet_csv, mdir, arm_stats, out):
    raw = open(ndet_csv, "rb").read()
    assert hashlib.md5(raw).hexdigest() == NDET_MD5, "ndet_per_clip.csv is not the file wave 9 used"
    frac = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(raw.decode().splitlines())}
    degen = {r["clip"] for r in json.load(open(arm_stats)) if (r["T"] - r["v2_masked"]) / r["T"] < 0.5}
    doc = {"rule": "frac_chosen_nonzero: none == 0, light (0, 0.1], medium (0.1, 0.3], heavy > 0.3",
           "source": "results/ndet_per_clip.csv", "source_md5": NDET_MD5,
           "degenerate_rule": "(T - v2_masked) / T < 0.5 in wave9/manifests/arm_stats.json"}
    for split in ("val", "test"):
        order, _ = rows(os.path.join(mdir, f"c0_{split}.tsv"))
        b = {name: [c for c in order if f(frac[c])] for name, f in BINS.items()}
        assert {k: len(v) for k, v in b.items()} == N_BIN[split], (split, {k: len(v) for k, v in b.items()})
        assert sum(map(len, b.values())) == len(order) == 800
        dg = [c for c in order if c in degen]
        assert len(dg) == N_DEGEN[split], (split, len(dg))
        doc[split] = {"order_sha256": sha(order), "bins": b, "degenerate": dg}
    json.dump(doc, open(out, "w"), indent=0, sort_keys=True)
    print(f"frozen bins -> {out} | md5 {hashlib.md5(open(out, 'rb').read()).hexdigest()}")


def load_bins(path, split, order):
    doc = json.load(open(path))[split]
    assert doc["order_sha256"] == sha(order), f"{split}: manifest clip order is not the one the bins were frozen on"
    pos = {c: i for i, c in enumerate(order)}
    idx = {name: [pos[c] for c in doc["bins"][name]] for name in BINS}
    assert {k: len(v) for k, v in idx.items()} == N_BIN[split]
    return idx, [pos[c] for c in doc["degenerate"]]


def load_runs(runs, mdir, seeds, on=None):
    """Hypotheses of every run, after the gates. on=None: each arm on its own pose; on='x': on the other arm's pose."""
    D = {}
    for split in ("val", "test"):
        order, texts = rows(os.path.join(mdir, f"{ARMS[0]}_{split}.tsv"))
        assert rows(os.path.join(mdir, f"{ARMS[1]}_{split}.tsv")) == (order, texts), f"{split}: arms differ in order or text"
        D[split, "order"], D[split, "refs"] = order, texts
        for arm in ARMS:
            pose = arm if on is None else ARMS[1 - ARMS.index(arm)]
            name = f"beam5_{split}_best.json" if on is None else f"beam5_{split}_best_on_{pose}.json"
            for s in seeds:
                j = json.load(open(os.path.join(runs, f"{arm}_s{s}", name)))
                assert j["manifest"].endswith(f"/{pose}_{split}.tsv"), j["manifest"]
                assert j["checkpoint"].endswith(f"/{arm}_s{s}/best.pt"), j["checkpoint"]
                assert j["beam"] == 5 and j["refs"] == texts and len(j["beam_hyps"]) == len(texts), (arm, s, split)
                assert abs(bleu(j["beam_hyps"], texts) - j["beam_result"]["metrics"][KEY]) < 1e-6, f"re-score {arm} s{s} {split}"
                D[split, arm, s] = j["beam_hyps"]
    return D


def sub(x, ii):
    return [x[i] for i in ii]


def did(D, split, seeds, a, b, ii, base, score=bleu, Da=None, Db=None):
    """Per-seed [score_a(ii) - score_b(ii)] - [score_a(base) - score_b(base)]; base=None gives the plain delta on ii."""
    Da, Db, r = Da or D, Db or D, D[split, "refs"]
    out = []
    for s in seeds:
        ha, hb = Da[split, a, s], Db[split, b, s]
        d = score(sub(ha, ii), sub(r, ii)) - score(sub(hb, ii), sub(r, ii))
        if base is not None:
            d -= score(sub(ha, base), sub(r, base)) - score(sub(hb, base), sub(r, base))
        out.append(d)
    return out


def chrf(h, r):
    import sacrebleu
    return st.mean(sacrebleu.sentence_chrf(detokenize(a), [detokenize(b)]).score for a, b in zip(h, r))


def boot_did(D, split, seeds, heavy, none, n, seed=0):
    """Clip-sampling interval of the seed-mean heavy DiD: clips resampled within each bin, same draw for every run."""
    rng, out = random.Random(seed), []
    for _ in range(n):
        hh, nn = rng.choices(heavy, k=len(heavy)), rng.choices(none, k=len(none))
        out.append(st.mean(did(D, split, seeds, "v2", "c0", hh, nn)))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1], sum(x > 0 for x in out) / n


# ------------------------------------------------------------------ modes

def inputs(arm_stats, mdir, bins):
    """Input description only (no model output): per bin, the share of frames on which C0 and V2 differ or V2 masks."""
    S = {r["clip"]: r for r in json.load(open(arm_stats))}
    for split in ("val", "test"):
        order, _ = rows(os.path.join(mdir, f"c0_{split}.tsv"))
        idx, _ = load_bins(bins, split, order)
        own = {i for i, c in enumerate(order) if S[c]["c0_ne_v2"] / S[c]["T"] > 0.3}
        for name, ii in idx.items():
            T = sum(S[order[i]]["T"] for i in ii)
            print(f"  {split:4s} {name:6s} n={len(ii):3d} frames {T:5d} | C0 != V2 on {100 * sum(S[order[i]]['c0_ne_v2'] for i in ii) / T:5.1f}% | "
                  f"V2 masks {100 * sum(S[order[i]]['v2_masked'] for i in ii) / T:5.1f}% | clips with no differing frame "
                  f"{sum(S[order[i]]['c0_ne_v2'] == 0 for i in ii):3d} | clips where C0 != V2 on > 30% of frames {len(own & set(ii)):3d}")
        print(f"  {split:4s} clips where C0 != V2 on > 30% of frames: {len(own)} in all")


def w9gate(runs, mdir, bins):
    D = load_runs(runs, mdir, W9_SEEDS)
    for split in ("val", "test"):
        idx, _ = load_bins(bins, split, D[split, "order"])
        for name in ("light", "medium", "heavy"):
            got = did(D, split, W9_SEEDS, "v2", "c0", idx[name], idx["none"])
            assert all(abs(g - e) < 6e-4 for g, e in zip(got, W9_DID[split, name])), (split, name, got)
            print(f"  w9 {split:4s} {name:6s} DiD {line(got)}")
    print("W9 GATE PASSED: the 18 recorded wave-9 DiD values (3 bins x 2 splits x 3 seeds) are reproduced")


def report(runs, mdir, bins, seeds, readouts_dir, n_boot, expected=SEEDS):
    """expected=SEEDS: the confirmatory read-out. expected=W9_SEEDS: the same lines for wave 9, with no verdict."""
    missing = [f"{a}_s{s}" for a in ARMS for s in seeds if not os.path.exists(os.path.join(runs, f"{a}_s{s}", "DONE"))]
    if missing or tuple(seeds) != tuple(expected):
        print(f"INCOMPLETE: no score is printed. Seeds {list(seeds)} (expected {list(expected)}); runs without DONE: {missing}")
        return 2
    confirm = tuple(expected) == SEEDS
    assert not (confirm and set(seeds) & set(W9_SEEDS)), "a wave-9 seed is in the confirmatory set"
    D = load_runs(runs, mdir, seeds)
    print(f"GATES PASSED: {2 * len(seeds)} runs DONE; {4 * len(seeds)} beam-5 BLEU values re-scored exactly; references = manifest text; "
          f"each arm decoded on its own pose")
    B = {}
    for split in ("val", "test"):
        B[split], dg = load_bins(bins, split, D[split, "order"])
        allc = list(range(len(D[split, "order"])))
        print(f"\n== {split}: n per bin " + ", ".join(f"{k} {len(v)}" for k, v in B[split].items()))
        for a in ARMS:
            print(f"  {a} whole split  " + " ".join(f"{bleu(D[split, a, s], D[split, 'refs']):.3f}" for s in seeds))
        print(f"  v2-c0 whole split {line([bleu(D[split, 'v2', s], D[split, 'refs']) - bleu(D[split, 'c0', s], D[split, 'refs']) for s in seeds])}")
        for name in ("none", "light", "medium", "heavy"):
            ii = B[split][name]
            print(f"  v2-c0 {name:6s} bin BLEU delta {line(did(D, split, seeds, 'v2', 'c0', ii, None))}"
                  if name == "none" else
                  f"  v2-c0 {name:6s} DiD vs none    {line(did(D, split, seeds, 'v2', 'c0', ii, B[split]['none']))}")
        keep = [i for i in allc if i not in set(dg)]
        print(f"  sensitivity (-{len(dg)} degenerate clips): whole split {line(did(D, split, seeds, 'v2', 'c0', keep, None))}")
        hk, nk = [i for i in B[split]['heavy'] if i not in set(dg)], [i for i in B[split]['none'] if i not in set(dg)]
        print(f"  sensitivity heavy DiD without them (heavy n={len(hk)}, none n={len(nk)}): {line(did(D, split, seeds, 'v2', 'c0', hk, nk))}")
        print(f"  heavy DiD in mean sentence chrF: {line(did(D, split, seeds, 'v2', 'c0', B[split]['heavy'], B[split]['none'], chrf))}")
        if n_boot:
            lo, hi, pos = boot_did(D, split, seeds, B[split]["heavy"], B[split]["none"], n_boot)
            print(f"  heavy DiD, clip sampling only ({n_boot} resamples within bins, seeds fixed): 95% [{lo:+.3f}, {hi:+.3f}], share > 0 {pos:.3f}")

    try:
        X = load_runs(runs, mdir, seeds, on="x")
    except FileNotFoundError as e:
        print(f"\n== cross-pose decodes missing ({e.filename}): decomposition not printed")
    else:
        print("\n== decomposition (descriptive): heavy DiD vs none of each single change")
        for split in ("val", "test"):
            h, n0 = B[split]["heavy"], B[split]["none"]
            for lab, a, Da, b, Db in (("decode pose v2 vs c0, model c0", "c0", X, "c0", D), ("decode pose v2 vs c0, model v2", "v2", D, "v2", X),
                                      ("model v2 vs c0, decode pose c0", "v2", X, "c0", D), ("model v2 vs c0, decode pose v2", "v2", D, "c0", X)):
                print(f"  {split:4s} {lab}: {line(did(D, split, seeds, a, b, h, n0, Da=Da, Db=Db))}")

    if readouts_dir:
        print("\n== whole-split secondaries, v2 - c0 paired per seed (tools/arm_readouts.py)")
        for split in ("val", "test"):
            rd = dict((l.split(" ", 1)[0], json.loads(l.split(" ", 1)[1])) for l in open(os.path.join(readouts_dir, f"readouts_{split}.txt")))
            if split == "test":
                assert all(rd[f"{a}_s{s}"]["unseen_n"] == 570 for a in ARMS for s in seeds), "unseen-target n is not 570"
            for k in ("bleu4_unseen", "content_precision", "content_recall", "len_ratio", "wer"):
                print(f"  {split:4s} {k:18s} {line([rd[f'v2_s{s}'][k] - rd[f'c0_s{s}'][k] for s in seeds])}")

    tv = {sp: did(D, sp, seeds, "v2", "c0", B[sp]["heavy"], B[sp]["none"]) for sp in ("val", "test")}
    v = verdict(tv["test"], tv["val"])
    print("\n== PRIMARY (pre-registered): heavy-bin DiD, V2 - C0, five new seeds" if confirm else
          "\n== heavy-bin DiD, V2 - C0, on the three WAVE-9 seeds (the seeds the contrast was found on: not a test)")
    print(f"  test       {line(tv['test'])}")
    print(f"  validation {line(tv['val'])}")
    if not confirm:
        return 0
    print(f"  (a) one-sided paired t on test, p {v['p_test']:.4f} < {ALPHA}: {'MET' if v['test_rejects'] else 'NOT MET'}"
          f" | (b) validation mean {v['val_mean']:+.3f} > 0: {'MET' if v['val_positive'] else 'NOT MET'}")
    print(f"  VERDICT: {'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}")
    return 0


def selftest():
    # closed-form t CDF against tabulated one-sided quantiles, and against SciPy where it is installed
    for df, t, p in ((4, 2.1318, 0.05), (4, 2.7764, 0.025), (4, 3.7469, 0.01), (2, 2.9200, 0.05), (2, 4.3027, 0.025)):
        assert abs((1 - t_cdf(t, df)) - p) < 2e-5, (df, t)
    assert abs(t_cdf(0, 4) - 0.5) < 1e-12 and abs(t_cdf(-2.1318, 4) - 0.05) < 2e-5
    try:
        from scipy import stats
        for d in ([2.080, 3.562, 1.194], [0.4, -0.2, 1.1, 0.3, 0.9], [-1.0, -0.5, 0.2, -0.1, -0.3]):
            t, p2 = stats.ttest_1samp(d, 0.0)
            assert abs(tstats(d)["p_one_sided"] - (p2 / 2 if t > 0 else 1 - p2 / 2)) < 1e-9, d
    except ImportError:
        print("  (SciPy not installed: closed forms checked against tabulated quantiles only)")
    # the rule: both conditions are needed
    assert verdict([2.0, 2.5, 1.5, 2.2, 1.8], [1, 1, 1, 1, 1])["confirmed"]
    assert not verdict([2.0, 2.5, 1.5, 2.2, 1.8], [-1, -1, 1, -1, -1])["confirmed"]       # validation mean <= 0
    assert not verdict([2.0, -2.5, 1.5, -2.2, 1.8], [1, 1, 1, 1, 1])["confirmed"]         # test does not reject
    assert not verdict([-2.0, -2.5, -1.5, -2.2, -1.8], [1, 1, 1, 1, 1])["confirmed"]      # wrong sign
    # end to end on a synthetic wave: 8 clips, heavy = the last two; v2 is right on heavy clips, c0 is wrong there
    refs = [f"this is sentence number {w} of the set ." for w in "one two three four five six seven eight".split()]
    wrong = "that was great ."
    with tempfile.TemporaryDirectory() as tmp:
        md, runs = os.path.join(tmp, "m"), os.path.join(tmp, "runs")
        os.makedirs(md)
        for a in ARMS:
            for split in ("val", "test"):
                open(os.path.join(md, f"{a}_{split}.tsv"), "w").write("".join(f"/x/{a}/clip{i}.npz\t{r}\n" for i, r in enumerate(refs)))
        order = [f"clip{i}" for i in range(8)]
        bins = os.path.join(tmp, "bins.json")
        json.dump({sp: {"order_sha256": sha(order), "degenerate": ["clip7"],
                        "bins": {"none": order[:4], "light": order[4:5], "medium": order[5:6], "heavy": order[6:]}}
                   for sp in ("val", "test")}, open(bins, "w"))
        global N_BIN
        keep, N_BIN = N_BIN, {sp: {"none": 4, "light": 1, "medium": 1, "heavy": 2} for sp in ("val", "test")}
        try:
            def dump(arm, s, split, hyps, pose=None):
                d = os.path.join(runs, f"{arm}_s{s}"); os.makedirs(d, exist_ok=True)
                name = f"beam5_{split}_best.json" if pose is None else f"beam5_{split}_best_on_{pose}.json"
                json.dump({"manifest": f"/m/{pose or arm}_{split}.tsv", "checkpoint": f"/r/{arm}_s{s}/best.pt", "beam": 5,
                           "beam_hyps": hyps, "refs": refs, "beam_result": {"metrics": {KEY: bleu(hyps, refs)}}},
                          open(os.path.join(d, name), "w"))
            for s in SEEDS:
                for split in ("val", "test"):
                    dump("c0", s, split, refs[:6] + [wrong, wrong]); dump("v2", s, split, refs)
                    dump("c0", s, split, refs, pose="v2"); dump("v2", s, split, refs[:6] + [wrong, wrong], pose="c0")
            assert report(runs, md, bins, SEEDS, "", 0) == 2                               # no DONE marker -> no score
            for a in ARMS:
                for s in SEEDS:
                    open(os.path.join(runs, f"{a}_s{s}", "DONE"), "w").close()
            assert report(runs, md, bins, SEEDS[:3], "", 0) == 2                           # not the pre-registered seeds
            assert report(runs, md, bins, SEEDS, "", 0, expected=W9_SEEDS) == 2            # nor a wave-9 read-out
            D = load_runs(runs, md, SEEDS)
            idx, dg = load_bins(bins, "test", D["test", "order"])
            d = did(D, "test", SEEDS, "v2", "c0", idx["heavy"], idx["none"])
            exp = bleu(refs[6:], refs[6:]) - bleu([wrong, wrong], refs[6:])                 # none-bin delta is exactly 0
            assert dg == [7] and all(abs(x - exp) < 1e-9 for x in d) and exp > 50, (d, exp)
            assert all(abs(x) < 1e-9 for x in did(D, "test", SEEDS, "c0", "c0", idx["heavy"], idx["none"]))
            X = load_runs(runs, md, SEEDS, on="x")          # here the whole effect is the decode pose, none the model
            assert all(abs(x - exp) < 1e-9 for x in did(D, "test", SEEDS, "c0", "c0", idx["heavy"], idx["none"], Da=X, Db=D))
            assert all(abs(x) < 1e-9 for x in did(D, "test", SEEDS, "v2", "c0", idx["heavy"], idx["none"], Da=X, Db=D))
            lo, hi, pos = boot_did(D, "test", SEEDS, idx["heavy"], idx["none"], 50)
            assert lo <= hi and pos == 1.0
            assert report(runs, md, bins, SEEDS, "", 20) == 0
            j = json.load(open(os.path.join(runs, "v2_s303", "beam5_test_best.json")))
            j["beam_result"]["metrics"][KEY] += 0.01                                        # a stored score that does not re-score
            json.dump(j, open(os.path.join(runs, "v2_s303", "beam5_test_best.json"), "w"))
            try:
                load_runs(runs, md, SEEDS); raise SystemExit("selftest: the re-score gate did not fire")
            except AssertionError:
                pass
            try:
                load_bins(bins, "val", order[::-1]); raise SystemExit("selftest: the clip-order gate did not fire")
            except AssertionError:
                pass
        finally:
            N_BIN = keep
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "freeze", "w9gate", "inputs", "report"))
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--runs"); ap.add_argument("--manifests")
    ap.add_argument("--bins", default=str(HERE / "w12i_bins.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--readouts_dir", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--full", action="store_true", help="w9gate: also print every read-out for the wave-9 seeds")
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest()
    elif a.mode == "freeze":
        freeze(*a.paths)
    elif a.mode == "w9gate":
        w9gate(a.runs, a.manifests, a.bins)
        if a.full:
            report(a.runs, a.manifests, a.bins, W9_SEEDS, a.readouts_dir, a.boot, expected=W9_SEEDS)
    elif a.mode == "inputs":
        inputs(a.paths[0], a.manifests, a.bins)
    else:
        sys.exit(report(a.runs, a.manifests, a.bins, a.seeds, a.readouts_dir, a.boot))


if __name__ == "__main__":
    main()
