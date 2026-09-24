#!/usr/bin/env python3
"""Decision readout for wave 8 (results/PREREG_WAVE8_REALDATA_2026-09-24.md, job 21734). Written before any result.

Reads <runs>/{R025,R050,R100,V050}_s{seed}/beam5_{val,test}_{best,soup_last5}.json (eval_beam.py output: beam_hyps,
refs, beam_result.metrics["tokenize=13a;detok=True"]) and the per-arm subset manifests <manifests>/<arm>_s<seed>.tsv.

Q1  Δ = R100 − R050 per seed on official-800 and unseen-570 (unseen = test reference not verbatim in the FULL 12,439
    train text), seed-level 95% t (df 4), sign counts, pooled paired sentence bootstrap ("test sampling only");
    per-seed OLS slope on log2 N; readings (i)/(ii)/(ii')/(iii)/(iv); validation and soup sign checks.
Q2  per seed, cells over TEST clips by episode: held = videos absent from V050's subset, kept = videos present IN FULL;
    the one partially cut video is dropped from both. DiD = (V050 − R050)_held − (V050 − R050)_kept, co-primary on all
    clips and on the fixed unseen set; per-seed seen-count DiD; single differences; secondary restricted to clips whose
    reference content words all occur in BOTH arms' training text. Readings R1/R2/R3.

  python tools/learning_curve.py --runs RUNS --manifests MAN --train M/train.tsv --test M/test.tsv
  python tools/learning_curve.py --selftest
"""
import argparse, json, math, os, re, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))
from slt.metrics import corpus_bleu4, detokenize          # noqa: E402
from stitch_lexicon import content_words                 # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
SIZES = {"R025": 3141, "R050": 6241, "R100": 12439}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
KEY = "tokenize=13a;detok=True"
VID = re.compile(r"(video_\d+)_")
norm = lambda s: " ".join(detokenize(s).lower().split())


def bleu(h, r, keep=None):
    if keep is not None:
        h = [x for x, k in zip(h, keep) if k]; r = [x for x, k in zip(r, keep) if k]
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")


def interval(d):
    m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw


def resolved(d):
    m, lo, hi = interval(d)
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def fmt(name, d):
    m, lo, hi = interval(d)
    return f"{name:34s} {' '.join(f'{x:+.3f}' for x in d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | +{sum(x > 0 for x in d)}/{len(d)}"


def load(runs, arm, s, split, ck, refs_expected=None):
    j = json.load(open(os.path.join(runs, f"{arm}_s{s}", f"beam5_{split}_{ck}.json")))
    h, r = j["beam_hyps"], j["refs"]
    b = bleu(h, r)
    assert abs(b - j["beam_result"]["metrics"][KEY]) < 0.002, f"{arm}_s{s} {split} {ck}: rescored {b} vs recorded"
    if refs_expected is not None:
        assert [" ".join(x.split()) for x in r] == refs_expected, f"{arm}_s{s} {split}: refs not in manifest order"
    return h, r


def read_manifest(p):
    rows = [l.rstrip("\n").split("\t", 1) for l in open(p) if "\t" in l]
    return [x[0] for x in rows], [x[1] for x in rows]


def ols_slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def boot(pairs, keep, n=2000, seed=0):
    """pairs: list of (hyps_a, hyps_b, refs) over seeds; one sentence resample per draw applied to every seed."""
    idx = [i for i, k in enumerate(keep) if k]; rng = np.random.default_rng(seed); out = np.empty(n)
    for t in range(n):
        ii = rng.choice(idx, size=len(idx), replace=True)
        out[t] = np.mean([bleu([a[i] for i in ii], [r[i] for i in ii]) - bleu([b[i] for i in ii], [r[i] for i in ii])
                          for a, b, r in pairs])
    return np.percentile(out, [2.5, 97.5])


def q1(a, test_refs, unseen):
    print("=== Q1 learning curve (primary Δ = R100 − R050, beam-5 best.pt) ===")
    res = {}
    for arm in SIZES:
        for s in SEEDS:
            for split in ("val", "test"):
                for ck in ("best", "soup_last5"):
                    h, r = load(a.runs, arm, s, split, ck, test_refs if split == "test" else None)
                    res[(arm, s, split, ck, "all")] = bleu(h, r)
                    if split == "test":
                        res[(arm, s, split, ck, "unseen")] = bleu(h, r, unseen)
                        if ck == "best":
                            res[(arm, s, "hyps")] = h
    d = {k: [res[("R100", s, sp, ck, sub)] - res[("R050", s, sp, ck, sub)] for s in SEEDS]
         for k, (sp, ck, sub) in dict(off=("test", "best", "all"), uns=("test", "best", "unseen"),
                                      val=("val", "best", "all"), soup=("test", "soup_last5", "all")).items()}
    for k, lab in (("off", "Δ official-800 test"), ("uns", f"Δ unseen-{sum(unseen)} test"),
                   ("val", "Δ validation"), ("soup", "Δ official test (soup)")):
        print(fmt(lab, d[k]))
    for sub in ("all", "unseen"):
        sl = [ols_slope([math.log2(n) for n in SIZES.values()],
                        [res[(arm, s, "test", "best", sub)] for arm in SIZES]) for s in SEEDS]
        print(fmt(f"slope per doubling ({sub})", sl))
    pairs = [(res[("R100", s, "hyps")], res[("R050", s, "hyps")], test_refs) for s in SEEDS]
    for lab, keep in (("official", [True] * len(test_refs)), ("unseen", unseen)):
        lo, hi = boot(pairs, keep, a.n)
        print(f"pooled sentence bootstrap Δ {lab} (test sampling only): [{lo:+.3f}, {hi:+.3f}]")
    ro, ru = resolved(d["off"]), resolved(d["uns"]); mu = st.mean(d["uns"])
    if ru == 1: r = "(i) more in-register real pairs still raise generalisation"
    elif ro == 1 and ru == 0 and mu < 0.3: r = "(ii) extra pairs mainly buy seen-target coverage"
    elif ro == 1 and ru == 0: r = "(ii') positive direction, not resolvable at 5 seeds"
    elif ro == 0 and ru == 0: r = "(iii) not data-limited within resolution"
    else: r = "(iv) bounded, inconclusive estimate"
    print("READING Q1:", r)
    for k in ("val", "soup"):
        if (st.mean(d[k]) > 0) != (st.mean(d["off"]) > 0):
            print(f"  NOTE: {k} Δ disagrees in sign with official test Δ -> report both, pick neither")


def q2(a, test_paths, test_refs, unseen, train_paths, train_texts):
    print("\n=== Q2 episode familiarity (V050 vs R050 at 6,241; DiD held − kept, cut video dropped) ===")
    per_video = {}
    for p in train_paths:
        v = VID.match(os.path.basename(p)).group(1); per_video[v] = per_video.get(v, 0) + 1
    tvid = [VID.match(os.path.basename(p)).group(1) for p in test_paths]
    full_train = {norm(t) for t in train_texts}
    out = {k: [] for k in ("off", "uns", "seen", "both", "held", "kept")}
    for s in SEEDS:
        vp, vt = read_manifest(os.path.join(a.manifests, f"V050_s{s}.tsv"))
        rp, rt = read_manifest(os.path.join(a.manifests, f"R050_s{s}.tsv"))
        kept_n = {}
        for p in vp:
            v = VID.match(os.path.basename(p)).group(1); kept_n[v] = kept_n.get(v, 0) + 1
        cut = [v for v, n in kept_n.items() if n < per_video[v]]
        assert len(cut) <= 1, f"seed {s}: {cut}"
        held = [v not in kept_n for v in tvid]
        kept = [v in kept_n and v not in cut for v in tvid]
        hv, _ = load(a.runs, "V050", s, "test", "best", test_refs)
        hr, _ = load(a.runs, "R050", s, "test", "best", test_refs)
        diff = lambda keep: bleu(hv, test_refs, keep) - bleu(hr, test_refs, keep)
        vwords = {w for t in vt for w in content_words(detokenize(t).lower())}
        rwords = {w for t in rt for w in content_words(detokenize(t).lower())}
        both = [all(w in vwords and w in rwords for w in content_words(detokenize(r).lower())) for r in test_refs]
        cellU = lambda c: [x and u for x, u in zip(c, unseen)]
        cellB = lambda c: [x and b for x, b in zip(c, both)]
        out["held"].append(diff(held)); out["kept"].append(diff(kept))
        out["off"].append(diff(held) - diff(kept))
        out["uns"].append(diff(cellU(held)) - diff(cellU(kept)))
        out["both"].append(diff(cellB(held)) - diff(cellB(kept)))
        vs = {norm(t) for t in vt}; rs = {norm(t) for t in rt}
        seen_v = lambda c: sum(x and norm(r) in vs for x, r in zip(c, test_refs))
        seen_r = lambda c: sum(x and norm(r) in rs for x, r in zip(c, test_refs))
        out["seen"].append((seen_v(held) - seen_r(held)) - (seen_v(kept) - seen_r(kept)))
        print(f"seed {s}: cut {cut} | held {sum(held)} (unseen {sum(cellU(held))}) kept {sum(kept)} "
              f"(unseen {sum(cellU(kept))}) | single diffs held {out['held'][-1]:+.3f} kept {out['kept'][-1]:+.3f}")
    print(fmt("DiD official (co-primary)", out["off"]))
    print(fmt("DiD unseen (co-primary)", out["uns"]))
    print(fmt("DiD both-arms-lexicon (secondary)", out["both"]))
    print(f"seen-count DiD per seed: {out['seen']}")
    ro, ru = resolved(out["off"]), resolved(out["uns"])
    if ro == -1 or ru == -1:
        r = "R1 full vs no exposure to a clip's own episode changes BLEU (sum of two half-dose contrasts)"
        if ru != -1: r += " -- official only: name verbatim-target exposure as part of familiarity"
    elif ro == 0 and ru == 0 and abs(st.mean(out["off"])) < 0.5 and abs(st.mean(out["uns"])) < 0.5:
        r = "R2 no resolvable episode-familiarity effect"
    else:
        r = "R3 bounded, inconclusive estimate"
    print("READING Q2:", r)


def selftest():
    j = json.load(open(os.path.join(HERE, "..", "results", "pose_augmentation_decodes", "pa_aug_s42", "beam5_test.json")))
    b = bleu(j["beam_hyps"], j["refs"]); assert abs(b - 11.443) < 0.002, b
    assert abs(ols_slope([1, 2, 3], [1, 3, 5]) - 2.0) < 1e-12
    assert resolved([1.0, 1.1, 0.9, 1.2, 1.05]) == 1 and resolved([-1, 1, -1, 1, 0.1]) == 0
    print(f"selftest OK: reproduces pa_aug_s42 beam-5 test {b:.3f}; slope and resolution helpers correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifests"); ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    train_paths, train_texts = read_manifest(a.train); test_paths, test_texts = read_manifest(a.test)
    assert len(train_paths) == 12439 and len(test_paths) == 800
    test_refs = [" ".join(t.split()) for t in test_texts]
    full = {norm(t) for t in train_texts}
    unseen = [norm(r) not in full for r in test_refs]
    print(f"unseen test clips (vs the FULL 12,439 train text): {sum(unseen)}")
    q1(a, test_refs, unseen)
    q2(a, test_paths, test_refs, unseen, train_paths, train_texts)


if __name__ == "__main__":
    sys.exit(main())
