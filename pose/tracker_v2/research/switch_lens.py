"""Where do person switches concentrate, and does the translator's error track them? (measurement lens, 2026-09-25)

No training, no new decode. Reads only files already in the repo (plus, optionally, the 60 tracker-v2 ground-truth
all-detection npz in a session scratchpad, for the proxy check).

DEFINITIONS (fixed before any number below was computed)
* switch exposure of a clip = frac_chosen_nonzero in results/ndet_per_clip.csv (job 8050): the fraction of frames on which
  the v1 person tracker overrode detection 0. Every model in the ladder, S6 included, trained and decoded on detection 0,
  and S6's RGB crops are also placed with those keypoints (results/dwpose_experiment/extract_rgb_feats.py), so an override
  frame is, to the tracker's precision, a frame whose pose (and hand crops) follow someone other than the signer.
  Proxy precision: v1 is right on 91% of decisive overrides (switch_video_89_193/README.md); section 0 checks the proxy
  against the tracker-v2 ground truth. "switched clip" = exposure > 0. Bins as in tracking_dose_response.py:
  none 0 / light (0,0.1] / medium (0.1,0.3] / heavy >0.3.
* bystanders = number of Non_Signer rows for the clip in Signer_Detection.xlsx (labelled other people in the scene).
* seen = the reference occurs verbatim among the official train manifest's targets.
* per-clip quality = sentence chrF (sacreBLEU, detokenised with slt.metrics.detokenize) [primary] and content-word recall
  (content_failure_analysis.content with stitch_lexicon.STOP; clips with no reference content word excluded) [secondary].
* PRIMARY MODEL (Q2): OLS  quality ~ exposure + ref_words + seen + log(frames),  95% CI by cluster bootstrap over videos
  (2,000 resamples, percentile). Robustness: + Signer_ID group (1 / 2 / other); + video fixed effects; binned exposure.
  A slope whose interval includes 0 is reported as null.
* Q3 "at stake": (a) share of test clips / reference tokens in each bin and subset BLEU; (b) matched substitution:
  replace each switched clip's sentence-level BLEU statistics with those of a random UNSWITCHED clip from the same
  (seen, reference-length bucket) cell, recompute corpus BLEU-4; video bootstrap outside, 2,000 draws. This is an upper
  bound on what fixing switches could return IF the whole matched deficit were caused by the switch. (c) wave 3 realised:
  per seed, swap in the tracked arm's hypotheses on switched clips only.

Usage: switch_lens.py [--gt_npz DIR] [--boot 2000]
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/tools"), str(ROOT / "results/tracker_v2")]
from slt.metrics import corpus_bleu4, detokenize
import sacrebleu
from sacrebleu.metrics import BLEU
from stitch_lexicon import STOP
from content_failure_analysis import content

R = ROOT / "results"
XLSX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx"
SDX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Signer Detection/Signer_Detection.xlsx"
W3 = R / "switch_video_89_193/w3_val"
BINS = (("none", lambda f: f == 0), ("light", lambda f: 0 < f <= 0.1),
        ("medium", lambda f: 0.1 < f <= 0.3), ("heavy", lambda f: f > 0.3))
BLEU13 = BLEU(tokenize="13a")
OUT = {}


def binname(f):
    return next(n for n, t in BINS if t(f))


def clips_of(p):
    return [(l.split("\t")[0].rsplit("/", 1)[1][:-4], l.rstrip("\n").split("\t")[1]) for l in open(p) if "\t" in l]


def chrf(h, r):
    return sacrebleu.sentence_chrf(detokenize(h), [detokenize(r)]).score


def recall(h, r):
    rc, hc = content(r, STOP), content(h, STOP)
    return math.nan if not rc else sum((Counter(rc) & Counter(hc)).values()) / len(rc)


def sent_stats(hyps, refs):
    return np.array(BLEU13._extract_corpus_statistics([detokenize(h) for h in hyps], [[detokenize(r) for r in refs]]))


def bleu_from(stats):
    return BLEU13._compute_score_from_stats([int(x) for x in stats.sum(0)]).score


def lenbucket(n):
    return min(n, 8) if n >= 6 else n          # 1..5 exact, 6-7 -> 6/7, 8+ -> 8


def ols(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def demean(M, g):
    M = M.astype(float).copy()
    for k in np.unique(g):
        m = g == k; M[m] -= M[m].mean(0)
    return M


def cluster_boot(fn, groups, rng, B):
    """fn(idx, copy_id) -> scalar or vector; resample clusters with replacement; copy_id labels each drawn cluster copy."""
    uniq = np.unique(groups); pos = {g: np.flatnonzero(groups == g) for g in uniq}; out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([pos[g] for g in pick]); cid = np.concatenate([np.full(len(pos[g]), j) for j, g in enumerate(pick)])
        out.append(fn(idx, cid))
    return np.array(out)


def pct(a):
    return [float(np.percentile(a, 2.5, axis=0)), float(np.percentile(a, 97.5, axis=0))]


def fmt(m, ci):
    return f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


# ------------------------------------------------------------------------------------------------------------ self-check
def selfcheck():
    rng = np.random.default_rng(0)
    # 1. cluster bootstrap recovers a known slope and its CI covers it
    g = np.repeat(np.arange(40), 20); x = rng.random(800); y = 2.0 - 3.0 * x + rng.normal(0, 1, 40)[g] + rng.normal(0, 1, 800)
    X = np.c_[np.ones(800), x]; b = ols(y, X)[1]
    bs = cluster_boot(lambda i, c: ols(y[i], X[i])[1], g, rng, 300); lo, hi = pct(bs)
    assert abs(b + 3) < 0.5 and lo < -3 < hi, (b, lo, hi)
    # 2. FE demeaning removes a cluster offset exactly
    assert np.allclose(demean(np.c_[np.r_[1., 3, 10, 12]], np.array([0, 0, 1, 1])).ravel(), [-1, 1, -1, 1])
    # 3. summed sentence stats reproduce sacreBLEU corpus BLEU; substituting a clip's stats by its own is a no-op
    h, r = ["i am possum .", "look .", "that is great ."], ["i am possum .", "look at that .", "that is good ."]
    st = sent_stats(h, r); assert abs(bleu_from(st) - corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r])) < 1e-9
    # 4. content recall
    assert recall("the cat sat .", "a cat sat on the mat .") == 2 / 3 and math.isnan(recall("x .", "the ."))
    assert lenbucket(3) == 3 and lenbucket(7) == 7 and lenbucket(12) == 8 and binname(0.0) == "none" and binname(0.31) == "heavy"
    print("SELF-CHECK OK")


# -------------------------------------------------------------------------------------------------------------- data
def load():
    nd = {r["clip"]: {k: float(v) for k, v in r.items() if k != "clip"} for r in csv.DictReader(open(R / "ndet_per_clip.csv"))}
    x = pd.read_excel(XLSX)
    # the tracked extraction holds a mis-staged NON-signer crop 'video_12_62_3_nonsigner' in place of train clip video_12_62,
    # which is one of the 2 xlsx train clips absent from the official manifests (asserted in gates); drop both.
    assert len(x) == 14041 and set(x.Video_Clip_Name) ^ set(nd) == {"video_12_62", "video_12_62_3_nonsigner"}
    x = x[x.Video_Clip_Name != "video_12_62"]
    sd = pd.read_excel(SDX)
    bys = Counter(sd.loc[sd.Label == "Non_Signer", "Video_Clip_Name"])
    df = pd.DataFrame({"clip": x.Video_Clip_Name, "video": x.Video_Name, "split": x.Split, "signer": x.Signer_ID,
                       "text": x.Subtitle})
    df["frames"] = df["clip"].map(lambda c: nd[c]["frames"]).astype(int)
    df["frac"] = df["clip"].map(lambda c: nd[c]["frac_chosen_nonzero"])
    df["changes"] = df["clip"].map(lambda c: nd[c]["chosen_changes"]).astype(int)
    df["frac_multi"] = df["clip"].map(lambda c: nd[c]["frac_multi"])
    df["byst"] = df["clip"].map(lambda c: bys.get(c, 0)).astype(int)
    df["sw"] = df.frac > 0
    df["swf"] = df.frac * df.frames                                   # switched frames
    df["bin"] = df.frac.map(binname)
    df["sgrp"] = df.signer.map(lambda s: str(s) if s in (1, 2) else "other")
    df.attrs["dropped_byst"] = bys.get("video_12_62", 0)
    return df.set_index("clip", drop=False)


def gates(df):
    g = {}
    # bystander figure recorded in CLAUDE.md 09-22k(7): 88.4% of Communication clips have a labelled bystander (12,410)
    d = df.attrs["dropped_byst"]                                      # add back the one dropped clip
    g["clips_with_bystander"] = int((df.byst > 0).sum()) + (d > 0); assert g["clips_with_bystander"] == 12410, g
    g["nonsigner_crops"] = int(df.byst.sum()) + d; assert g["nonsigner_crops"] == 14803, g
    # prevalence recorded by wave 3 (w3_val/tracking_prevalence.json)
    prev = json.load(open(W3 / "tracking_prevalence.json"))["agg"]
    man = {s: clips_of(R / f"real_data_learning_curve/{s}.tsv") for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        c = [k for k, _ in man[s]]; sub = df.loc[c]
        got = (len(c), int(sub.sw.sum()), int(sub.frames.sum()), int(round(sub.swf.sum())))
        want = (prev[s]["clips"], prev[s]["sw_clips"], prev[s]["frames"], prev[s]["sw_frames"])
        assert got == want, (s, got, want)
        assert [t for _, t in man[s]] == list(sub.text), f"{s} manifest text != xlsx Subtitle"
    g["prevalence_matches_wave3"] = True
    assert "video_12_62" not in {k for s in man.values() for k, _ in s}
    return g, man


def load_dump(path, key, refs):
    rows = [json.loads(l) for l in open(path)]
    assert [r["idx"] for r in rows] == list(range(len(refs))) and [r["ref"] for r in rows] == refs, f"{path} misaligned"
    return [r[key] for r in rows]


def w3(split, arm, seed, refs):
    d = json.load(open(W3 / (f"{arm}_s{seed}.json" if split == "val" else f"{arm}_s{seed}_test.json")))
    assert d["refs"] == refs and len(d["beam_hyps"]) == 800
    got = corpus_bleu4([detokenize(x) for x in d["beam_hyps"]], [detokenize(x) for x in refs])
    assert abs(got - d["beam_result"]["metrics"]["tokenize=13a;detok=True"]) < 1e-6, f"w3 gate {arm} s{seed} {split}"
    return d["beam_hyps"]


# -------------------------------------------------------------------------------------------------------- section 0
def proxy_check(df, gt_dir):
    """Does frac_chosen_nonzero track the TRUE share of frames where detection 0 is not the signer? (60 GT clips)"""
    if not gt_dir or not Path(gt_dir).is_dir():
        print("\n[0] proxy check SKIPPED (no --gt_npz directory)"); return None
    import score_gt
    from scipy.stats import spearmanr
    key = json.load(open(R / "tracker_v2/gt_key.json")); lab = json.load(open(R / "tracker_v2/gt_labels.json"))
    idx0 = score_gt.base_trackers()["index0"]; per = defaultdict(Counter); cache = {}
    for pid, k in key.items():
        l = lab[pid]
        if l == "unclear":
            continue
        c = k["clip"]
        if c not in cache:
            d = dict(np.load(Path(gt_dir) / f"{c}.npz")); cache[c] = (d, idx0(d))
        d, ch = cache[c]
        truth = None if l == "none" else k["letters"][l]
        per[c][score_gt.outcome(int(ch[k["frame"]]), truth, d["DET_BOX"][k["frame"]])] += 1
    rows = [(c, df.at[c, "frac"], (v["WRONG"]) / sum(v.values())) for c, v in per.items()]
    fr, wr = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    rho = spearmanr(fr, wr)
    res = {"clips": len(rows), "panels": int(sum(sum(v.values()) for v in per.values())),
           "spearman": [float(rho.statistic), float(rho.pvalue)],
           "mean_true_idx0_wrong_by_bin": {b: [float(wr[[binname(f) == b for f in fr]].mean()),
                                               float(fr[[binname(f) == b for f in fr]].mean())]
                                           for b in ("light", "medium", "heavy")},
           "overall_true_vs_proxy": [float(wr.mean()), float(fr.mean())]}
    print(f"\n[0] PROXY CHECK on {res['clips']} tracker-v2 ground-truth clips ({res['panels']} labelled panels)")
    print(f"    Spearman(frac_chosen_nonzero, true share of frames with detection 0 on a non-signer) = "
          f"{rho.statistic:.3f} (p {rho.pvalue:.2g})")
    for b, (t, p) in res["mean_true_idx0_wrong_by_bin"].items():
        print(f"    {b:6s}: true index-0 wrong {t:.3f} vs proxy {p:.3f}")
    return res


# -------------------------------------------------------------------------------------------------------- section 1
def prevalence(df, rng, B):
    print("\n[1] PREVALENCE over all 14,041 Communication clips")
    res = {}
    tot = {"clips": len(df), "switched_clips": int(df.sw.sum()), "frames": int(df.frames.sum()), "switched_frames": float(df.swf.sum())}
    tot["clip_rate"] = tot["switched_clips"] / tot["clips"]; tot["frame_share"] = tot["switched_frames"] / tot["frames"]
    res["overall"] = tot
    print(f"    switched clips {tot['switched_clips']}/{tot['clips']} = {tot['clip_rate']:.1%}; "
          f"switched frames {tot['frame_share']:.2%} of all frames; bins "
          + ", ".join(f"{b} {int((df.bin == b).sum())}" for b, _ in BINS))
    vid = df.video.to_numpy()

    def rate_ci(mask_fn, label):
        """switched-clip rate and switched-frame share within a stratum, video-cluster bootstrap."""
        sub = df[mask_fn(df)]
        v = sub.video.to_numpy(); sw = sub.sw.to_numpy().astype(float); swf = sub.swf.to_numpy(); fr = sub.frames.to_numpy()
        bs = cluster_boot(lambda i, c: (sw[i].mean(), swf[i].sum() / fr[i].sum()), v, rng, B)
        r = {"n": len(sub), "videos": int(sub.video.nunique()), "clip_rate": float(sw.mean()), "clip_rate_ci": pct(bs[:, 0]),
             "frame_share": float(swf.sum() / fr.sum()), "frame_share_ci": pct(bs[:, 1]),
             "heavy_rate": float((sub.bin == "heavy").mean())}
        print(f"    {label:22s} n {r['n']:5d} ({r['videos']:2d} videos)  switched clips {r['clip_rate']:.1%} "
              f"[{r['clip_rate_ci'][0]:.1%}, {r['clip_rate_ci'][1]:.1%}]  switched frames {r['frame_share']:.2%} "
              f"[{r['frame_share_ci'][0]:.2%}, {r['frame_share_ci'][1]:.2%}]  heavy {r['heavy_rate']:.1%}")
        return r

    # --- by video: concentration
    pv = df.groupby("video").agg(n=("sw", "size"), rate=("sw", "mean"), swf=("swf", "sum"), fr=("frames", "sum"))
    pv["share"] = pv.swf / pv.fr
    top = pv.sort_values("swf", ascending=False)
    k = max(1, round(0.1 * len(pv)))
    conc = {"videos": len(pv), "rate_min_median_max": [float(pv.rate.min()), float(pv.rate.median()), float(pv.rate.max())],
            "top10pct_videos_share_of_switched_frames": float(top.swf.iloc[:k].sum() / pv.swf.sum()),
            "top10pct_videos_share_of_frames": float(top.fr.iloc[:k].sum() / pv.fr.sum()),
            "videos_with_no_switched_clip": int((pv.rate == 0).sum())}
    # between-video variance of the per-clip switched indicator vs a permutation null (shuffle clips across videos)
    sw = df.sw.to_numpy().astype(float); codes = pd.factorize(vid)[0]; n = np.bincount(codes)

    def between(labels):
        m = np.bincount(labels, weights=sw) / n
        return float(np.sum(n * (m - sw.mean()) ** 2) / np.sum((sw - sw.mean()) ** 2))

    obs = between(codes); null = np.array([between(rng.permutation(codes)) for _ in range(1000)])
    conc["between_video_share_of_variance"] = obs; conc["null_p95"] = float(np.percentile(null, 95))
    conc["perm_p"] = float((1 + (null >= obs).sum()) / 1001)
    conc["top5"] = [(v, int(r.n), float(r.rate), float(r.share)) for v, r in top.head(5).iterrows()]
    conc["lowest5_rate"] = [(v, int(r.n), float(r.rate)) for v, r in pv.sort_values("rate").head(5).iterrows()]
    res["by_video"] = conc
    print(f"    BY VIDEO ({conc['videos']} videos): switched-clip rate min/median/max "
          + "/".join(f"{x:.1%}" for x in conc["rate_min_median_max"])
          + f"; the top 10% of videos hold {conc['top10pct_videos_share_of_switched_frames']:.1%} of switched frames "
            f"({conc['top10pct_videos_share_of_frames']:.1%} of frames); videos with none: {conc['videos_with_no_switched_clip']}")
    print(f"    between-video share of variance in 'switched' {obs:.3f} vs shuffle null p95 {conc['null_p95']:.3f} "
          f"(perm p {conc['perm_p']:.3f}); top by switched frames: "
          + "; ".join(f"{v} n{a} {b:.0%} clips {c:.1%} frames" for v, a, b, c in conc["top5"]))
    # --- strata
    res["by_signer"] = {}
    counts = df.signer.value_counts()
    for s in [1, 2] + [s for s in counts.index if s not in (1, 2) and counts[s] >= 150]:
        res["by_signer"][str(s)] = rate_ci(lambda d, s=s: d.signer == s, f"Signer_ID {s}")
    res["by_signer"]["other(all)"] = rate_ci(lambda d: ~d.signer.isin([1, 2]), "Signer_ID other (all)")
    res["by_bystanders"] = {}
    for b in (0, 1, 2, 3):
        res["by_bystanders"][str(b)] = rate_ci((lambda d, b=b: d.byst == b) if b < 3 else (lambda d: d.byst >= 3),
                                               f"bystanders {b}{'+' if b == 3 else ''}")
    q = np.quantile(df.frames, [0.25, 0.5, 0.75]); edges = [0, *q, np.inf]
    res["length_quartile_edges"] = [float(x) for x in q]; res["by_length"] = {}
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        res["by_length"][f"Q{i + 1}"] = rate_ci(lambda d, lo=lo, hi=hi: (d.frames > lo) & (d.frames <= hi),
                                                f"frames Q{i + 1} ({lo:.0f},{hi:.0f}]")
    res["by_split"] = {s: rate_ci(lambda d, s=s: d.split == s, f"split {s}") for s in ("train", "dev", "test")}
    # --- joint: logistic model, odds ratios with video-cluster bootstrap
    from sklearn.linear_model import LogisticRegression
    X = np.c_[np.log2(df.frames), df.byst == 1, df.byst == 2, df.byst >= 3, df.sgrp == "2", df.sgrp == "other"].astype(float)
    names = ["log2 frames (x2 length)", "bystanders 1 vs 0", "bystanders 2 vs 0", "bystanders 3+ vs 0",
             "Signer_ID 2 vs 1", "Signer_ID other vs 1"]
    y = sw.astype(int)
    fit = lambda i: LogisticRegression(C=np.inf, max_iter=2000).fit(X[i], y[i]).coef_[0]
    b0 = fit(np.arange(len(y))); bs = cluster_boot(lambda i, c: fit(i), vid, rng, min(B, 400))
    res["logistic_OR"] = {nm: [float(np.exp(b0[j])), [float(np.exp(np.percentile(bs[:, j], 2.5))),
                                                       float(np.exp(np.percentile(bs[:, j], 97.5)))]] for j, nm in enumerate(names)}
    print("    JOINT LOGISTIC (switched ~ log2 frames + bystanders + signer), odds ratio [video-bootstrap 95%]:")
    for nm, (o, ci) in res["logistic_OR"].items():
        print(f"      {nm:26s} {o:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    return res


# -------------------------------------------------------------------------------------------------------- section 2
def clip_table(df, man_split, hyps, seen_set):
    c = [k for k, _ in man_split]; refs = [t for _, t in man_split]
    t = df.loc[c, ["clip", "video", "frames", "frac", "bin", "sgrp", "byst"]].reset_index(drop=True)
    t["chrf"] = [chrf(h, r) for h, r in zip(hyps, refs)]
    t["recall"] = [recall(h, r) for h, r in zip(hyps, refs)]
    t["ref_words"] = [sum(w.isalpha() for w in r.split()) for r in refs]
    t["seen"] = [r in seen_set for r in refs]
    return t


def design(t, extra=()):
    cols = [np.ones(len(t)), t.frac.to_numpy(), t.ref_words.to_numpy(), t.seen.to_numpy(float), np.log(t.frames.to_numpy())]
    names = ["const", "switch_frac", "ref_words", "seen", "log_frames"]
    if "signer" in extra:
        cols += [(t.sgrp == "2").to_numpy(float), (t.sgrp == "other").to_numpy(float)]; names += ["signer2", "signer_other"]
    if "split" in extra:                      # heterogeneity: does the slope differ between test and validation?
        te = (t.split == "test").to_numpy(float)
        cols += [te, te * t.frac.to_numpy()]; names += ["is_test", "switch_frac_x_test"]
    if "bins" in extra:
        cols[1:2] = [(t.bin == b).to_numpy(float) for b in ("light", "medium", "heavy")]
        names[1:2] = ["light", "medium", "heavy"]
    return np.column_stack(cols), names


def regress(t, y, rng, B, extra=(), fe=False, label=""):
    ok = ~np.isnan(y); t = t[ok].reset_index(drop=True); y = y[ok]
    X, names = design(t, extra); g = t.video.to_numpy()
    if fe:
        X, names = X[:, 1:], names[1:]

    def est(i, cid):
        Xi, yi = X[i], y[i]
        if fe:
            Xi, yi = demean(Xi, cid), demean(yi[:, None], cid).ravel()
        return ols(yi, Xi)
    b = est(np.arange(len(y)), g if fe else None)
    bs = cluster_boot(est, g, rng, B)
    out = {nm: [float(b[j]), pct(bs[:, j])] for j, nm in enumerate(names) if nm in ("switch_frac", "switch_frac_x_test", "light", "medium", "heavy", "seen", "ref_words")}
    out["n"] = int(len(y)); out["mean_y"] = float(y.mean())
    key = [k for k in ("switch_frac", "switch_frac_x_test", "heavy", "medium", "light") if k in out]
    print(f"    {label:52s} n {len(y):4d} mean {y.mean():6.2f} | " + " | ".join(f"{k} {fmt(*out[k])}" for k in key)
          + f" | seen {fmt(*out['seen'])}")
    return out


def quality(df, man, rng, B):
    seen_set = {t for _, t in man["train"]}
    res = {}
    print("\n[2] DOES PER-CLIP QUALITY TRACK SWITCH EXPOSURE?  (coefficients in outcome units; switch_frac per 1.0 = all frames)")
    tables = {}
    for split, rec_bleu, n_seen in (("test", 11.802, 230), ("val", 12.470, 251)):
        refs = [t for _, t in man[split]]
        hyps = load_dump(R / f"dumps_S6_{split}/hyps_LP=0.3.jsonl", "hyp_beam", refs)
        got = corpus_bleu4([detokenize(x) for x in hyps], [detokenize(x) for x in refs])
        assert abs(got - rec_bleu) < 1e-3, f"S6 {split} gate: {got}"
        assert abs(bleu_from(sent_stats(hyps, refs)) - got) < 1e-9
        t = clip_table(df, man[split], hyps, seen_set); assert int(t.seen.sum()) == n_seen, (split, t.seen.sum())
        tables[split] = (t, hyps, refs)
        print(f"  S6 {split}: re-scores to BLEU-4 {got:.3f} (recorded {rec_bleu}); seen {n_seen}")
    # signer gate: 09-23d recorded S6 test BLEU by signer 14.39 / 9.97 / 3.12
    t, hyps, refs = tables["test"]
    for s, want in (("1", 14.39), ("2", 9.97), ("other", 3.12)):
        i = np.flatnonzero(t.sgrp == s)
        got = corpus_bleu4([detokenize(hyps[j]) for j in i], [detokenize(refs[j]) for j in i])
        assert abs(got - want) < 0.01, (s, got)
    print("  signer join gate: S6 test BLEU by Signer_ID reproduces 14.39 / 9.97 / 3.12")
    # balance table: are switched clips different on the controls?
    bal = t.groupby("bin").agg(n=("clip", "size"), ref_words=("ref_words", "mean"), seen=("seen", "mean"),
                               frames=("frames", "median"), signer1=("sgrp", lambda s: (s == "1").mean()),
                               byst=("byst", "mean"), chrf=("chrf", "mean"), recall=("recall", "mean")).reindex([b for b, _ in BINS])
    print("  test balance by bin (raw means):\n" + bal.round(3).to_string())
    res["test_balance"] = bal.round(4).reset_index().to_dict(orient="records")
    for split in ("test", "val"):
        t = tables[split][0]; r = res[f"S6_{split}"] = {}
        print(f"  --- S6 {split}")
        for yname in ("chrf", "recall"):
            y = t[yname].to_numpy(float) * (100 if yname == "recall" else 1)
            r[yname] = {"primary": regress(t, y, rng, B, label=f"{yname} PRIMARY (ref_words+seen+log frames)"),
                        "signer": regress(t, y, rng, B, ("signer",), label=f"{yname} + Signer_ID group"),
                        "video_FE": regress(t, y, rng, B, fe=True, label=f"{yname} + video fixed effects"),
                        "bins": regress(t, y, rng, B, ("bins",), label=f"{yname} binned (vs none)")}
    # pooled test+val (more power), S6
    tp = pd.concat([tables["test"][0].assign(split="test"), tables["val"][0].assign(split="val")], ignore_index=True)
    print("  --- S6 test+val pooled")
    res["S6_pooled"] = {}
    for y in ("chrf", "recall"):
        v = tp[y].to_numpy(float) * (100 if y == "recall" else 1)
        res["S6_pooled"][y] = {"primary": regress(tp, v, rng, B, label=f"{y} pooled PRIMARY"),
                               "signer": regress(tp, v, rng, B, ("signer",), label=f"{y} pooled, + Signer_ID group"),
                               "video_FE": regress(tp, v, rng, B, fe=True, label=f"{y} pooled, + video fixed effects")}
    res["S6_pooled"]["chrf_split_interaction"] = regress(tp, tp.chrf.to_numpy(float), rng, B, ("split", "signer"),
                                                        label="chrf pooled + signer + test x switch_frac")
    res["S6_pooled"]["chrf_bins"] = regress(tp, tp.chrf.to_numpy(float), rng, B, ("bins", "signer"), label="chrf pooled binned + signer")
    # replication with 5 single models trained and decoded on index-0 pose (wave 3 control arm): per-clip mean over seeds
    print("  --- wave-3 index-0 models (w3C, 5 seeds, beam 5), per-clip chrF averaged over seeds")
    w3tab = {}
    for split in ("test", "val"):
        refs = tables[split][2]; t = tables[split][0].copy()
        C = {s: w3(split, "w3C", s, refs) for s in (42, 101, 202, 303, 404)}
        T = {s: w3(split, "w3T", s, refs) for s in (42, 101, 202, 303, 404)}
        t["chrf_C"] = np.mean([[chrf(h, r) for h, r in zip(C[s], refs)] for s in C], 0)
        t["chrf_T"] = np.mean([[chrf(h, r) for h, r in zip(T[s], refs)] for s in T], 0)
        w3tab[split] = (t, C, T)
        res[f"w3C_{split}"] = regress(t, t.chrf_C.to_numpy(), rng, B, ("signer",), label=f"w3C {split} chrF + signer")
    # quasi-causal lens: same clips, input changed (tracked v1 vs index-0). delta_T-C per clip vs exposure.
    print("  --- wave-3 tracked minus index-0 (per-clip chrF, seed-mean): does the benefit of tracking grow with exposure?")
    for split in ("test", "val"):
        t = w3tab[split][0]
        res[f"w3_TminusC_{split}"] = {
            "slope": regress(t, (t.chrf_T - t.chrf_C).to_numpy(), rng, B, label=f"w3 T-C {split}: slope on switch_frac"),
            "bins": regress(t, (t.chrf_T - t.chrf_C).to_numpy(), rng, B, ("bins",), label=f"w3 T-C {split}: bins vs none")}
    return res, tables, w3tab


# -------------------------------------------------------------------------------------------------------- section 3
def at_stake(tables, w3tab, rng, B):
    print("\n[3] HOW MUCH TEST BLEU-4 COULD PLAUSIBLY BE AT STAKE?")
    res = {}
    for split in ("test", "val"):
        res[split] = stake_split(split, *tables[split], rng, B)
    res.update(w3_realised(tables, w3tab, rng))
    return res


def stake_split(split, t, hyps, refs, rng, B):
    print(f"  --- S6 {split}")
    res = {}
    st = sent_stats(hyps, refs); base = bleu_from(st); g = t.video.to_numpy()
    comp = {}
    for b, _ in BINS:
        i = np.flatnonzero(t.bin == b)
        sub = st[i]
        boot = cluster_boot(lambda j, c: bleu_from(sub[j]), g[i], rng, B)
        comp[b] = {"clips": len(i), "clip_share": len(i) / len(t), "ref_token_share": float(sub[:, 1].sum() / st[:, 1].sum()),
                   "bleu": bleu_from(sub), "bleu_ci": pct(boot)}
        print(f"    {b:6s} {len(i):3d} clips ({len(i) / len(t):.1%}; {comp[b]['ref_token_share']:.1%} of reference tokens) "
              f"S6 BLEU-4 {comp[b]['bleu']:6.2f} [{comp[b]['bleu_ci'][0]:.2f}, {comp[b]['bleu_ci'][1]:.2f}]")
    res["composition"] = comp
    # matched substitution with an outer video bootstrap
    cell = np.array([f"{int(s)}_{lenbucket(n)}" for s, n in zip(t.seen, t.ref_words)])

    def cf(idx, cid, target):
        """on a bootstrap sample idx: replace stats of target clips with those of a random unswitched clip, same cell."""
        s = st[idx].copy(); tb = t.bin.to_numpy()[idx]; ce = cell[idx]
        donors = defaultdict(list)
        for k in range(len(idx)):
            if tb[k] == "none": donors[ce[k]].append(k)
        for k in range(len(idx)):
            if tb[k] in target and donors[ce[k]]:
                s[k] = st[idx][donors[ce[k]][rng.integers(len(donors[ce[k]]))]]
        return bleu_from(s) - bleu_from(st[idx])
    for name, target in (("all switched", ("light", "medium", "heavy")), ("heavy only", ("heavy",))):
        point = np.mean([cf(np.arange(len(t)), None, target) for _ in range(200)])
        boot = cluster_boot(lambda j, c: cf(j, c, target), g, rng, B)
        res[f"matched_sub_{name}"] = {"delta": float(point), "ci": pct(boot)}
        print(f"    matched substitution ({name}): BLEU-4 {base:.2f} -> {base + point:.2f}, delta {fmt(point, pct(boot))} "
              "(upper bound: assumes the matched deficit is all caused by the switch)")
    return res


def w3_realised(tables, w3tab, rng):
    """Wave 3, per seed: corpus BLEU change when ONLY the switched clips take the tracked arm's hypotheses, minus a placebo
    (the same swap on a random, equally sized set of UNSWITCHED clips, where the input is identical in both arms, so any
    change there is the training difference alone; 200 draws per seed). excess = realised - placebo."""
    res = {}
    tcrit = 2.776
    ci = lambda d: [float(np.mean(d) - tcrit * np.std(d, ddof=1) / 5 ** .5), float(np.mean(d) + tcrit * np.std(d, ddof=1) / 5 ** .5)]
    print("  --- wave 3 realised (v1 tracker, 5 seeds): swap tracked hypotheses in on switched clips only")
    for split in ("test", "val"):
        tt, C, T = w3tab[split]; rf = [detokenize(x) for x in tables[split][2]]
        sw = np.flatnonzero((tt.bin != "none").to_numpy()); un = np.flatnonzero((tt.bin == "none").to_numpy())
        d_sw, d_pl = [], []
        for s in C:
            sC, sT = sent_stats(C[s], tables[split][2]), sent_stats(T[s], tables[split][2]); bC = bleu_from(sC)

            def swap(ix):
                m = sC.copy(); m[ix] = sT[ix]; return bleu_from(m) - bC
            d_sw.append(swap(sw)); d_pl.append(np.mean([swap(rng.choice(un, len(sw), replace=False)) for _ in range(200)]))
        ex = [a - b for a, b in zip(d_sw, d_pl)]
        res[f"w3_{split}"] = {"switched_swap": [float(np.mean(d_sw)), ci(d_sw)], "placebo_swap": [float(np.mean(d_pl)), ci(d_pl)],
                              "excess": [float(np.mean(ex)), ci(ex), [round(x, 3) for x in ex]]}
        print(f"    {split}: switched-clip swap {fmt(np.mean(d_sw), ci(d_sw))}; size-matched placebo swap on unswitched "
              f"{fmt(np.mean(d_pl), ci(d_pl))}; EXCESS {fmt(np.mean(ex), ci(ex))} ({sum(x > 0 for x in ex)}/5 seeds)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_npz", default="")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    selfcheck()
    rng = np.random.default_rng(20260925)
    df = load(); g, man = gates(df); OUT["gates"] = g
    print("GATES OK:", g)
    OUT["proxy"] = proxy_check(df, a.gt_npz)
    OUT["prevalence"] = prevalence(df, rng, a.boot)
    OUT["quality"], tables, w3tab = quality(df, man, rng, a.boot)
    OUT["at_stake"] = at_stake(tables, w3tab, rng, a.boot)
    json.dump(OUT, open(Path(__file__).with_name("switch_lens_results.json"), "w"), indent=1, default=str)
    print("\nwrote switch_lens_results.json")


if __name__ == "__main__":
    main()
