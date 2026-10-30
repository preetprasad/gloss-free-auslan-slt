"""Adversarial verifier of the 'emp-where-it-matters' lens (switch_lens.py / switch_lens_controls.py), 2026-09-26.

Independent re-implementation: shares NO code with switch_lens*.py. Uses only the raw files, slt.metrics
(detokenize + corpus_bleu4, for the gates), sacrebleu (sentence chrF, 13a tokenizer, BLEU.compute_bleu), and the
STOP list constant from stitch_lexicon (a data table, not code). Every estimate is re-derived here.

Extra checks the lens did not run:
  * within-stratum proxy validity (the 60 GT clips were sampled BY proxy stratum, never at exposure 0);
  * cluster-robust (CR1) sandwich CIs next to the pairs cluster bootstrap;
  * within-video permutation test of the test slope, and a within-video test<->val label-swap test of the split
    interaction (is "sampling noise" consistent with the data?);
  * leave-one-video-out influence on the test slope;
  * matched substitution with signer added to the matching cell (does the "upper bound" shrink?);
  * exact-expectation versions of the substitution and placebo swaps (no Monte-Carlo donor noise).
Run: ~/.pyenv/versions/3.14.3/bin/python verify_switch_lens.py   (writes verify_switch_lens_output.txt via tee)
"""
import csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4          # noqa: E402  (gate functions)
import sacrebleu                                         # noqa: E402
from sacrebleu.metrics import BLEU                       # noqa: E402
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a  # noqa: E402
sys.path.insert(0, str(ROOT / "src/tools"))
from stitch_lexicon import STOP                          # noqa: E402  (data constant)

R = ROOT / "results"
GT_NPZ = Path("/private/tmp/claude-501/-Users-preet-Honours/c69cfc0c-6e97-4020-b91d-72bcf24d0e82/scratchpad/gt/npz")
XLSX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx"
SDX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Signer Detection/Signer_Detection.xlsx"
W3 = R / "switch_video_89_193/w3_val"
TOK = Tokenizer13a()
RNG = np.random.default_rng(20260926)
OUT = {}


# ------------------------------------------------------------------------------------------------ small primitives
def ngram_stats(h, r):
    """[hyp_len, ref_len, match1..4, total1..4] for one detokenised pair under 13a (single reference)."""
    ht, rt = TOK(h).split(), TOK(r).split()
    s = [len(ht), len(rt)]
    m, t = [], []
    for n in range(1, 5):
        hc = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
        rc = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
        m.append(sum((hc & rc).values())); t.append(max(len(ht) - n + 1, 0))
    return s + m + t


def bleu(stats):
    """Corpus BLEU-4 (sacreBLEU 'exp' smoothing) from summed stats; accepts non-integer (expected) counts."""
    s = np.asarray(stats, float).sum(0) if np.ndim(stats) == 2 else np.asarray(stats, float)
    sys_len, ref_len, cor, tot = s[0], s[1], s[2:6], s[6:10]
    if not cor.any():
        return 0.0
    bp = 1.0 if sys_len >= ref_len else (math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0)
    logs, sm = [], 1.0
    for c, t in zip(cor, tot):
        if t == 0:
            logs.append(-9999999999); continue
        if c == 0:
            sm *= 2; logs.append(math.log(100.0 / (sm * t)))
        else:
            logs.append(math.log(100.0 * c / t))
    return bp * math.exp(sum(logs) / 4)


def ols_fit(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    return b, XtX_inv


def cr1(y, X, g):
    """OLS with CR1 cluster-robust covariance; returns b, se, df=G-1."""
    b, XtX_inv = ols_fit(y, X); e = y - X @ b
    G = np.unique(g); meat = np.zeros((X.shape[1],) * 2)
    for k in G:
        m = g == k; u = X[m].T @ e[m]; meat += np.outer(u, u)
    n, p = X.shape; c = len(G) / (len(G) - 1) * (n - 1) / (n - p)
    V = c * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V)), len(G) - 1


def boot_clusters(g, B):
    uniq = np.unique(g); pos = {k: np.flatnonzero(g == k) for k in uniq}
    for _ in range(B):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        yield np.concatenate([pos[k] for k in pick]), np.concatenate([np.full(len(pos[k]), j) for j, k in enumerate(pick)])


def within_demean(M, g):
    M = np.asarray(M, float).copy()
    for k in np.unique(g):
        m = g == k; M[m] -= M[m].mean(0)
    return M


def q(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def tq(df):
    from scipy.stats import t
    return float(t.ppf(0.975, df))


def logit_irls(y, X, it=50):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


# ------------------------------------------------------------------------------------------------------ self-check
def selfcheck():
    # BLEU from my own n-gram stats == sacrebleu corpus BLEU on a toy set
    H = ["the cat sat on the mat .", "hello there .", "i am possum ."]
    Rf = ["the cat is on the mat .", "hello .", "i am possum ."]
    mine = bleu([ngram_stats(h, r) for h, r in zip(H, Rf)])
    ref = sacrebleu.corpus_bleu(H, [Rf], tokenize="13a").score
    assert abs(mine - ref) < 1e-9, (mine, ref)
    # CR1 on a constructed clustered design recovers the slope exactly with noiseless y
    X = np.c_[np.ones(20), np.arange(20.0)]; y = 3 + 2 * X[:, 1]
    b, se, _ = cr1(y, X, np.repeat(np.arange(5), 4)); assert abs(b[1] - 2) < 1e-9 and se.max() < 1e-6
    # IRLS reproduces a known logistic coefficient on a large simulated sample
    rs = np.random.default_rng(0); x = rs.normal(size=20000); yy = (rs.random(20000) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(float)
    bb = logit_irls(yy, np.c_[np.ones(20000), x]); assert abs(bb[1] - 1.2) < 0.08 and abs(bb[0] - 0.5) < 0.06, bb
    # within-demean removes group means
    assert np.allclose(within_demean(np.array([[1.], [3.], [10.], [14.]]), np.array([0, 0, 1, 1])).ravel(), [-1, 1, -2, 2])
    print("SELF-CHECK OK")


# ------------------------------------------------------------------------------------------------------------- data
def load():
    nd = pd.read_csv(R / "ndet_per_clip.csv").set_index("clip")
    x = pd.read_excel(XLSX).set_index("Video_Clip_Name")
    sd = pd.read_excel(SDX)
    byst = Counter(sd.loc[sd.Label == "Non_Signer", "Video_Clip_Name"])
    man = {}
    for s in ("train", "val", "test"):
        rows = [l.rstrip("\n").split("\t") for l in open(R / f"real_data_learning_curve/{s}.tsv") if l.strip()]
        man[s] = [(Path(p).stem, t) for p, t in rows]
    allc = [c for s in man for c, _ in man[s]]
    assert len(allc) == len(set(allc)) == 14039
    miss = [c for c in allc if c not in nd.index]; assert not miss, miss[:5]
    for s in man:                                             # text join: manifest == xlsx Subtitle
        assert all(x.at[c, "Subtitle"] == t for c, t in man[s]), s
        want = {"train": "train", "val": "dev", "test": "test"}[s]
        assert (x.loc[[c for c, _ in man[s]], "Split"] == want).all(), s
    df = pd.DataFrame({"clip": allc, "split": [s for s in man for _ in man[s]]}).set_index("clip", drop=False)
    df["video"] = x.loc[df.index, "Video_Name"].values
    df["signer"] = x.loc[df.index, "Signer_ID"].values
    df["frames"] = nd.loc[df.index, "frames"].astype(int).values
    df["frac"] = nd.loc[df.index, "frac_chosen_nonzero"].values
    df["frac_multi"] = nd.loc[df.index, "frac_multi"].values
    df["max_ndet"] = nd.loc[df.index, "max_ndet"].values
    df["byst"] = [byst.get(c, 0) for c in df.index]
    df["swf"] = np.round(df.frac * df.frames)
    df["sw"] = df.frac > 0
    df["bin"] = pd.cut(df.frac, [-1, 0, 0.1, 0.3, 1.01], labels=["none", "light", "medium", "heavy"]).astype(str)
    df["sgrp"] = np.where(df.signer == 1, "1", np.where(df.signer == 2, "2", "other"))
    # corpus-wide bystander gate (CLAUDE.md 09-22k(7)): all 14,041 xlsx clips
    OUT["gate_bystander_all_xlsx"] = [int(sum(byst.get(c, 0) > 0 for c in x.index)), int(sum(byst.get(c, 0) for c in x.index))]
    assert OUT["gate_bystander_all_xlsx"] == [12410, 14803], OUT["gate_bystander_all_xlsx"]
    prev = json.load(open(W3 / "tracking_prevalence.json"))["agg"]
    for s, k in (("train", "train"), ("val", "val"), ("test", "test")):
        sub = df[df.split == s]
        got = [len(sub), int(sub.sw.sum()), int(sub.frames.sum()), int(sub.swf.sum())]
        want = [prev[k]["clips"], prev[k]["sw_clips"], prev[k]["frames"], prev[k]["sw_frames"]]
        assert got == want, (s, got, want)
    print("GATES OK: manifests<->xlsx text+split, bystander 12410/14803, per-split prevalence == wave-3 record")
    return df, man


# ------------------------------------------------------------------------------------------------ Q0 proxy validity
def proxy(df):
    key = json.load(open(R / "tracker_v2/gt_key.json")); lab = json.load(open(R / "tracker_v2/gt_labels.json"))
    strata = {c: b for part in json.load(open(R / "tracker_v2/gt_clips.json")).values() for c, b in part}
    per = defaultdict(lambda: [0, 0]); cache = {}
    for pid, k in key.items():
        l = lab.get(pid)
        if l in (None, "unclear"):
            continue
        c, f = k["clip"], k["frame"]
        d = cache.get(c) or cache.setdefault(c, dict(np.load(GT_NPZ / f"{c}.npz")))
        choice = 0 if d["NDET"][f] > 0 else -1
        truth = None if l == "none" else k["letters"][l]
        if truth is None:
            wrong = choice >= 0
        elif choice < 0:
            wrong = False                                     # a miss, not a wrong person
        else:
            a, b = d["DET_BOX"][f][choice], d["DET_BOX"][f][truth]
            ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            wrong = not (choice == truth or (u > 0 and inter / u >= 0.6))
        per[c][0] += wrong; per[c][1] += 1
    from scipy.stats import spearmanr, pearsonr
    cl = sorted(per); tr = np.array([per[c][0] / per[c][1] for c in cl]); px = df.loc[cl, "frac"].to_numpy()
    st = np.array([strata[c] for c in cl])
    res = {"clips": len(cl), "panels": int(sum(v[1] for v in per.values())),
           "spearman_all": float(spearmanr(px, tr).statistic), "pearson_all": float(pearsonr(px, tr).statistic)}
    for b in ("light", "medium", "heavy"):
        m = st == b
        res[f"{b}_mean_true_vs_proxy"] = [float(tr[m].mean()), float(px[m].mean())]
        res[f"{b}_within_spearman"] = float(spearmanr(px[m], tr[m]).statistic)
    # pooled within-stratum association: Spearman of stratum-demeaned values
    res["within_stratum_pearson_demeaned"] = float(pearsonr(within_demean(px[:, None], st).ravel(),
                                                            within_demean(tr[:, None], st).ravel()).statistic)
    res["sampled_at_exposure_0"] = int((px == 0).sum())
    OUT["proxy"] = res
    print("\n[0] PROXY (60 GT clips; my own IoU>=0.6 duplicate rule, index-0 = det 0 when NDET>0)")
    for k, v in res.items():
        print(f"    {k}: {v}")


# ------------------------------------------------------------------------------------------------ Q1 prevalence
def prevalence(df):
    print("\n[1] PREVALENCE (14,039 manifest clips; lens used 14,040 xlsx clips)")
    r = {"clips": len(df), "sw": int(df.sw.sum()), "clip_rate": float(df.sw.mean()),
         "frame_share": float(df.swf.sum() / df.frames.sum()),
         "bins": df.bin.value_counts().to_dict()}
    pv = df.groupby("video").agg(n=("sw", "size"), rate=("sw", "mean"), swf=("swf", "sum"), fr=("frames", "sum"))
    top = pv.sort_values("swf", ascending=False); k = round(0.1 * len(pv))
    r["videos"] = len(pv); r["rate_min_med_max"] = [float(pv.rate.min()), float(pv.rate.median()), float(pv.rate.max())]
    r["top10pct_share_sw_frames"] = float(top.swf.iloc[:k].sum() / pv.swf.sum())
    r["top10pct_share_frames"] = float(top.fr.iloc[:k].sum() / pv.fr.sum())
    r["videos_zero"] = int((pv.rate == 0).sum())
    y = df.sw.to_numpy(float); codes = pd.factorize(df.video)[0]; n = np.bincount(codes)

    def eta2(c):
        m = np.bincount(c, weights=y) / n
        return float(np.sum(n * (m - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2))
    obs = eta2(codes); null = np.array([eta2(RNG.permutation(codes)) for _ in range(1000)])
    r["eta2_video"] = obs; r["eta2_null_p95"] = float(np.percentile(null, 95)); r["perm_p"] = float((1 + (null >= obs).sum()) / 1001)
    # ICC(1) on the switched indicator, a chance-corrected alternative to eta^2
    k0 = (len(y) - (n ** 2).sum() / len(y)) / (len(n) - 1)
    msb = np.sum(n * (np.bincount(codes, weights=y) / n - y.mean()) ** 2) / (len(n) - 1)
    msw = (np.sum((y - y.mean()) ** 2) - msb * (len(n) - 1)) / (len(y) - len(n))
    r["icc1_video"] = float((msb - msw) / (msb + (k0 - 1) * msw))

    def strat(mask, label):
        sub = df[mask]; g = sub.video.to_numpy(); sw = sub.sw.to_numpy(float); swf = sub.swf.to_numpy(); fr = sub.frames.to_numpy()
        bs = np.array([(sw[i].mean(), swf[i].sum() / fr[i].sum()) for i, _ in boot_clusters(g, 1000)])
        out = {"n": len(sub), "clip_rate": float(sw.mean()), "ci": q(bs[:, 0]), "frame_share": float(swf.sum() / fr.sum()),
               "fs_ci": q(bs[:, 1])}
        print(f"    {label:18s} n {out['n']:5d} clip {out['clip_rate']:.1%} [{out['ci'][0]:.1%},{out['ci'][1]:.1%}] "
              f"frames {out['frame_share']:.2%} [{out['fs_ci'][0]:.2%},{out['fs_ci'][1]:.2%}]")
        return out
    r["signer"] = {s: strat(df.sgrp == s, f"signer {s}") for s in ("1", "2", "other")}
    r["byst"] = {b: strat((df.byst == b) if b < 3 else (df.byst >= 3), f"bystanders {b}{'+' if b == 3 else ''}") for b in range(4)}
    qe = np.quantile(df.frames, [.25, .5, .75]); e = [0, *qe, np.inf]
    r["len"] = {f"Q{i+1}": strat((df.frames > e[i]) & (df.frames <= e[i + 1]), f"frames Q{i+1}") for i in range(4)}
    r["split"] = {s: strat(df.split == s, f"split {s}") for s in ("train", "val", "test")}
    # joint logistic, my own IRLS; video cluster bootstrap
    X = np.c_[np.ones(len(df)), np.log2(df.frames), df.byst == 1, df.byst == 2, df.byst >= 3,
              df.sgrp == "2", df.sgrp == "other"].astype(float)
    names = ["log2frames", "byst1", "byst2", "byst3+", "signer2", "signer_other"]
    b0 = logit_irls(y, X); g = df.video.to_numpy()
    bs = np.array([logit_irls(y[i], X[i]) for i, _ in boot_clusters(g, 300)])
    r["logit_OR"] = {nm: [float(np.exp(b0[j + 1])), [float(np.exp(v)) for v in q(bs[:, j + 1])]] for j, nm in enumerate(names)}
    # sensitivity the lens did not run: add frac_multi (crowding) -- does signer 2 still stand out?
    # population attributable fraction under the joint model: share of predicted switched clips removed if every clip
    # had 0 bystanders, vs if every clip were Signer_ID 1 (which factor 'drives' more switching in THIS corpus?)
    pr = lambda M: (1 / (1 + np.exp(-(M @ b0)))).sum()
    base_n = pr(X); Xb = X.copy(); Xb[:, 2:5] = 0; Xs = X.copy(); Xs[:, 5:7] = 0
    r["PAF_bystanders_to_0"] = float(1 - pr(Xb) / base_n); r["PAF_signer_to_1"] = float(1 - pr(Xs) / base_n)
    X2 = np.c_[X, df.frac_multi]; b2 = logit_irls(y, X2)
    r["logit_OR_plus_frac_multi"] = {nm: float(np.exp(b2[j + 1])) for j, nm in enumerate(names + ["frac_multi"])}
    OUT["prevalence"] = r
    for kk in ("clips", "sw", "clip_rate", "frame_share", "bins", "videos", "rate_min_med_max", "top10pct_share_sw_frames",
               "top10pct_share_frames", "videos_zero", "eta2_video", "eta2_null_p95", "perm_p", "icc1_video", "logit_OR",
               "logit_OR_plus_frac_multi", "PAF_bystanders_to_0", "PAF_signer_to_1"):
        print(f"    {kk}: {r[kk]}")


# ---------------------------------------------------------------------------------------------- Q2 per-clip quality
def content(t):
    return [w for w in t.lower().split() if w.isalpha() and w not in STOP]


def clip_table(df, man, split, hyps):
    seen = {t for _, t in man["train"]}
    c = [k for k, _ in man[split]]; refs = [t for _, t in man[split]]
    t = df.loc[c].reset_index(drop=True).copy()
    t["chrf"] = [sacrebleu.sentence_chrf(detokenize(h), [detokenize(r)]).score for h, r in zip(hyps, refs)]
    rc = [content(r) for r in refs]; hc = [content(h) for h in hyps]
    t["recall"] = [np.nan if not a else 100 * sum((Counter(a) & Counter(b)).values()) / len(a) for a, b in zip(rc, hc)]
    t["ref_words"] = [sum(w.isalpha() for w in r.split()) for r in refs]
    t["ref_tok13a"] = [len(TOK(detokenize(r)).split()) for r in refs]    # alternative length control
    t["seen"] = [r in seen for r in refs]
    t["stats"] = [ngram_stats(detokenize(h), detokenize(r)) for h, r in zip(hyps, refs)]
    return t


def design(t, signer=False, split=False, lencol="ref_words", bins=False):
    cols = [np.ones(len(t))]; names = ["const"]
    if bins:
        for b in ("light", "medium", "heavy"):
            cols.append((t.bin == b).to_numpy(float)); names.append(b)
    else:
        cols.append(t.frac.to_numpy(float)); names.append("frac")
    cols += [t[lencol].to_numpy(float), t.seen.to_numpy(float), np.log(t.frames.to_numpy(float))]
    names += [lencol, "seen", "logframes"]
    if signer:
        cols += [(t.sgrp == "2").to_numpy(float), (t.sgrp == "other").to_numpy(float)]; names += ["s2", "sother"]
    if split:
        te = (t.split == "test").to_numpy(float); cols += [te, te * t.frac.to_numpy(float)]; names += ["is_test", "frac_x_test"]
    return np.column_stack(cols), names


def fit_report(t, y, label, coef="frac", fe=False, B=2000, **kw):
    ok = ~np.isnan(y); t = t[ok].reset_index(drop=True); y = y[ok]
    X, names = design(t, **kw); g = t.video.to_numpy(); j = names.index(coef)
    if fe:
        Xd, yd = within_demean(X[:, 1:], g), within_demean(y[:, None], g).ravel()
        b, se, dfree = cr1(yd, Xd, g); j -= 1
        bs = [np.linalg.lstsq(within_demean(X[i][:, 1:], cid), within_demean(y[i][:, None], cid).ravel(), rcond=None)[0][j]
              for i, cid in boot_clusters(g, B)]
    else:
        b, se, dfree = cr1(y, X, g)
        bs = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][j] for i, _ in boot_clusters(g, B)]
    tc = tq(dfree)
    res = {"n": int(len(y)), "b": float(b[j]), "cr1_ci": [float(b[j] - tc * se[j]), float(b[j] + tc * se[j])], "boot_ci": q(bs)}
    print(f"    {label:48s} n {res['n']:4d} {coef} {res['b']:+7.2f} CR1 [{res['cr1_ci'][0]:+.2f},{res['cr1_ci'][1]:+.2f}]"
          f" boot [{res['boot_ci'][0]:+.2f},{res['boot_ci'][1]:+.2f}]")
    return res


def perm_within_video(t, y, B=2000):
    """Null: exposure exchangeable among clips of the same video. Refit the primary model on each shuffle."""
    X, names = design(t); j = names.index("frac"); g = t.video.to_numpy()
    obs = np.linalg.lstsq(X, y, rcond=None)[0][j]; groups = [np.flatnonzero(g == k) for k in np.unique(g)]
    null = []
    for _ in range(B):
        Xp = X.copy(); fr = X[:, j].copy()
        for ix in groups:
            fr[ix] = fr[RNG.permutation(ix)]
        Xp[:, j] = fr; null.append(np.linalg.lstsq(Xp, y, rcond=None)[0][j])
    null = np.array(null)
    return float(obs), float((1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1))


def split_swap_test(tp, B=2000):
    """Null: test and val clips are exchangeable within each video. Swap split labels within video (keeping each video's
    test count), refit pooled + signer + interaction; two-sided p for the frac_x_test coefficient."""
    X, names = design(tp, signer=True, split=True); j = names.index("frac_x_test"); y = tp.chrf.to_numpy(float)
    obs = np.linalg.lstsq(X, y, rcond=None)[0][j]
    g = tp.video.to_numpy(); te0 = (tp.split == "test").to_numpy(); fr = tp.frac.to_numpy(float)
    groups = [np.flatnonzero(g == k) for k in np.unique(g)]; ji = names.index("is_test"); null = []
    for _ in range(B):
        te = te0.copy()
        for ix in groups:
            te[ix] = te0[RNG.permutation(ix)]
        Xp = X.copy(); Xp[:, ji] = te; Xp[:, j] = te * fr
        null.append(np.linalg.lstsq(Xp, y, rcond=None)[0][j])
    null = np.array(null)
    return float(obs), float((1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1))


def quality(df, man):
    print("\n[2] PER-CLIP QUALITY vs SWITCH EXPOSURE (S6, beam-10 LP0.3 dumps)")
    tabs = {}
    for split, want, nseen in (("test", 11.802, 230), ("val", 12.470, 251)):
        rows = [json.loads(l) for l in open(R / f"dumps_S6_{split}/hyps_LP=0.3.jsonl")]
        refs = [t for _, t in man[split]]
        assert [r["ref"] for r in rows] == refs and [r["idx"] for r in rows] == list(range(800)), f"{split} dump misaligned"
        hyps = [r["hyp_beam"] for r in rows]
        got = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
        assert abs(got - want) < 1e-3, (split, got)
        t = clip_table(df, man, split, hyps)
        assert abs(bleu(np.stack(t.stats)) - got) < 1e-9, "own n-gram stats do not reproduce corpus BLEU"
        assert int(t.seen.sum()) == nseen, (split, t.seen.sum())
        tabs[split] = t
        print(f"  gate: S6 {split} BLEU-4 {got:.3f} (recorded {want}); own n-gram stats reproduce it; seen {nseen}")
    t = tabs["test"]
    sg = {s: bleu(np.stack(t.stats[t.sgrp == s])) for s in ("1", "2", "other")}
    assert [round(sg[s], 2) for s in ("1", "2", "other")] == [14.39, 9.97, 3.12], sg
    print(f"  gate: S6 test BLEU by Signer_ID {sg['1']:.2f} / {sg['2']:.2f} / {sg['other']:.2f}")
    print("  raw means by bin (test | val):")
    for s in ("test", "val"):
        print("   ", s, tabs[s].groupby("bin").agg(n=("chrf", "size"), chrf=("chrf", "mean"), seen=("seen", "mean"),
                                                   refw=("ref_words", "mean"), s2=("sgrp", lambda v: (v == "2").mean())).round(2).to_dict("index"))
    r = {}
    for s in ("test", "val"):
        tt = tabs[s]; y = tt.chrf.to_numpy(float); yr = tt.recall.to_numpy(float)
        print(f"  --- {s}")
        r[s] = {"primary": fit_report(tt, y, "chrF primary"),
                "alt_len_13a": fit_report(tt, y, "chrF primary, 13a ref-token length", lencol="ref_tok13a"),
                "signer": fit_report(tt, y, "chrF + signer", signer=True),
                "videoFE": fit_report(tt, y, "chrF + video FE", fe=True),
                "heavy_bin": fit_report(tt, y, "chrF bins (heavy vs none)", coef="heavy", bins=True),
                "recall": fit_report(tt, yr, "content recall primary")}
        # crowding control, own construction
        t2 = tt.assign(ref_words2=tt.ref_words)
        X, nm = design(t2, signer=True); X = np.c_[X, t2.byst, t2.frac_multi]
        b, se, dfr = cr1(y, X, t2.video.to_numpy()); j = nm.index("frac")
        r[s]["signer_byst_multi_cr1"] = [float(b[j]), [float(b[j] - tq(dfr) * se[j]), float(b[j] + tq(dfr) * se[j])]]
        print(f"    {'chrF + signer + byst + frac_multi (CR1)':48s} frac {b[j]:+7.2f} [{r[s]['signer_byst_multi_cr1'][1][0]:+.2f},"
              f"{r[s]['signer_byst_multi_cr1'][1][1]:+.2f}]")
        obs, p = perm_within_video(tt, y)
        r[s]["perm_within_video"] = [obs, p]
        print(f"    within-video permutation of exposure: slope {obs:+.2f}, two-sided p {p:.4f}")
    # leave-one-video-out on the test slope
    tt = tabs["test"]; y = tt.chrf.to_numpy(float); X, nm = design(tt); j = nm.index("frac"); g = tt.video.to_numpy()
    lovo = {}
    for k in np.unique(g):
        m = g != k; lovo[k] = np.linalg.lstsq(X[m], y[m], rcond=None)[0][j]
    lv = pd.Series(lovo).sort_values()
    r["test_lovo"] = {"min": [lv.index[0], float(lv.iloc[0])], "max": [lv.index[-1], float(lv.iloc[-1])]}
    heavy_by_video = tt[tt.bin == "heavy"].video.value_counts()
    r["test_heavy_videos"] = {"n_videos": int(len(heavy_by_video)), "max_per_video": int(heavy_by_video.max())}
    print(f"  test LOVO slope range {lv.iloc[0]:+.2f} (drop {lv.index[0]}) .. {lv.iloc[-1]:+.2f} (drop {lv.index[-1]}); "
          f"64 heavy test clips spread over {len(heavy_by_video)} videos, max {heavy_by_video.max()} per video")
    # pooled + interaction, and the within-video split-swap test
    tp = pd.concat([tabs["test"], tabs["val"]], ignore_index=True)
    print("  --- pooled test+val")
    r["pooled"] = {"primary": fit_report(tp, tp.chrf.to_numpy(float), "pooled chrF primary"),
                   "signer": fit_report(tp, tp.chrf.to_numpy(float), "pooled chrF + signer", signer=True),
                   "interaction": fit_report(tp, tp.chrf.to_numpy(float), "pooled + signer + frac x test", coef="frac_x_test",
                                             signer=True, split=True)}
    obs, p = split_swap_test(tp)
    r["pooled"]["split_swap_perm"] = [obs, p]
    print(f"    within-video test<->val label-swap test of frac x test: {obs:+.2f}, two-sided p {p:.4f}")
    OUT["quality"] = r
    return tabs


# ---------------------------------------------------------------------------------------------------- Q3 at stake
def stake(tabs, man):
    print("\n[3] AT STAKE")
    r = {}
    for split in ("test", "val"):
        t = tabs[split]; S = np.stack(t.stats).astype(float); base = bleu(S); g = t.video.to_numpy()
        comp = {}
        for b in ("none", "light", "medium", "heavy"):
            m = (t.bin == b).to_numpy()
            comp[b] = {"clips": int(m.sum()), "ref_tok_share": float(S[m, 1].sum() / S[:, 1].sum()), "bleu": bleu(S[m])}
        sw_tok = 1 - comp["none"]["ref_tok_share"]
        lb = np.minimum(t.ref_words.to_numpy(), 8)

        def expected_sub(idx, target, cellcols):
            """expected corpus BLEU if each target clip took the MEAN stats of unswitched clips in its matching cell."""
            St = S[idx].copy(); bn = t.bin.to_numpy()[idx]
            cell = np.array(["|".join(map(str, z)) for z in zip(*[c[idx] for c in cellcols])])
            don = defaultdict(list)
            for i2, (bb, cc) in enumerate(zip(bn, cell)):
                if bb == "none":
                    don[cc].append(i2)
            mean = {cc: St[v].mean(0) for cc, v in don.items()}
            orig = St.copy(); nomatch = 0
            for i2, (bb, cc) in enumerate(zip(bn, cell)):
                if bb in target:
                    if cc in mean:
                        St[i2] = mean[cc]
                    else:
                        nomatch += 1
            return bleu(St) - bleu(orig), nomatch
        seen = t.seen.to_numpy().astype(int); sgrp = t.sgrp.to_numpy()
        cells = {"seen x len": [seen, lb], "seen x len x signer": [seen, lb, sgrp]}
        res = {"base": base, "composition": comp, "switched_ref_tok_share": sw_tok}
        for tgt_name, tgt in (("all_switched", ("light", "medium", "heavy")), ("heavy", ("heavy",))):
            for cname, cc in cells.items():
                pt, nm = expected_sub(np.arange(len(t)), tgt, cc)
                bs = [expected_sub(i, tgt, cc)[0] for i, _ in boot_clusters(g, 1000)]
                res[f"{tgt_name} | {cname}"] = [float(pt), q(bs), nm]
                print(f"  {split} matched substitution {tgt_name:13s} cell={cname:20s} {pt:+.3f} [{q(bs)[0]:+.3f},{q(bs)[1]:+.3f}]"
                      f" (unmatched targets {nm})")
        print(f"  {split} base {base:.3f}; switched clips' ref-token share {sw_tok:.3f}; " +
              "; ".join(f"{b} n{v['clips']} tok {v['ref_tok_share']:.3f} BLEU {v['bleu']:.2f}" for b, v in comp.items()))
        r[split] = res
    # wave-3 realised, exact expectation for the placebo
    for split in ("test", "val"):
        t = tabs[split]; refs = [x for _, x in man[split]]
        sw = (t.bin != "none").to_numpy(); d_tot, d_sw, d_pl = [], [], []
        for s in (42, 101, 202, 303, 404):
            dd = {}
            for arm in ("w3C", "w3T"):
                d = json.load(open(W3 / (f"{arm}_s{s}.json" if split == "val" else f"{arm}_s{s}_test.json")))
                assert d["refs"] == refs, (arm, s, split)
                st = np.array([ngram_stats(detokenize(h), detokenize(x)) for h, x in zip(d["beam_hyps"], refs)], float)
                rec = d["beam_result"]["metrics"]["tokenize=13a;detok=True"]
                assert abs(bleu(st) - rec) < 1e-6, (arm, s, split, bleu(st), rec)
                dd[arm] = st
            C, T = dd["w3C"], dd["w3T"]; bC = bleu(C)
            d_tot.append(bleu(T) - bC)
            m = C.copy(); m[sw] = T[sw]; d_sw.append(bleu(m) - bC)
            frac_pl = sw.sum() / (~sw).sum()                   # expected counts under a random equal-size unswitched subset
            e = C.sum(0) + frac_pl * (T[~sw] - C[~sw]).sum(0); d_pl.append(bleu(e) - bC)
        ex = np.array(d_sw) - np.array(d_pl); tc = tq(4)
        ci = lambda v: [float(np.mean(v) - tc * np.std(v, ddof=1) / 5 ** .5), float(np.mean(v) + tc * np.std(v, ddof=1) / 5 ** .5)]
        r[f"w3_{split}"] = {"total_T_minus_C": [float(np.mean(d_tot)), ci(d_tot)], "switched_swap": [float(np.mean(d_sw)), ci(d_sw)],
                            "placebo_expected": [float(np.mean(d_pl)), ci(d_pl)], "excess": [float(ex.mean()), ci(ex), int((ex > 0).sum())],
                            "per_seed_excess": [round(float(v), 3) for v in ex]}
        print(f"  w3 {split}: total T-C {np.mean(d_tot):+.3f} {ci(d_tot)}; swap on switched {np.mean(d_sw):+.3f}; "
              f"expected placebo {np.mean(d_pl):+.3f}; EXCESS {ex.mean():+.3f} [{ci(ex)[0]:+.3f},{ci(ex)[1]:+.3f}] ({(ex > 0).sum()}/5)")
    OUT["stake"] = r


def w3_per_clip(tabs, man):
    """Replication with the 5 wave-3 index-0 single models, and the within-clip tracked-minus-index-0 contrast."""
    print("\n[2b] WAVE-3 per-clip replication (seed-mean sentence chrF) + max_ndet>=2 restriction for S6")
    r = {}
    for split in ("test", "val"):
        t = tabs[split].copy(); refs = [x for _, x in man[split]]
        for arm in ("w3C", "w3T"):
            m = []
            for s in (42, 101, 202, 303, 404):
                d = json.load(open(W3 / (f"{arm}_s{s}.json" if split == "val" else f"{arm}_s{s}_test.json")))
                m.append([sacrebleu.sentence_chrf(detokenize(h), [detokenize(x)]).score for h, x in zip(d["beam_hyps"], refs)])
            t[arm] = np.mean(m, 0)
        r[split] = {"w3C_signer": fit_report(t, t.w3C.to_numpy(float), f"{split} w3C chrF + signer", signer=True),
                    "TminusC_slope": fit_report(t, (t.w3T - t.w3C).to_numpy(float), f"{split} w3 T-C slope"),
                    "TminusC_heavy": fit_report(t, (t.w3T - t.w3C).to_numpy(float), f"{split} w3 T-C heavy vs none",
                                                coef="heavy", bins=True),
                    "S6_maxndet2": fit_report(t[t.max_ndet >= 2].reset_index(drop=True),
                                              t.loc[t.max_ndet >= 2, "chrf"].to_numpy(float), f"{split} S6 chrF primary, max_ndet>=2")}
    OUT["w3_per_clip"] = r


def main():
    selfcheck()
    df, man = load()
    proxy(df)
    prevalence(df)
    tabs = quality(df, man)
    w3_per_clip(tabs, man)
    stake(tabs, man)
    json.dump(OUT, open(Path(__file__).with_name("verify_switch_lens_results.json"), "w"), indent=1, default=str)
    print("\nwrote verify_switch_lens_results.json")


if __name__ == "__main__":
    main()
