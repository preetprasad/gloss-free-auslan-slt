"""Can a pose-only detector tell a SIGNER from a NON-SIGNER?  (signing-activity lens, 2026-09-25)

Data: tracked DWPose of the 1,600 official val+test signer clips (results/nonsigner/tracked) and of the 1,678 dev+test
Non_Signer crops (results/nonsigner/npz).  Interpretable per-clip features -> logistic regression, 5-fold CV GROUPED by
Video_Name (scenes never cross folds).  Reuses the non-signer gate's own B3 filter, AUROC and video-cluster bootstrap.

    python signing_activity.py selftest
    python signing_activity.py run        # writes results.json next to this file
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src/tools"))
import nonsigner_gate as g  # noqa: E402  (skel_dist, cluster_boot, B=2000, CONF=0.3)

NS, RES = ROOT / "results/nonsigner", ROOT / "results"
XLSX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Signer Detection/Signer_Detection.xlsx"
CONF, FPS, RAISED = g.CONF, 25.0, 0.8   # RAISED: wrist above (shoulder line + 0.8 shoulder widths), fixed a priori
LH, RH = np.arange(92, 112), np.arange(113, 133)   # finger keypoints; 91 / 112 are the hand roots

KIN = ["wrist_speed_med", "wrist_speed_p90", "finger_motion_med", "hand_height_med", "raised_share",
       "hand_face_dist_med", "body_motion_med", "head_motion_med"]
FULL = KIN + ["hand_conf", "wrist_visible_share", "valid_share", "log_T"]      # the task's feature list
EXTRA = ["shoulder_px"]                                                         # framing confound, never in a model above


# ---------------------------------------------------------------- features
def features(k):
    """k (T,133,3) raw pixels + conf.  Speeds in shoulder widths per second; heights/distances in shoulder widths."""
    T = len(k)
    w = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)
    ok = (k[:, 5, 2] >= CONF) & (k[:, 6, 2] >= CONF) & (w > 1e-3)
    mid = (k[:, 5, :2] + k[:, 6, :2]) / 2
    ws = np.where(ok, w, np.nan)
    z = (k[:, :, :2] - mid[:, None]) / ws[:, None, None]            # shoulder-centred, shoulder-width units
    c = k[:, :, 2]
    wr = [(9, c[:, 9] >= CONF), (10, c[:, 10] >= CONF)]
    nanmed = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
    nanp90 = lambda a: float(np.nanpercentile(a, 90)) if np.isfinite(a).any() else np.nan

    pair = ok[1:] & ok[:-1]
    def speed(j, m):                                                 # per-frame speed of joint j (both frames confident)
        d = np.linalg.norm(z[1:, j] - z[:-1, j], axis=1) * FPS
        return np.where(pair & m[1:] & m[:-1], d, np.nan)
    wsp = np.fmax(speed(9, wr[0][1]), speed(10, wr[1][1]))           # fmax ignores a NaN hand
    fm = []
    for root, fing in ((91, LH), (112, RH)):
        hc = c[:, fing].mean(1) >= CONF
        rel = z[:, fing] - z[:, [root]]
        d = np.linalg.norm(rel[1:] - rel[:-1], axis=2).mean(1) * FPS
        fm.append(np.where(pair & hc[1:] & hc[:-1], d, np.nan))
    fmo = np.fmax(*fm)
    height = np.fmax(*[np.where(ok & m, -z[:, j, 1], np.nan) for j, m in wr])         # + = above shoulder line
    face = np.fmin(*[np.where(ok & m & (c[:, 0] >= CONF), np.linalg.norm(z[:, j] - z[:, 0], axis=1), np.nan)
                     for j, m in wr])
    raised = np.where(ok, np.nan_to_num(height, nan=-9) > -RAISED, False)
    body = np.where(pair, np.linalg.norm(mid[1:] - mid[:-1], axis=1) / ws[1:] * FPS, np.nan)
    head = speed(0, c[:, 0] >= CONF)
    vis = (c[:, 9] >= CONF) | (c[:, 10] >= CONF)
    return {"wrist_speed_med": nanmed(wsp), "wrist_speed_p90": nanp90(wsp), "finger_motion_med": nanmed(fmo),
            "hand_height_med": nanmed(height), "raised_share": float(raised[ok].mean()) if ok.any() else np.nan,
            "hand_face_dist_med": nanmed(face), "body_motion_med": nanmed(body), "head_motion_med": nanmed(head),
            "hand_conf": float(c[:, 91:133].mean()), "wrist_visible_share": float(vis[ok].mean()) if ok.any() else np.nan,
            "valid_share": float(ok.mean()) if T else 0.0, "log_T": float(np.log(max(T, 1))),
            "shoulder_px": float(np.median(w[ok])) if ok.any() else np.nan}


# ---------------------------------------------------------------- statistics
def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 ties (higher score = signer). g.auroc ranks -x, so pass -score."""
    return g.auroc(-np.asarray(pos, float), -np.asarray(neg, float))


def boot_auc(score, y, vid):
    """Video-cluster bootstrap (the gate's own resampler, B=2000, seed 0)."""
    by = {}
    for s, t, v in zip(score, y, vid):
        by.setdefault(v, ([], []))[1 - t].append(s)                 # [0] signers, [1] non-signers
    def stat(vs):
        p = [x for v in vs for x in by[v][0]]; n = [x for v in vs for x in by[v][1]]
        return auc(p, n) if p and n else None
    return g.cluster_boot(list(by), stat)


def boot_mean(x, vid):
    by = {}
    for a, v in zip(x, vid):
        by.setdefault(v, []).append(a)
    return g.cluster_boot(list(by), lambda vs: float(np.mean([a for v in vs for a in by[v]])))


def model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=5000))


def folds(y, grp, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, grp))


def oof(X, y, grp, seed=0):
    return cross_val_predict(model(), X, y, groups=grp, cv=folds(y, grp, seed), method="decision_function")


def nested_accept(X, y, grp, seed=0, recall=0.95, fixed_score=None):
    """Honest operating point: in each outer fold the threshold is the (1-recall) quantile of INNER out-of-fold signer
    scores on the training videos only; the model is then refit on all training videos. fixed_score = a score that needs
    no fitting (e.g. -ce): the threshold is then the same quantile of the training-fold signers' scores."""
    acc = np.zeros(len(y), bool)
    for tr, te in folds(y, grp, seed):
        if fixed_score is not None:
            s_tr, s_te = fixed_score[tr], fixed_score[te]
        else:
            s_tr = cross_val_predict(model(), X[tr], y[tr], groups=grp[tr], cv=folds(y[tr], grp[tr], seed + 1),
                                     method="decision_function")
            s_te = model().fit(X[tr], y[tr]).decision_function(X[te])
        acc[te] = s_te >= np.quantile(s_tr[y[tr] == 1], 1 - recall)
    return acc


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ctr - h, ctr + h


# ---------------------------------------------------------------- data
def load():
    sd = pd.read_excel(XLSX)
    man = {s: [Path(l.split("\t")[0]).stem for l in open(RES / f) if "\t" in l]
           for s, f in (("test", "TEST_manifest_official.tsv"), ("dev", "_local_val_manifest.tsv"))}
    assert len(man["test"]) == len(man["dev"]) == 800 and not set(man["test"]) & set(man["dev"])
    vid = dict(zip(sd.Video_Clip_Name, sd.Video_Name))
    neg = sd[(sd.Label == "Non_Signer") & sd.Split.isin(["dev", "test"])].copy()
    neg["cid"] = neg.Person_Video_Clip.str[:-4]
    have = {p.stem for p in (NS / "npz").glob("*.npz")}
    neg = neg[neg.cid.isin(have)]
    assert len(have) == 1678 and set(neg.cid) == have, "npz set != Signer_Detection Non_Signer dev/test rows"
    assert (neg.Split == "test").sum() == 847 and (neg.Split == "dev").sum() == 831
    assert set(neg[neg.Split == "test"].Video_Clip_Name) <= set(man["test"])
    assert set(neg[neg.Split == "dev"].Video_Clip_Name) <= set(man["dev"])
    rows = [dict(cid=c, parent=c, y=1, split=s, video=vid[c], path=NS / "tracked" / f"{c}.npz")
            for s in ("test", "dev") for c in man[s]]
    rows += [dict(cid=r.cid, parent=r.Video_Clip_Name, y=0, split=r.Split, video=r.Video_Name,
                  path=NS / "npz" / f"{r.cid}.npz") for r in neg.itertuples()]
    df = pd.DataFrame(rows)
    assert df.path.map(Path.exists).all() and len(df) == 3278
    return df


def b3_flags(df):
    """Reproduce the gate's B3 same-person flags with its recorded threshold (a reproduction gate, not a new test)."""
    rep = json.load(open(NS / "nonsigner_gate_results.json"))["B3"]
    thr, flag, why = rep["null_p1_threshold"], {}, {}
    for r in df[df.y == 0].itertuples():
        x, s = np.load(r.path)["KPS"], np.load(NS / "tracked" / f"{r.parent}.npz")["KPS"]
        if abs(len(x) - len(s)) > 2:
            why[r.cid] = "length_mismatch"; continue
        n = min(len(x), len(s)); d = g.skel_dist(x[:n], s[:n])
        if d is None:
            why[r.cid] = "no_confident_frames"; continue
        flag[r.cid] = d < thr
    for s in ("test", "dev"):
        ids = df[(df.y == 0) & (df.split == s)].cid
        got = (sum(flag.get(c, False) for c in ids), sum(c in why for c in ids))
        want = (rep["per_split"][s]["flagged_same_person"], sum(rep["per_split"][s]["untestable"].values()))
        assert got == want, f"B3 reproduction failed on {s}: {got} vs {want}"
    return {c for c, f in flag.items() if f}


# ---------------------------------------------------------------- run
def run():
    df = load()
    F = pd.DataFrame([features(np.load(p)["KPS"]) for p in df.path])
    df = pd.concat([df.reset_index(drop=True), F], axis=1)
    flagged = b3_flags(df)
    D = json.load(open(NS / "decodes.json"))
    df["ce"] = [D["neg"][c]["ce"] if y == 0 else D["trk_test" if s == "test" else "trk_val"][c]["ce"]
                for c, y, s in zip(df.cid, df.y, df.split)]
    # reproduction gate: the recorded tracked-arm confidence AUROC (test signers vs primary test non-signers)
    rec = json.load(open(NS / "nonsigner_gate_results.json"))["ii_tracked"]["auroc"]
    t = df[(df.split == "test") & ~df.cid.isin(flagged)]
    a = boot_auc(-t.ce.to_numpy(), t.y.to_numpy(), t.video.to_numpy())
    assert np.allclose(a, rec, atol=1e-9), f"confidence AUROC does not reproduce: {a} vs {rec}"
    print("gates passed: B3 flags 16/26 reproduced; confidence AUROC", [round(x, 4) for x in a])

    P = df[~df.cid.isin(flagged)].reset_index(drop=True)             # primary population (gate B15, dev + test)
    y, grp = P.y.to_numpy(), P.video.to_numpy()
    out = {"n": {"signers": int(y.sum()), "non_signers": int((1 - y).sum()), "videos": int(len(set(grp))),
                 "excluded_b3_same_person": len(flagged)}}
    cols = {"full": FULL, "kinematic": KIN, "hand_height_only": ["hand_height_med"], "length_only": ["log_T"],
            "framing_only": ["shoulder_px"]}
    S = {}
    for name, cs in cols.items():
        X = P[cs].to_numpy(float)
        S[name] = oof(X, y, grp, 0)
        acc = nested_accept(X, y, grp, 0)
        out[name] = {"features": cs, "auroc": boot_auc(S[name], y, grp),
                     "auroc_cv_seeds_0_9": [round(auc(o[y == 1], o[y == 0]), 4)
                                            for o in (oof(X, y, grp, sd) for sd in range(10))],
                     "op95_signer_recall": boot_mean(acc[y == 1], grp[y == 1]),
                     "op95_nonsigner_rejected": boot_mean(~acc[y == 0], grp[y == 0])}
        print(name, out[name]["auroc"], "reject@95", out[name]["op95_nonsigner_rejected"])
    for sp in ("test", "dev"):                                       # per split, from the same out-of-fold scores
        m_ = (P.split == sp).to_numpy()
        out[f"auroc_{sp}_only"] = {"full": boot_auc(S["full"][m_], y[m_], grp[m_]),
                                   "confidence_gate": boot_auc(-P.ce.to_numpy()[m_], y[m_], grp[m_])}
    # translator-confidence gate on the SAME rows, SAME folds, SAME threshold rule (no model to fit)
    sce = -P.ce.to_numpy()
    acc = nested_accept(None, y, grp, 0, fixed_score=sce)
    out["confidence_gate"] = {"auroc": boot_auc(sce, y, grp), "op95_signer_recall": boot_mean(acc[y == 1], grp[y == 1]),
                              "op95_nonsigner_rejected": boot_mean(~acc[y == 0], grp[y == 0])}
    # shuffled-label control (global permutation, same CV)
    rng, X = np.random.default_rng(0), P[FULL].to_numpy(float)
    null = [auc(o[yy == 1], o[yy == 0]) for yy in (rng.permutation(y) for _ in range(50))
            for o in [oof(X, yy, grp, 0)]]
    out["shuffled_label_full"] = {"n_perm": 50, "mean": float(np.mean(null)), "p95": float(np.percentile(null, 95)),
                                  "max": float(np.max(null)),
                                  "p_ge_observed": float(np.mean(np.array(null) >= out["full"]["auroc"][0]))}
    # univariate AUROC (no fitting; > 0.5 = higher in signers)
    out["univariate"] = {}
    for c in FULL + EXTRA:
        m = P[c].notna().to_numpy()
        out["univariate"][c] = {"auroc": boot_auc(P[c].to_numpy()[m], y[m], grp[m]),
                                "median_signer": float(P[c][m & (y == 1)].median()),
                                "median_nonsigner": float(P[c][m & (y == 0)].median()), "n_missing": int((~m).sum())}
    # scene-controlled: each non-signer against its own signer (same clip, same scene, same fold)
    idx = dict(zip(P.cid, range(len(P))))
    for name in ("full", "kinematic"):
        pr = [(S[name][idx[r.parent]], S[name][i], r.video) for i, r in P[P.y == 0].iterrows() if r.parent in idx]
        win = [1.0 if a_ > b_ else 0.5 if a_ == b_ else 0.0 for a_, b_, _ in pr]
        out[name]["paired_same_clip_signer_higher"] = boot_mean(win, [v for *_, v in pr]) + (len(pr),)
    # standardised coefficients of the full model fitted on everything (interpretation only)
    m = model().fit(P[FULL].to_numpy(float), y)
    coef = m[-1].coef_[0]
    out["full"]["coef_std"] = {c: round(float(v), 3) for c, v in zip(FULL + [f"missing_{i}" for i in range(len(coef) - len(FULL))], coef)}
    # label noise: blind second-rater labels of the 60 B4-audited test crops (b4_labels_rater2.tsv)
    lab = pd.read_csv(HERE / "b4_labels_rater2.tsv", sep="\t", comment="#")
    acc_full = nested_accept(P[FULL].to_numpy(float), y, grp, 0)
    lab = lab[lab.crop.isin(idx)]
    out["b4_rater2"] = {}
    for L, gr in lab.groupby("label"):
        ii = [idx[c] for c in gr.crop]
        out["b4_rater2"][L] = {"n": len(ii), "median_oof_score": float(np.median(S["full"][ii])),
                               "share_accepted_as_signer_op95": float(acc_full[ii].mean())}
    sI, qI = [idx[c] for c in lab[lab.label == "S"].crop], [idx[c] for c in lab[lab.label == "Q"].crop]
    out["b4_rater2"]["auroc_S_vs_Q"] = auc(S["full"][sI], S["full"][qI])
    k, n = int((lab.label == "S").sum()), len(lab)
    lo, hi = wilson(k, n)
    out["label_noise_ceiling"] = {"signing_share": [k / n, lo, hi],
                                  "auroc_ceiling_if_signing_negatives_look_like_signers": [1 - k / n / 2, 1 - hi / 2, 1 - lo / 2],
                                  "orig_rater_23_of_60": [1 - 23 / 60 / 2] + [1 - x / 2 for x in wilson(23, 60)[::-1]]}
    # latency: features from the FIRST W frames of each clip only (a live detector must decide from a short window)
    out["first_w_frames"] = {}
    for W in (13, 25, 50):
        Fw = pd.DataFrame([features(np.load(p)["KPS"][:W]) for p in P.path])
        out["first_w_frames"][W] = {name: boot_auc(oof(Fw[cs].to_numpy(float), y, grp, 0), y, grp)
                                    for name, cs in (("full", FULL), ("kinematic", KIN))}
        print("first", W, "frames", out["first_w_frames"][W])
    P[["cid", "y", "split", "video", "parent", "ce"] + FULL + EXTRA].assign(oof_full=S["full"], oof_kin=S["kinematic"]) \
        .to_csv(HERE / "features_oof.csv", index=False)
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "univariate"}, indent=1, default=float)[:6000])


# ---------------------------------------------------------------- self-check
def selftest():
    T = 50
    k = np.zeros((T, 133, 3), np.float32); k[..., 2] = 0.9
    k[:, 5, :2], k[:, 6, :2] = (100, 100), (0, 100)                 # shoulder width 100 px, shoulders at y = 100
    k[:, 0, :2] = (50, 40)                                          # nose
    k[:, 9, :2], k[:, 10, :2] = (90, 280), (10, 280)                # wrists 1.8 widths below the shoulder line
    k[:, 91:112, :2] = k[:, [9], :2]; k[:, 112:133, :2] = k[:, [10], :2]
    f = features(k)
    assert f["wrist_speed_med"] == 0 and f["raised_share"] == 0 and abs(f["hand_height_med"] + 1.8) < 1e-6
    assert f["finger_motion_med"] == 0 and f["body_motion_med"] == 0 and f["valid_share"] == 1.0
    s = k.copy(); s[:, 10, 1] = 100 + 20 * np.sin(np.arange(T))     # right wrist at the shoulder line, oscillating
    fs = features(s)
    assert fs["wrist_speed_med"] > 1 and fs["raised_share"] == 1.0 and fs["hand_height_med"] > -0.25
    s2 = k.copy(); s2[1::2, 5:7, 2] = 0                             # every other frame loses the shoulders
    assert features(s2)["valid_share"] == 0.5 and np.isnan(features(s2)["wrist_speed_med"])
    assert auc([2, 3], [0, 1]) == 1.0 and auc([0], [1]) == 0.0 and auc([1], [1]) == 0.5
    rng = np.random.default_rng(0)                                  # nested threshold realises ~95% recall
    yy = np.r_[np.ones(400, int), np.zeros(400, int)]
    X = (rng.normal(size=(800, 2)) + yy[:, None]).astype(float)
    grp = np.repeat(np.arange(80), 10); rng.shuffle(grp)
    acc = nested_accept(X, yy, grp, 0)
    assert 0.9 < acc[yy == 1].mean() < 0.99, acc[yy == 1].mean()
    assert all(not set(grp[tr]) & set(grp[te]) for tr, te in folds(yy, grp, 0)), "video leaked across folds"
    print("selftest passed")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
