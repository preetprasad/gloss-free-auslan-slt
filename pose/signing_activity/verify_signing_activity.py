"""ADVERSARIAL VERIFIER of the signing-activity lens (2026-09-26). Shares NO code with signing_activity*.py.

Independent from the lens in: feature code (per-clip median scale, not per-frame), fold assignment (own shuffled
video partition), imputation (no missing indicators), bootstrap (own video-cluster resampler), label source (video
id parsed from the clip name, cross-checked against Signer_Detection.xlsx). Adds checks the lens did not run:
  * orientation confound: shoulder width shrinks when a person turns side-on, which inflates every shoulder-width
    normalised distance. Re-run with a yaw-invariant VERTICAL scale (nose-to-shoulder-line height) and stratify by
    a yaw proxy (shoulder width / neck height).
  * per-category AUROC of signers vs each second-rater class of the audited non-signers (S / Q / M).

    python verify_signing_activity.py selftest
    python verify_signing_activity.py run      # writes verify.json
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NS = ROOT / "results/nonsigner"
XLSX = ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Signer Detection/Signer_Detection.xlsx"
COMM = ROOT / ("data/raw/Auslan-Daily/Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/"
               "Auslan-Daily_Communication.xlsx")
C, FPS = 0.3, 25.0
LSH, RSH, NOSE, LWR, RWR = 5, 6, 0, 9, 10
LROOT, RROOT = 91, 112
FEATS = ["wspd_med", "wspd_p90", "fing_med", "height_med", "raised", "face_med", "body_med", "head_med",
         "hconf", "wvis", "valid", "logT"]


# ------------------------------------------------------------------ features (own implementation)
def feats(k, scale_kind="sw"):
    k = np.asarray(k, float)
    T = len(k)
    x, y, c = k[..., 0], k[..., 1], k[..., 2]
    sh_ok = (c[:, LSH] >= C) & (c[:, RSH] >= C)
    sw = np.hypot(x[:, LSH] - x[:, RSH], y[:, LSH] - y[:, RSH])
    sh_ok &= sw > 1
    mx, my = (x[:, LSH] + x[:, RSH]) / 2, (y[:, LSH] + y[:, RSH]) / 2
    neck = my - y[:, NOSE]                                   # vertical, yaw-invariant; > 0 when nose above shoulders
    neck_ok = sh_ok & (c[:, NOSE] >= C) & (neck > 1)
    out = {"sw_px": np.median(sw[sh_ok]) if sh_ok.any() else np.nan,
           "neck_px": np.median(neck[neck_ok]) if neck_ok.any() else np.nan}
    out["yaw_ratio"] = out["sw_px"] / out["neck_px"] if neck_ok.any() and sh_ok.any() else np.nan
    out["nose_off"] = (np.median(np.abs(x[sh_ok & (c[:, NOSE] >= C), NOSE] - mx[sh_ok & (c[:, NOSE] >= C)])
                                 / sw[sh_ok & (c[:, NOSE] >= C)])
                       if (sh_ok & (c[:, NOSE] >= C)).any() else np.nan)
    S = out["sw_px"] if scale_kind == "sw" else out["neck_px"]
    if not np.isfinite(S):
        return {**out, **{f: np.nan for f in FEATS}, "valid": float(sh_ok.mean()) if T else 0.0,
                "logT": np.log(max(T, 1)), "hconf": float(c[:, 91:133].mean())}
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
    wok = [(c[:, j] >= C) & sh_ok for j in (LWR, RWR)]
    hts = [np.where(m, (my - y[:, j]) / S, np.nan) for j, m in zip((LWR, RWR), wok)]
    height = np.fmax(hts[0], hts[1])
    fd = [np.where(m & (c[:, NOSE] >= C), np.hypot(x[:, j] - x[:, NOSE], y[:, j] - y[:, NOSE]) / S, np.nan)
          for j, m in zip((LWR, RWR), wok)]
    face = np.fmin(fd[0], fd[1])

    def spd(jx, jy, m):                                     # speed in scale units / s, both frames valid
        d = np.hypot(np.diff(jx), np.diff(jy)) / S * FPS
        return np.where(m[1:] & m[:-1], d, np.nan)
    ws = np.fmax(spd(x[:, LWR] - mx, y[:, LWR] - my, wok[0]), spd(x[:, RWR] - mx, y[:, RWR] - my, wok[1]))
    fm = []
    for r in (LROOT, RROOT):
        idx = np.arange(r + 1, r + 21)
        hm = c[:, idx].mean(1) >= C
        rx, ry = x[:, idx] - x[:, [r]], y[:, idx] - y[:, [r]]
        d = np.hypot(np.diff(rx, axis=0), np.diff(ry, axis=0)).mean(1) / S * FPS
        fm.append(np.where(hm[1:] & hm[:-1], d, np.nan))
    fing = np.fmax(*fm)
    body = spd(mx, my, sh_ok)
    head = spd(x[:, NOSE], y[:, NOSE], sh_ok & (c[:, NOSE] >= C))
    # fixed thresholds in each unit (neck mode stays yaw-invariant: -1.6 neck heights ~ -0.8 sw at a frontal ratio of 2)
    raised = np.nan_to_num(height, nan=-99)[sh_ok] > (-0.8 if scale_kind == "sw" else -1.6)
    out.update(wspd_med=med(ws), wspd_p90=float(np.nanpercentile(ws, 90)) if np.isfinite(ws).any() else np.nan,
               fing_med=med(fing), height_med=med(height), raised=float(raised.mean()) if sh_ok.any() else np.nan,
               face_med=med(face), body_med=med(body), head_med=med(head), hconf=float(c[:, 91:133].mean()),
               wvis=float(((c[:, LWR] >= C) | (c[:, RWR] >= C))[sh_ok].mean()) if sh_ok.any() else np.nan,
               valid=float(sh_ok.mean()), logT=float(np.log(max(T, 1))))
    return out


# ------------------------------------------------------------------ CV / stats (own implementation)
def vfolds(groups, seed, k=5):
    u = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    fid = {v: i % k for i, v in enumerate(u)}
    f = np.array([fid[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def mk():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=4000))


def oof(X, y, g, seed=0):
    s = np.full(len(y), np.nan)
    for tr, te in vfolds(g, seed):
        assert not set(g[tr]) & set(g[te])
        s[te] = mk().fit(X[tr], y[tr]).decision_function(X[te])
    return s


def nested_acc(X, y, g, seed=0, fixed=None, recall=0.95):
    acc = np.zeros(len(y), bool)
    for tr, te in vfolds(g, seed):
        if fixed is None:
            inner = oof(X[tr], y[tr], g[tr], seed + 100)
            thr = np.quantile(inner[y[tr] == 1], 1 - recall)
            s_te = mk().fit(X[tr], y[tr]).decision_function(X[te])
        else:
            thr = np.quantile(fixed[tr][y[tr] == 1], 1 - recall)
            s_te = fixed[te]
        acc[te] = s_te >= thr
    return acc


def cboot(g, stat, B=2000, seed=0):
    """video-cluster bootstrap: returns (point, lo, hi)."""
    u = np.array(sorted(set(g)))
    rows = {v: np.where(g == v)[0] for v in u}
    rng = np.random.default_rng(seed)
    pt = stat(np.arange(len(g)))
    vals = []
    for _ in range(B):
        ii = np.concatenate([rows[v] for v in rng.choice(u, len(u))])
        v = stat(ii)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(pt), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def auc_ci(s, y, g):
    def st(ii):
        yy = y[ii]
        return roc_auc_score(yy, s[ii]) if 0 < yy.sum() < len(yy) else None
    return cboot(g, st)


def mean_ci(x, g):
    x = np.asarray(x, float)
    return cboot(g, lambda ii: x[ii].mean())


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [k / n, m - h, m + h]


# ------------------------------------------------------------------ data
def load():
    sd = pd.read_excel(XLSX)
    man = {}
    for sp, f in (("test", "TEST_manifest_official.tsv"), ("dev", "_local_val_manifest.tsv")):
        man[sp] = [Path(l.split("\t")[0]).stem for l in open(ROOT / "results" / f) if l.strip()]
    assert len(man["test"]) == len(man["dev"]) == 800 and not set(man["test"]) & set(man["dev"])
    negfiles = sorted(p.stem for p in (NS / "npz").glob("*.npz"))
    assert len(negfiles) == 1678
    neg = sd[sd.Label == "Non_Signer"].assign(cid=lambda d: d.Person_Video_Clip.str.replace(".mp4", "", regex=False))
    neg = neg.set_index("cid").loc[negfiles]
    # label handling: the crop's own row says Non_Signer, the parent clip has a Signer row, splits agree with manifests
    sig_rows = sd[sd.Label == "Signer"].set_index("Video_Clip_Name")
    rows = []
    for sp in ("test", "dev"):
        for c_ in man[sp]:
            assert sig_rows.loc[c_, "Split"] == sp, c_
            rows.append(dict(cid=c_, parent=c_, y=1, split=sp, path=NS / "tracked" / f"{c_}.npz"))
    for cid, r in neg.iterrows():
        assert r.Label == "Non_Signer" and r.Split in ("dev", "test")
        assert r.Video_Clip_Name in set(man[r.Split]), cid
        assert cid.startswith(r.Video_Clip_Name + "_") and cid.endswith("_nonsigner")
        rows.append(dict(cid=cid, parent=r.Video_Clip_Name, y=0, split=r.Split, path=NS / "npz" / f"{cid}.npz"))
    df = pd.DataFrame(rows)
    df["video"] = [re.match(r"(video_\d+)_", p).group(1) for p in df.parent]
    # cross-check own video parse against the xlsx's Video_Name (lowercased)
    vn = dict(zip(sd.Video_Clip_Name, sd.Video_Name.str.lower()))
    assert all(vn[p] == v for p, v in zip(df.parent, df.video)), "video id mismatch vs xlsx"
    assert (df.y == 0).sum() == 1678 and (df.y == 1).sum() == 1600 and df.cid.is_unique
    assert ((df.y == 0) & (df.split == "test")).sum() == 847 and ((df.y == 0) & (df.split == "dev")).sum() == 831
    return df


# ------------------------------------------------------------------ run
def run():
    df = load()
    for kind in ("sw", "neck"):
        F = pd.DataFrame([feats(np.load(p)["KPS"], kind) for p in df.path])
        F.columns = [f"{kind}:{c}" if c in FEATS else c for c in F.columns]
        df = pd.concat([df, F.loc[:, ~F.columns.isin(df.columns)]], axis=1)
    D = json.load(open(NS / "decodes.json"))
    df["ce"] = [D["neg"][c]["ce"] if y == 0 else D["trk_test" if s == "test" else "trk_val"][c]["ce"]
                for c, y, s in zip(df.cid, df.y, df.split)]
    lens = pd.read_csv(HERE / "features_oof.csv")
    b3 = set(df.cid[df.y == 0]) - set(lens.cid)           # ids only; counts checked against the gate json
    gate = json.load(open(NS / "nonsigner_gate_results.json"))["B3"]["per_split"]
    assert sum(df.split[df.cid.isin(b3)] == "test") == gate["test"]["flagged_same_person"] == 16
    assert sum(df.split[df.cid.isin(b3)] == "dev") == gate["dev"]["flagged_same_person"] == 26
    out = {"n_all": {"signers": 1600, "nonsigners": 1678, "videos": int(df.video.nunique())}}

    def evaluate(P, cols, name, seeds=(0, 1, 2, 3, 4), op=True):
        X, y, g = P[cols].to_numpy(float), P.y.to_numpy(), P.video.to_numpy()
        s = oof(X, y, g, 0)
        r = {"auroc": auc_ci(s, y, g), "auroc_seeds": [round(roc_auc_score(y, oof(X, y, g, sd)), 4) for sd in seeds]}
        if op:
            a = nested_acc(X, y, g, 0)
            r["op95_recall"] = mean_ci(a[y == 1], g[y == 1])
            r["op95_reject"] = mean_ci(~a[y == 0], g[y == 0])
        out[name] = r
        print(name, json.dumps(r))
        return s

    P = df[~df.cid.isin(b3)].reset_index(drop=True)
    y, g = P.y.to_numpy(), P.video.to_numpy()
    SW = [f"sw:{c}" for c in FEATS]
    NK = [f"neck:{c}" for c in FEATS]
    s_full = evaluate(P, SW, "full_sw_1636")
    evaluate(P, NK, "full_neck_1636")
    evaluate(P, ["sw:logT"], "length_only", op=False)
    evaluate(P, ["sw_px"], "framing_only", op=False)
    evaluate(P, ["yaw_ratio", "nose_off"], "orientation_only", op=False)
    evaluate(P, ["sw:height_med"], "height_sw_only", op=False)
    evaluate(P, ["neck:height_med"], "height_neck_only", op=False)
    s_all = evaluate(df, SW, "full_sw_all1678")
    # missingness by class (imputation must not be the signal)
    out["missing_rate"] = {c: [float(P[c][y == 1].isna().mean()), float(P[c][y == 0].isna().mean())] for c in SW + NK
                           if P[c].isna().any()}
    # univariate
    out["univariate"] = {}
    for c in SW + NK + ["yaw_ratio", "nose_off", "sw_px", "neck_px"]:
        m = P[c].notna().to_numpy()
        out["univariate"][c] = [round(roc_auc_score(y[m], P[c].to_numpy()[m]), 4),
                                round(float(P[c][m & (y == 1)].median()), 3),
                                round(float(P[c][m & (y == 0)].median()), 3)]
    # confidence gate, same rows
    sce = -P.ce.to_numpy()
    a = nested_acc(None, y, g, 0, fixed=sce)
    out["conf_gate"] = {"auroc": auc_ci(sce, y, g), "op95_recall": mean_ci(a[y == 1], g[y == 1]),
                        "op95_reject": mean_ci(~a[y == 0], g[y == 0]),
                        "auroc_test_only": auc_ci(sce[P.split == "test"], y[P.split == "test"], g[P.split == "test"])}
    print("conf", out["conf_gate"])
    # shuffled labels
    X = P[SW].to_numpy(float)
    rng = np.random.default_rng(7)
    null = [roc_auc_score(yy, oof(X, yy, g, 0)) for yy in (rng.permutation(y) for _ in range(100))]
    out["shuffled"] = {"mean": float(np.mean(null)), "p95": float(np.percentile(null, 95)), "max": float(max(null))}
    # split-wise
    for sp in ("dev", "test"):
        m = (P.split == sp).to_numpy()
        out[f"auroc_{sp}"] = auc_ci(s_full[m], y[m], g[m])
    # orientation strata: AUROC of the full score within yaw_ratio quintiles (pooled edges)
    yr = P.yaw_ratio.to_numpy()
    ok = np.isfinite(yr)
    edges = np.quantile(yr[ok], [0, .2, .4, .6, .8, 1])
    out["yaw_strata"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (yr >= lo) & (yr <= hi)
        out["yaw_strata"].append({"range": [round(lo, 3), round(hi, 3)], "n_sig": int(y[m].sum()),
                                  "n_neg": int((1 - y[m]).sum()), "auroc": auc_ci(s_full[m], y[m], g[m])})
    # paired within clip: non-signer vs its own signer (same scene, same fold)
    sc = dict(zip(P.cid, s_full))
    win = [(float(sc[r.parent] > s_), r.video) for r, s_ in zip(P[P.y == 0].itertuples(), s_full[y == 0])
           if r.parent in sc]
    out["paired_signer_higher"] = mean_ci([w for w, _ in win], np.array([v for _, v in win])) + [len(win)]
    # Signer_ID strata
    sid = pd.read_excel(COMM).set_index("Video_Clip_Name").Signer_ID
    P["sid"] = P.cid.map(sid)
    out["by_signer_id"] = {}
    for nm, m in (("id1", P.sid == 1), ("id2", P.sid == 2), ("other", (P.y == 1) & ~P.sid.isin([1, 2]))):
        mm = (m | (P.y == 0)).to_numpy()
        out["by_signer_id"][nm] = {"n": int(m.sum()), "auroc": auc_ci(s_full[mm], y[mm], g[mm])}
    # top-1 among people of a clip
    hits, ch, vv = [], [], []
    for par, gg in P.assign(s=s_full).groupby("parent"):
        if gg.y.sum() == 1 and len(gg) > 1:
            hits.append(float(gg.s[gg.y == 1].iloc[0] > gg.s[gg.y == 0].max())); ch.append(1 / len(gg))
            vv.append(gg.video.iloc[0])
    out["top1"] = {"n": len(hits), "hit": mean_ci(hits, np.array(vv)), "chance": float(np.mean(ch))}
    # second-rater categories: signers vs each class of audited negatives; acceptance at the op95 threshold
    lab = pd.read_csv(HERE / "b4_labels_rater2.tsv", sep="\t", comment="#").set_index("crop").label
    acc = nested_acc(X, y, g, 0)
    out["rater2"] = {}
    idx = dict(zip(P.cid, range(len(P))))
    for L in ("S", "Q", "M", "A"):
        ii = [idx[c] for c in lab.index[lab == L] if c in idx]
        k = int(acc[ii].sum())
        out["rater2"][L] = {"n": len(ii), "accepted": k, "accept_wilson": wilson(k, len(ii)),
                            "auroc_signers_vs_class": round(roc_auc_score(
                                np.r_[np.ones(int(y.sum())), np.zeros(len(ii))],
                                np.r_[s_full[y == 1], s_full[ii]]), 4)}
    # first 25 frames only
    F25 = pd.DataFrame([feats(np.load(p)["KPS"][:25], "sw") for p in P.path])
    s25 = oof(F25[FEATS].to_numpy(float), y, g, 0)
    out["first25"] = auc_ci(s25, y, g)
    json.dump(out, open(HERE / "verify.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("univariate",)}, indent=1, default=float))


def orient():
    """Is part of the separability head/body TURN (listeners face the signer) rather than signing activity?"""
    df = load()
    lens = pd.read_csv(HERE / "features_oof.csv")
    P = df[df.cid.isin(set(lens.cid) | set(df.cid[df.y == 1]))].reset_index(drop=True)
    assert len(P) == 3236
    F = pd.DataFrame([feats(np.load(p)["KPS"], "neck") for p in P.path])
    Fs = pd.DataFrame([feats(np.load(p)["KPS"], "sw") for p in P.path])
    y, g = P.y.to_numpy(), P.video.to_numpy()
    res = {}
    sets = {"full_neck": (F, FEATS), "full_sw": (Fs, FEATS),
            "full_sw+turn": (Fs.assign(t1=F.nose_off, t2=F.yaw_ratio), FEATS + ["t1", "t2"]),
            "turn_only": (F, ["nose_off", "yaw_ratio"]),
            "motion_only_neck": (F, ["wspd_med", "wspd_p90", "fing_med", "body_med", "head_med"]),
            "hands_motion_only_neck": (F, ["wspd_med", "wspd_p90", "fing_med"]),
            "posture_only_neck": (F, ["height_med", "raised", "face_med"])}
    S = {}
    for nm, (Fr, cols) in sets.items():
        S[nm] = oof(Fr[cols].to_numpy(float), y, g, 0)
        res[nm] = auc_ci(S[nm], y, g)
        print(nm, res[nm])
    no = F.nose_off.to_numpy()
    ok = np.isfinite(no)
    edges = np.quantile(no[ok], [0, .2, .4, .6, .8, 1])
    res["full_neck_within_nose_off_quintiles"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (no >= lo) & (no <= hi)
        res["full_neck_within_nose_off_quintiles"].append(
            {"range": [round(lo, 3), round(hi, 3)], "n_sig": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
             "auroc": auc_ci(S["full_neck"][m], y[m], g[m])})
    res["corr_fullneck_score_vs_nose_off"] = float(np.corrcoef(S["full_neck"][ok], no[ok])[0, 1])
    json.dump(res, open(HERE / "verify_orient.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ self-check
def selftest():
    T = 40
    k = np.zeros((T, 133, 3)); k[..., 2] = 0.9
    k[:, LSH, :2], k[:, RSH, :2] = (200, 100), (100, 100)       # sw 100 px, shoulder line y=100
    k[:, NOSE, :2] = (150, 50)                                   # neck 50 px
    k[:, LWR, :2], k[:, RWR, :2] = (190, 250), (110, 250)        # 1.5 sw below the line
    k[:, 91:112, :2] = k[:, [LWR], :2]; k[:, 112:133, :2] = k[:, [RWR], :2]
    f = feats(k)
    assert abs(f["height_med"] + 1.5) < 1e-9 and f["raised"] == 0 and f["wspd_med"] == 0 and f["valid"] == 1
    assert abs(f["yaw_ratio"] - 2.0) < 1e-9 and abs(feats(k, "neck")["height_med"] + 3.0) < 1e-9
    side = k.copy(); side[:, LSH, 0] = 175                      # turned side-on: sw 75, neck unchanged
    assert abs(feats(side, "neck")["height_med"] + 3.0) < 1e-9  # vertical scale is yaw-invariant
    assert feats(side)["height_med"] < -1.9                     # shoulder-width scale inflates it
    up = k.copy(); up[:, RWR, 1] = 100 + 10 * np.sin(np.arange(T))
    fu = feats(up)
    assert fu["raised"] == 1 and fu["wspd_med"] > 1
    g = np.repeat(np.arange(30), 10)
    assert all(not set(g[a]) & set(g[b]) for a, b in vfolds(g, 3))
    rng = np.random.default_rng(1)
    yy = np.r_[np.ones(150, int), np.zeros(150, int)]
    X = rng.normal(size=(300, 2)) + yy[:, None] * 1.5
    s = oof(X, yy, g, 0)
    assert roc_auc_score(yy, s) > 0.8
    a = nested_acc(X, yy, g, 0)
    assert 0.88 < a[yy == 1].mean() < 1.0
    pt, lo, hi = auc_ci(s, yy, g)
    assert lo <= pt <= hi
    print("selftest passed")


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "orient": orient}[sys.argv[1]]()
