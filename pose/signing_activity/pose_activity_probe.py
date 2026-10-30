"""Can a cheap pose-motion signing-activity detector separate Auslan-Daily signers from the Non_Signer crops that the
translator's own confidence cannot (AUROC 0.51, results/NONSIGNER_GATE_2026-09-22.md)?   Stages: selftest | run

Exploratory, not pre-registered. Features follow Moryossef et al. 2020 (arXiv 2008.04637, sec. 3.1, Eq. 1): keypoint
displacement between consecutive frames, scaled by fps, with poses normalised to shoulder width 1. Here each frame is
also shoulder-centred (removes camera pan and walking) and aggregated per clip, because Auslan-Daily has clip labels only.

Populations are exactly the non-signer gate's (reused from src/tools/nonsigner_gate.py):
  positives  = the 800 test / 800 val signer clips (arm ii = person-tracked pose, primary; arm i = stored index-0 pose)
  negatives  = Non_Signer crops, test 847 / dev 831, minus the B3 same-person flags (16 / 26) -> 831 / 805
Everything (orientation, scaler, logistic regression, thresholds) is fitted on DEV (val signers + dev negatives);
TEST is scored once. Harness gate: the translator-confidence AUROC recomputed here must equal the recorded value.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/tools")]
import nonsigner_gate as g  # noqa: E402  (norm_skel, skel_dist, wrist_speed, auroc, cluster_boot, negatives, manifest)

FPS = 25.0
HAND = np.r_[91:133]
ARM = np.r_[7:11]
FACE = np.r_[23:91]
TIPS, ROOTS = np.r_[95, 99, 103, 107, 111, 116, 120, 124, 128, 132], np.r_[[91] * 5, [112] * 5]
FEATS = ["hand_med", "hand_p90", "arm_med", "finger_med", "face_med", "wrist_height", "hand_conf", "valid_frac"]
EXTRA = ["elbow_frac"]   # scored singly, NOT in the logistic regression (added after the first run, see README)
ELBOW_RAD = 1.1          # SDTrack heuristic threshold (Albanie et al. 2021 tech. report, sec. 3)
B3_THR = 0.4008016274869442                     # recorded null_p1_threshold (nonsigner_gate_results.json, B3)
REC_AUROC = {"i_production": 0.5094930806257522, "ii_tracked": 0.5019073405535499}


def clip_features(kps):
    """Per-clip aggregates of per-frame shoulder-normalised motion (shoulder widths per second)."""
    N = g.norm_skel(kps)
    fl = {k: [] for k in ("hand", "arm", "face", "finger")}
    for a, b in zip(N[:-1], N[1:]):
        if a is None or b is None:
            continue
        (xa, ca), (xb, cb) = a, b
        for k, idx in (("hand", HAND), ("arm", ARM), ("face", FACE)):
            m = (ca[idx] >= g.CONF) & (cb[idx] >= g.CONF)
            fl[k].append(np.linalg.norm(xb[idx][m] - xa[idx][m], axis=1).mean() * FPS if m.any() else np.nan)
        m = (ca[TIPS] >= g.CONF) & (ca[ROOTS] >= g.CONF) & (cb[TIPS] >= g.CONF) & (cb[ROOTS] >= g.CONF)
        ra, rb = xa[TIPS] - xa[ROOTS], xb[TIPS] - xb[ROOTS]
        fl["finger"].append(np.linalg.norm(rb[m] - ra[m], axis=1).mean() * FPS if m.any() else np.nan)
    ok = [n for n in N if n is not None]
    hy = [min(x[j, 1] for j in (9, 10) if c[j] >= g.CONF) for x, c in ok if max(c[9], c[10]) >= g.CONF]
    act, meas = 0, 0
    for x, c in ok:                                      # SDTrack-style: interior elbow angle < 1.1 rad on either arm
        angs = []
        for sh, el, wr in ((5, 7, 9), (6, 8, 10)):
            if min(c[sh], c[el], c[wr]) >= g.CONF:
                u, v = x[sh] - x[el], x[wr] - x[el]
                nu = np.linalg.norm(u) * np.linalg.norm(v)
                if nu > 1e-9:
                    angs.append(np.arccos(np.clip(u @ v / nu, -1, 1)))
        if angs:
            meas += 1; act += min(angs) < ELBOW_RAD
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    return {"hand_med": med(np.array(fl["hand"])),
            "hand_p90": float(np.nanpercentile(fl["hand"], 90)) if np.isfinite(fl["hand"]).any() else np.nan,
            "arm_med": med(np.array(fl["arm"])), "finger_med": med(np.array(fl["finger"])),
            "face_med": med(np.array(fl["face"])),
            "wrist_height": -float(np.median(hy)) if hy else np.nan,       # higher = hands higher (y points down)
            "hand_conf": float(np.mean([c[HAND].mean() for _, c in ok])) if ok else np.nan,
            "valid_frac": len(ok) / max(len(N), 1), "elbow_frac": act / meas if meas else np.nan}


def selftest():
    k = np.zeros((10, 133, 3)); k[..., 2] = 1.0
    k[:, 5, :2], k[:, 6, :2] = (0, 0), (10, 0)          # shoulder width 10 px
    f = clip_features(k)
    assert f["hand_med"] == 0 and f["arm_med"] == 0 and f["valid_frac"] == 1.0, f
    k[:, HAND, 0] += np.arange(10)[:, None] * 10.0      # every hand kp moves 1 shoulder width per frame
    f = clip_features(k)
    assert abs(f["hand_med"] - FPS) < 1e-9 and f["finger_med"] == 0, f   # rigid hand translation: no articulation
    k[:, 5, 2] = 0.0                                     # no confident shoulders -> nothing valid
    f = clip_features(k)
    assert f["valid_frac"] == 0 and np.isnan(f["hand_med"]), f
    k[:, 5, 2] = 1.0
    k[:, 7, :2], k[:, 9, :2] = (0, 10), (0, 0)          # elbow straight below shoulder, wrist back at shoulder: angle 0
    assert clip_features(k)["elbow_frac"] == 1.0
    k[:, 9, :2] = (0, 20)                               # straight arm: angle pi -> inactive
    k[:, 10, 2] = 0.0
    assert clip_features(k)["elbow_frac"] == 0.0
    assert abs(g.auroc(-np.array([2.0]), -np.array([1.0])) - 1.0) < 1e-12   # orientation used below
    print("selftest OK")


def run():
    t0 = time.time()
    RES, NS = g.RES, g.NS
    D = json.load(open(NS / "decodes.json"))
    sd = pd.read_excel(g.DS / "Signer Detection/Signer_Detection.xlsx")
    vid = dict(zip(sd.Video_Clip_Name, sd.Video_Name))
    neg = g.negatives().copy()
    neg["cid"] = neg.Person_Video_Clip.str[:-4]
    pmap, split_of = dict(zip(neg.cid, neg.Video_Clip_Name)), dict(zip(neg.cid, neg.Split))
    test = g.manifest(RES / "TEST_manifest_official.tsv")
    val = g.manifest(RES / "_local_val_manifest.tsv")
    test_ids, val_ids = [g.cid(p) for p, _ in test], [g.cid(p) for p, _ in val]
    load = lambda p: np.load(p)["KPS"]
    npz = lambda c: load(NS / "npz" / f"{c}.npz")
    trk = lambda c: load(NS / "tracked" / f"{c}.npz")

    # negatives: same population as the gate (drop no-person crops and B3 same-person flags)
    keep = {}
    for c in neg.cid:
        if D["neg"][c]["vT"] == 0:
            continue
        x, y = npz(c), trk(pmap[c])
        if abs(len(x) - len(y)) <= 2:
            n = min(len(x), len(y))
            d = g.skel_dist(x[:n], y[:n])
            if d is not None and d < B3_THR:
                continue
        keep[c] = split_of[c]
    nt = [c for c, s in keep.items() if s == "test"]
    nd = [c for c, s in keep.items() if s == "dev"]
    assert (len(nt), len(nd)) == (831, 805), (len(nt), len(nd))

    # harness gate: the recorded translator-confidence AUROC, same populations
    for arm, key in (("i_production", "sig_test"), ("ii_tracked", "trk_test")):
        a = g.auroc([D[key][c]["ce"] for c in test_ids], [D["neg"][c]["ce"] for c in nt])
        assert abs(a - REC_AUROC[arm]) < 1e-9, (arm, a)
    print("harness gate OK: translator-confidence AUROC reproduces", REC_AUROC)

    tf = time.time()
    FN = {c: clip_features(npz(c)) for c in nt + nd}
    per_clip_ms = (time.time() - tf) / len(FN) * 1000
    lens = [len(npz(c)) for c in nt + nd]
    out = {"n": {"test_sig": len(test_ids), "val_sig": len(val_ids), "test_neg": len(nt), "dev_neg": len(nd)},
           "feature_ms_per_clip": per_clip_ms, "feature_ms_per_frame": per_clip_ms / float(np.mean(lens))}
    ws_rule = np.nanpercentile([g.wrist_speed(trk(c)) for c in val_ids], 10)   # the gate's quiescence rule, arm ii
    quiescent = {c for c in nt if g.wrist_speed(npz(c)) < ws_rule}

    for arm, sig, ce_key in (("ii_tracked", lambda p, c: trk(c), "trk"), ("i_production", lambda p, c: load(p), "sig")):
        FS = {c: clip_features(sig(p, c)) for p, c in [(p, g.cid(p)) for p, _ in test + val]}
        X = lambda cs, F: np.array([[F[c][f] for f in FEATS + EXTRA] for c in cs], float)
        Xd = np.vstack([X(val_ids, FS), X(nd, FN)]); yd = np.r_[np.ones(len(val_ids)), np.zeros(len(nd))]
        Xt = np.vstack([X(test_ids, FS), X(nt, FN)]); yt = np.r_[np.ones(len(test_ids)), np.zeros(len(nt))]
        A = {"nan_features_test": int(np.isnan(Xt).any(1).sum()), "single": {}}
        videos_t = [vid[c] for c in test_ids] + [vid[pmap[c]] for c in nt]

        def boot(score):
            ts, tn = {}, {}
            for s, y, v in zip(score, yt, videos_t):
                (ts if y else tn).setdefault(v, []).append(s)
            allv = sorted(set(ts) | set(tn))
            return g.cluster_boot(allv, lambda vs: g.auroc([-x for v in vs for x in ts.get(v, [])],
                                                           [-x for v in vs for x in tn.get(v, [])])
                                  if any(tn.get(v) for v in vs) and any(ts.get(v) for v in vs) else None)

        for i, f in enumerate(FEATS + EXTRA):  # orientation chosen on dev, then scored on test
            med = np.nanmedian(Xd[:, i])
            sgn = 1.0 if g.auroc(-np.nan_to_num(Xd[yd == 1, i], nan=med), -np.nan_to_num(Xd[yd == 0, i], nan=med)) >= 0.5 else -1.0
            s = sgn * np.nan_to_num(Xt[:, i], nan=med)
            A["single"][f] = {"sign": sgn, "test_auroc": boot(s)}
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
        nf = len(FEATS)
        clf.fit(Xd[:, :nf], yd)
        s_t, s_d = clf.decision_function(Xt[:, :nf]), clf.decision_function(Xd[:, :nf])
        A["logreg"] = {"coef_std": dict(zip(FEATS, map(float, clf[-1].coef_[0]))), "test_auroc": boot(s_t),
                       "dev_auroc_insample": g.auroc(-s_d[yd == 1], -s_d[yd == 0])}
        tau = json.load(open(NS / "nonsigner_gate_results.json"))[arm]["q0.5"]["tau"]
        ce_t = np.r_[[D[f"{ce_key}_test"][c]["ce"] for c in test_ids], [D["neg"][c]["ce"] for c in nt]]
        sig_t, neg_t = yt == 1, yt == 0
        q_mask = np.r_[np.zeros(len(test_ids), bool), [c in quiescent for c in nt]]
        ops = {}
        for keep_sig in (0.99, 0.95, 0.90, 0.80):
            thr = np.quantile(s_d[yd == 1], 1 - keep_sig)                   # fixed on DEV signers
            det = s_t >= thr
            both = det & (ce_t <= tau)
            ops[str(keep_sig)] = {"thr": float(thr),
                                  "det_pass_signers": float(det[sig_t].mean()), "det_pass_nonsigners": float(det[neg_t].mean()),
                                  "det_pass_quiescent_neg": float(det[q_mask].mean()),
                                  "det_pass_moving_neg": float(det[neg_t & ~q_mask].mean()),
                                  "both_pass_signers": float(both[sig_t].mean()), "both_pass_nonsigners": float(both[neg_t].mean())}
        A["ops"] = ops
        A["ce_gate_alone"] = {"tau": tau, "pass_signers": float((ce_t <= tau)[sig_t].mean()),
                              "pass_nonsigners": float((ce_t <= tau)[neg_t].mean())}
        idx = {c: i for i, c in enumerate(test_ids)}
        paired = [s_t[idx[pmap[c]]] > s_t[len(test_ids) + j] for j, c in enumerate(nt)]
        A["paired_signer_above_own_nonsigner"] = float(np.mean(paired))
        A["n_quiescent_neg_test"] = len(quiescent)
        out[arm] = A
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(HERE / "probe_results.json", "w"), indent=1)
    for arm in ("ii_tracked", "i_production"):
        A = out[arm]
        print(f"\n== {arm}: logistic regression test AUROC {A['logreg']['test_auroc']}  (dev in-sample {A['logreg']['dev_auroc_insample']:.3f})")
        print("  single-feature test AUROC:", {f: round(v["test_auroc"][0], 3) for f, v in A["single"].items()})
        print("  std coefs:", {k: round(v, 2) for k, v in A["logreg"]["coef_std"].items()})
        print("  ce gate alone:", A["ce_gate_alone"])
        for k, v in A["ops"].items():
            print(f"  keep {k} of dev signers:", {kk: round(vv, 3) for kk, vv in v.items()})
        print("  paired signer > own non-signer:", round(A["paired_signer_above_own_nonsigner"], 3),
              "| quiescent test negatives:", A["n_quiescent_neg_test"], "| NaN rows:", A["nan_features_test"])
    print(f"\nfeature cost {out['feature_ms_per_clip']:.2f} ms/clip = {out['feature_ms_per_frame']:.3f} ms/frame (python, M3 Max)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("selftest", "run"))
    a = ap.parse_args()
    selftest() if a.stage == "selftest" else (selftest(), run())
