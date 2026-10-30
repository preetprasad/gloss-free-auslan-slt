"""Fit the frozen signing-activity model shipped in demo/static/engine/activity_model.js.

Features: the 8 kinematic features of results/signing_activity/signing_activity.py (KIN). Fitted on the DEV split only
(val signer clips + dev Non_Signer crops), with the 60 tracker ground-truth clips and every crop from their parent clips
removed from the fit. The TEST split is scored once. Protocol: results/person_labels/PREREG_PERSON_LABELS_2026-09-30.md.

    python fit_activity_model.py            # prints the report, writes activity_model.json and the JS module
    python fit_activity_model.py --selftest
Use the repo .venv or ~/.venvs/honours_py312_twin (never the pyenv 3.14 interpreter: numpy #28681).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV = ROOT / "results/signing_activity/features_oof.csv"
GT = ROOT / "results/tracker_v2/gt_clips.json"
KIN = ["wrist_speed_med", "wrist_speed_p90", "finger_motion_med", "hand_height_med", "raised_share",
       "hand_face_dist_med", "body_motion_med", "head_motion_med"]      # == signing_activity.KIN


def auc(pos, neg):
    """P(pos > neg) + 0.5 ties, by ranks."""
    s = np.r_[pos, neg]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot(score, y, video, B=2000, seed=0):
    """Video-cluster bootstrap of the AUROC: (point, lo, hi)."""
    by = {}
    for s, t, v in zip(score, y, video):
        by.setdefault(v, ([], []))[int(t)].append(s)
    vids = list(by)
    rng = np.random.default_rng(seed)

    def stat(vs):
        p = [x for v in vs for x in by[v][1]]
        n = [x for v in vs for x in by[v][0]]
        return auc(p, n) if p and n else None
    pt = stat(vids)
    draws = [stat([vids[i] for i in rng.integers(0, len(vids), len(vids))]) for _ in range(B)]
    draws = [d for d in draws if d is not None]
    return pt, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score_by_formula(m, X):
    """intercept + sum(coef * (impute(x) - mean) / scale): the arithmetic of demo/static/engine/activity.js."""
    X = np.where(np.isnan(X), np.asarray(m["median"]), X)
    return m["intercept"] + ((X - np.asarray(m["mean"])) / np.asarray(m["scale"])) @ np.asarray(m["coef"])


def scene_top(df, score):
    """Share of signer clips whose score exceeds every Non_Signer crop from the same clip (ties count half)."""
    df = df.assign(score=score)
    neg = df[df.y == 0].groupby("parent").score.max()
    sig = df[df.y == 1].set_index("cid").score
    common = neg.index.intersection(sig.index)
    win = [1.0 if sig[c] > neg[c] else 0.5 if sig[c] == neg[c] else 0.0 for c in common]
    return float(np.mean(win)), len(win)


def main():
    df = pd.read_csv(CSV)
    gt = json.load(open(GT))
    gt_ids = {c for part in gt.values() for c, _ in part}
    assert len(gt_ids) == 60, len(gt_ids)
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    fit = dev[~dev.cid.isin(gt_ids) & ~dev.parent.isin(gt_ids)]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(fit[KIN].to_numpy(float), fit.y.to_numpy())
    imp, sc, lr = model[0], model[1], model[2]
    out = {
        "features": KIN, "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "fit": {"rows": int(len(fit)), "signers": int(fit.y.sum()), "non_signers": int((1 - fit.y).sum()),
                "removed_gt_rows": int(len(dev) - len(fit)), "split": "dev"},
        "source": "results/signing_activity/features_oof.csv",
    }
    s_test = model.decision_function(test[KIN].to_numpy(float))
    formula = score_by_formula(out, test[KIN].to_numpy(float))            # what activity.js computes
    assert np.allclose(formula, s_test, atol=1e-9), np.abs(formula - s_test).max()
    p, lo, hi = cluster_boot(s_test, test.y.to_numpy(), test.video.to_numpy())
    top, n_scenes = scene_top(test, s_test)
    s_fit = model.decision_function(fit[KIN].to_numpy(float))
    out["test"] = {"auroc": [p, lo, hi], "signers": int(test.y.sum()), "non_signers": int((1 - test.y).sum()),
                   "signer_above_every_listener_in_scene": top, "scenes": n_scenes}
    out["fit_auroc_in_sample"] = auc(s_fit[fit.y.to_numpy() == 1], s_fit[fit.y.to_numpy() == 0])
    print(json.dumps({k: v for k, v in out.items() if k not in ("median", "mean", "scale", "coef")}, indent=1))
    print("coef (standardised):", dict(zip(KIN, np.round(out["coef"], 3))))
    (HERE / "activity_model.json").write_text(json.dumps(out, indent=1))
    js = ROOT / "demo/static/engine/activity_model.js"
    js.write_text(
        "// GENERATED by results/person_labels/fit_activity_model.py - do not edit. Frozen signing-activity model.\n"
        "// Fitted on the DEV split only (val signer clips + dev Non_Signer crops), with the 60 tracker ground-truth clips\n"
        "// removed; the TEST split was scored once. Provenance and numbers: activity_model.json next to the fit script and\n"
        "// results/person_labels/PREREG_PERSON_LABELS_2026-09-30.md. Score = intercept + sum(coef * (impute(x) - mean) / scale).\n"
        f"export const ACTIVITY_MODEL = Object.freeze({json.dumps({k: out[k] for k in ('features', 'median', 'mean', 'scale', 'coef', 'intercept', 'fit', 'test')}, indent=2)});\n")
    print("wrote", HERE / "activity_model.json", "and", js)


def selftest():
    y = np.r_[np.ones(50), np.zeros(50)]
    s = np.r_[np.arange(50) + 100.0, np.arange(50)]
    assert auc(s[y == 1], s[y == 0]) == 1.0 and auc([1.0], [1.0]) == 0.5
    p, lo, hi = cluster_boot(s, y, np.repeat(np.arange(10), 10)[np.r_[np.arange(0, 100)]], B=50)
    assert p == 1.0 and lo <= p <= hi
    df = pd.DataFrame({"cid": ["a", "n1", "n2"], "parent": ["a", "a", "a"], "y": [1, 0, 0]})
    assert scene_top(df, np.array([2.0, 1.0, 0.0])) == (1.0, 1)
    assert scene_top(df, np.array([0.5, 1.0, 0.0])) == (0.0, 1)
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
