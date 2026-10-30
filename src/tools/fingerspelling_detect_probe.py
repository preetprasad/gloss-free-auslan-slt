"""Is CONVERSATIONAL Auslan visually harder than BROADCAST Auslan? A task-level test.

Motivation (2026-09-09): hand kinematics say Auslan-Daily Communication is a visual outlier --
hand speed 9.86x citation form, hands at the face 2.38x -- while Auslan-Daily NEWS, which is also
continuous signing, matches citation form almost exactly (speed 1.10x, on-face 0.99x). If that is
real and not an artefact of the statistics, a genuine visual task should be markedly harder on
Communication than on News, at the same estimator and with the same descriptor.

Fingerspelling detection is the only labelled visual task available on BOTH splits (spotting is
Communication-only). Positives are annotated fingerspelling spans; negatives are length-matched
spans drawn from the same clips that do not overlap any annotation, so the classifier cannot win on
clip identity or span length. Folds are CLIP-DISJOINT.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.continuous_id_probe import span_descriptor, GROUPS  # noqa: E402


def logistic_auc(tr_x, tr_y, te_x, te_y, iters=600, lr=0.1):
    mu, sd = tr_x.mean(0), tr_x.std(0)
    sd = np.where(sd > 1e-7, sd, 1.0)
    x = np.c_[np.ones(len(tr_x)), (tr_x - mu) / sd]
    xt = np.c_[np.ones(len(te_x)), (te_x - mu) / sd]
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        w -= lr * (x.T @ (p - tr_y) / len(x) + 1e-3 * np.r_[0.0, w[1:]])
    s = xt @ w
    pos, neg = s[te_y == 1], s[te_y == 0]
    if not len(pos) or not len(neg):
        return np.nan, np.nan
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    pred = (s >= 0).astype(int)
    bal = 0.5 * ((pred[te_y == 1] == 1).mean() + (pred[te_y == 0] == 0).mean())
    return auc, float(bal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_csv", required=True)
    ap.add_argument("--clip_dirs", nargs="+", required=True)
    ap.add_argument("--comm_xlsx", default="data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fs = pd.read_csv(a.fs_csv)
    # a handful of rows carry no frame bounds; they cannot define a span either way
    n0 = len(fs)
    fs = fs.dropna(subset=["Begin_Frame", "End_Frame"]).copy()
    fs["Begin_Frame"] = fs.Begin_Frame.astype(int)
    fs["End_Frame"] = fs.End_Frame.astype(int)
    if len(fs) != n0:
        print(f"dropped {n0 - len(fs)} rows with missing frame bounds")
    comm = pd.read_excel(a.comm_xlsx)
    C = {str(r.Video_Clip_Name) for r in comm.itertuples()}
    fs["domain"] = fs.cid.map(lambda c: "Communication" if str(c) in C else "News")

    idx = {}
    for d in a.clip_dirs:
        for f in glob.glob(os.path.join(d, "*.npz")):
            idx.setdefault(os.path.basename(f)[:-4], f)

    spans = {}
    for r in fs.itertuples():
        spans.setdefault(str(r.cid), []).append((int(r.Begin_Frame), int(r.End_Frame)))

    rng = np.random.default_rng(a.seed)
    for dom in ("Communication", "News"):
        X, y, cl = [], [], []
        sub = fs[fs.domain == dom]
        for cid, grp in sub.groupby("cid"):
            p = idx.get(str(cid))
            if not p:
                continue
            with np.load(p) as z:
                k = z["KPS"].astype(np.float32)
            occupied = spans.get(str(cid), [])
            for r in grp.itertuples():
                b, e = max(0, int(r.Begin_Frame)), min(len(k), int(r.End_Frame))
                if e - b < 5:
                    continue
                v = span_descriptor(k[b:e], GROUPS["hands"], 8, 0.10, True, True)
                if v is None:
                    continue
                X.append(v); y.append(1); cl.append(str(cid))
                # length-matched negative from the SAME clip, not overlapping any annotation
                L = e - b
                for _ in range(20):
                    nb = int(rng.integers(0, max(1, len(k) - L)))
                    ne = nb + L
                    if any(not (ne <= ob or nb >= oe) for ob, oe in occupied):
                        continue
                    nv = span_descriptor(k[nb:ne], GROUPS["hands"], 8, 0.10, True, True)
                    if nv is not None:
                        X.append(nv); y.append(0); cl.append(str(cid))
                    break
        X, y, cl = np.array(X), np.array(y), np.array(cl)
        if len(X) < 60:
            print(f"{dom:<14} too few spans ({len(X)})")
            continue
        uc = np.unique(cl); rng.shuffle(uc)
        aucs, bals = [], []
        for part in np.array_split(uc, a.folds):
            te = np.isin(cl, part); tr = ~te
            if tr.sum() < 40 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            au, ba = logistic_auc(X[tr], y[tr], X[te], y[te])
            if np.isfinite(au):
                aucs.append(au); bals.append(ba)
        # shuffled-label control = empirical chance
        sh = []
        for i in range(5):
            yy = np.random.default_rng(a.seed + i).permutation(y)
            f = []
            for part in np.array_split(uc, a.folds):
                te = np.isin(cl, part); tr = ~te
                if tr.sum() < 40 or te.sum() < 10 or len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
                    continue
                au, _ = logistic_auc(X[tr], yy[tr], X[te], yy[te])
                if np.isfinite(au):
                    f.append(au)
            if f:
                sh.append(np.mean(f))
        print(f"{dom:<14} n={len(X):>5} ({int(y.sum())} pos)  clips {len(uc):>4}   "
              f"AUC {np.mean(aucs):.3f}   bal-acc {np.mean(bals):.3f}   [shuffled AUC {np.mean(sh):.3f}]")


if __name__ == "__main__":
    main()
