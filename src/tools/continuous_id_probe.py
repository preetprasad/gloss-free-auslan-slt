"""Can DWPose identify WHICH SIGN is being produced, in the CONTINUOUS domain?

THE GAP THIS CLOSES
-------------------
Every visual probe run on this project so far has been CROSS-domain: isolated MM-WLAuslan
prototypes matched against continuous Auslan-Daily windows (spotting recall 0.05), or citation
clips retrieved from continuous segments (3.5%).  Those numbers confound two very different
diagnoses:

  (i)  the pose representation does not carry sign identity at all  -> no decoder/objective work
       can help, and the fix must be at the input (hand resolution, RGB, a better estimator);
  (ii) the representation carries sign identity fine, but ISOLATED->CONTINUOUS transfer fails
       -> the representation is not the wall, and the headroom is in how the translator
       segments and exploits it.

This probe separates them by staying entirely WITHIN the continuous domain, using Auslan-Daily's
own `Sign_Spotting.xlsx`: 1,008 human-labelled sign spans over 120 glosses on real continuous
TRAIN clips.  No translation model, no checkpoint, no val/test manifest.

CONTROLS (all required -- the raw accuracy is meaningless without them)
  * CLIP-DISJOINT folds: a span's clip never appears in both train and test.  Without this a
    classifier can match on scene/signer identity; the 2026-09-08 retrieval baseline measured
    that same-video leakage roughly DOUBLES apparent performance.
  * shuffled-label control: the empirical chance level at this sample size, not the nominal 1/120.
  * span-length-only baseline: some signs are simply longer; length must not be the signal.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

# COCO-WholeBody-133 layout
GROUPS = {
    "body":       list(range(0, 17)),
    "face":       list(range(23, 91)),
    "hands":      list(range(91, 133)),
    "body_hands": list(range(0, 17)) + list(range(91, 133)),
    "all":        list(range(0, 133)),
}


# COCO-WholeBody-133 hands: 91 = left wrist + 20 pts, 112 = right wrist + 20 pts.
# Within a hand the ordering is wrist, thumb(1-4), index(5-8), MIDDLE(9-12), ring, pinky,
# so the wrist->middle-MCP distance (offset 9) is the palm length.
HANDS = ((91, 91 + 9), (112, 112 + 9))


def _hand_local(xy: np.ndarray) -> np.ndarray:
    """Express each hand around its own wrist, scaled by palm length.

    This is `hierarchical_pose_norm`'s idea ported to COCO-WholeBody-133 (the shipped
    implementation hard-requires HALPE-136 and therefore raises on every current run, which is
    why it has never been executed).  Motivation is measured, not assumed: the continuous-domain
    identification probe finds hands carry the most sign identity AND lose the most of it
    (40.0% -> 18.1% top-1), and under whole-frame standardisation finger detail is a tiny
    fraction of the signer-box variance.
    """
    out = xy.copy()
    for wrist, mid in HANDS:
        hi = wrist + 21
        w = xy[:, wrist:wrist + 1, :]
        scale = np.linalg.norm(xy[:, mid, :] - xy[:, wrist, :], axis=-1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)[:, None, None]
        out[:, wrist:hi, :] = np.clip((xy[:, wrist:hi, :] - w) / scale, -5.0, 5.0)
    return out


def span_descriptor(kps: np.ndarray, idx: list[int], n_t: int, min_conf: float,
                    use_velocity: bool, hand_local: bool = False) -> np.ndarray | None:
    """Frame-normalised, root-relative, time-resampled trajectory descriptor.

    Frame normalisation reproduces slt/data.py coord_norm=='frame' -- the space the translator
    actually consumes -- which also sidesteps the all-zero BBOX in every DWPose npz.
    """
    if kps.shape[0] < 3:
        return None
    xy = kps[..., :2].astype(np.float64)
    conf = kps[..., 2]
    T = xy.shape[0]
    flat = xy.reshape(T, -1)
    xy = (xy - flat.mean(1)[:, None, None]) / (flat.std(1)[:, None, None] + 1e-6)
    shoulders = xy[:, [5, 6]].mean(1)                     # torso-anchored, not frame-centre
    xy = xy - shoulders[:, None, :]
    if hand_local and xy.shape[1] >= 133:
        xy = _hand_local(xy)
    xy[conf < min_conf] = np.nan
    xy = xy[:, idx, :]
    # resample the span to a fixed number of timesteps so descriptors are comparable
    grid = np.linspace(0, T - 1, n_t)
    lo = np.floor(grid).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (grid - lo)[:, None, None]
    d = (1 - w) * xy[lo] + w * xy[hi]
    if use_velocity and n_t > 1:
        d = np.concatenate([d, np.diff(d, axis=0, prepend=d[:1])], axis=1)
    v = d.reshape(-1)
    if not np.isfinite(v).any():
        return None
    return np.nan_to_num(v, nan=0.0)


def nearest_centroid(tr_x, tr_y, te_x, k=5):
    """Return (top1 predictions, top-k class arrays) under cosine similarity to class centroids."""
    classes = np.unique(tr_y)
    cents = np.stack([tr_x[tr_y == c].mean(0) for c in classes])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    q = te_x / (np.linalg.norm(te_x, axis=1, keepdims=True) + 1e-9)
    sim = q @ cents.T
    order = np.argsort(-sim, axis=1)[:, :min(k, len(classes))]
    return classes[order[:, 0]], classes[order]


def clip_disjoint_eval(X, y, clips, rng, folds=5, per=None):
    """Top-1 with clip-disjoint folds; a clip is never in both sides. `per` (dict) receives index -> top-1 correct."""
    uc = np.unique(clips)
    rng.shuffle(uc)
    parts = np.array_split(uc, folds)
    correct = top5 = total = 0
    for p in parts:
        te = np.isin(clips, p)
        tr = ~te
        keep = np.isin(y[te], np.unique(y[tr]))       # only score classes the fold can predict
        if tr.sum() < 20 or keep.sum() == 0:
            continue
        gold = y[te][keep]
        pred, topk = nearest_centroid(X[tr], y[tr], X[te][keep])
        if per is not None:
            per.update(zip(np.flatnonzero(te)[keep].tolist(), (pred == gold).tolist()))
        correct += int((pred == gold).sum())
        top5 += int((topk == gold[:, None]).any(1).sum())
        total += int(keep.sum())
    if not total:
        return float("nan"), float("nan"), 0
    return correct / total, top5 / total, total


def select_rows(y, max_per_class: int, min_per_class: int, seed: int) -> np.ndarray:
    """Row indices kept by the per-class subsample, then by the minimum class size (the rule main() applies)."""
    sel = np.arange(len(y))
    if max_per_class:
        srng = np.random.default_rng(seed); sel = []
        for c in np.unique(y):
            ix = np.flatnonzero(y == c); srng.shuffle(ix); sel.extend(ix[:max_per_class])
        sel = np.sort(np.array(sel))
    u, n = np.unique(y[sel], return_counts=True)
    return sel[np.isin(y[sel], u[n >= min_per_class])]


def main() -> None:
    import pandas as pd      # only the command line needs pandas; the functions above import without it

    ap = argparse.ArgumentParser()
    ap.add_argument("--spot_csv", required=True, help="safe spotting spans (Comm TRAIN only)")
    ap.add_argument("--clip_dirs", nargs="+", required=True)
    ap.add_argument("--n_timesteps", type=int, default=8)
    ap.add_argument("--min_conf", type=float, default=0.10)
    ap.add_argument("--min_per_class", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_per_class", type=int, default=0, help="subsample to match another domain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_pred", default="", help="CSV of per-span top-1 correctness, for paired tests")
    a = ap.parse_args()
    dumped = []

    d = pd.read_csv(a.spot_csv, dtype={"Video_Name_Clip": str})   # MM-WLAuslan ids are zero-padded ('05841')
    index = {}
    for cd in a.clip_dirs:
        if os.path.isdir(cd):
            for f in os.listdir(cd):
                if f.endswith(".npz"):
                    index.setdefault(f[:-4], os.path.join(cd, f))

    rows, missing = [], 0
    for r in d.itertuples():
        cid = re.sub(r"_signer\.mp4$", "", str(r.Video_Name_Clip))
        p = index.get(cid)
        if p is None:
            missing += 1
            continue
        rows.append((cid, str(r.Query_Gloss).strip().lower(), int(r.Begin_Frame), int(r.End_Frame), p))
    print(f"spans usable {len(rows)} (missing clips {missing}); "
          f"clips {len({r[0] for r in rows})}; glosses {len({r[1] for r in rows})}")

    cache: dict[str, np.ndarray] = {}
    variants = [(n, i, True, False) for n, i in GROUPS.items()]
    variants += [(n + " [HAND-LOCAL]", GROUPS[n], True, True) for n in ("hands", "body_hands", "all")]
    for name, idx, use_v, hloc in variants:
        if True:
            X, y, cl, ln, ky = [], [], [], [], []
            for cid, gloss, b, e, p in rows:
                if p not in cache:
                    with np.load(p) as z:
                        cache[p] = z["KPS"].astype(np.float32)
                k = cache[p]
                b2, e2 = max(0, b), min(len(k), max(b + 3, e))
                seg = k[b2:e2]
                v = span_descriptor(seg, idx, a.n_timesteps, a.min_conf, use_v, hloc)
                if v is None:
                    continue
                X.append(v); y.append(gloss); cl.append(cid); ln.append(len(seg)); ky.append(f"{cid}:{b}:{e}")
            X = np.array(X); y = np.array(y); cl = np.array(cl); ln = np.array(ln, float); ky = np.array(ky)
            sel = select_rows(y, a.max_per_class, a.min_per_class, a.seed)
            X, y, cl, ln, ky = X[sel], y[sel], cl[sel], ln[sel], ky[sel]
            n_cls = len(np.unique(y))
            rng = np.random.default_rng(a.seed)
            per = {}
            acc, acc5, n = clip_disjoint_eval(X, y, cl, rng, a.folds, per)
            dumped += [(name, ky[i], y[i], int(c)) for i, c in sorted(per.items())]
            # controls
            rng2 = np.random.default_rng(a.seed)
            sh = [clip_disjoint_eval(X, rng2.permutation(y), cl, np.random.default_rng(a.seed + i), a.folds)[0]
                  for i in range(5)]
            lacc = clip_disjoint_eval(ln.reshape(-1, 1), y, cl, np.random.default_rng(a.seed), a.folds)[0]
            tag = f"{name}{'+vel' if use_v else ''}"
            print(f"  {tag:<16} dim {X.shape[1]:>5}  classes {n_cls:>3}  n {n:>4}  "
                  f"TOP-1 {acc:6.1%}  TOP-5 {acc5:6.1%}   [chance {1/n_cls:5.2%} | shuffled "
                  f"{np.mean(sh):5.2%} | length-only {lacc:5.2%}]   lift {acc/max(np.mean(sh),1e-9):5.1f}x")
    if a.dump_pred:
        pd.DataFrame(dumped, columns=["variant", "span", "gloss", "correct"]).to_csv(a.dump_pred, index=False)


if __name__ == "__main__":
    main()
