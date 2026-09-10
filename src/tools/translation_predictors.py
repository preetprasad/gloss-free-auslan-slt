"""Does anything about the INPUT predict whether a clip is translated well?

Every per-clip analysis this project has run is REFERENCE-side (how many content words the target
has, whether the target was seen in training). Those cannot be computed at inference time, so even
when they correlate they cannot drive a system. This asks the inference-time question instead: given
only the pose clip, can we tell whether the translation will be any good?

It matters two ways.
  * If YES -> a confidence/abstention signal is constructible, and the deployment brief's plan to show
    "tracking quality but never translation confidence" collapses, because tracking quality WOULD BE a
    translation-confidence proxy.
  * If NO -> the failure is uniform in the strongest available sense: not merely unpredictable from the
    reference, but unpredictable from the input the model actually consumes.

Controls, because a correlation hunt over 12 features x 3 outcomes will manufacture significance:
  * shuffled-outcome null band (the p95 of |r| under 500 permutations) -- the honest threshold,
  * a reference-side feature (content-word count) whose uninformativeness is already on record,
  * a cross-validated multivariate model, since features could be jointly predictive while individually
    weak. Grouped by VIDEO so a fold cannot memorise a session.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.stitch_lexicon import content_words                      # noqa: E402
from slt.metrics import detokenize                                  # noqa: E402

L_HAND, R_HAND = slice(91, 112), slice(112, 133)


def clip_features(kps: np.ndarray, max_src_frames: int = 256) -> dict | None:
    """Inference-time-available descriptors. Conventions copied from hand_visibility_cv.py
    (shoulder-width scale, 0.35 proximity thresholds) so the two analyses are comparable."""
    if kps.shape[0] < 3 or kps.shape[1] < 133:
        return None
    conf, xy = kps[..., 2], kps[..., :2]
    scale = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
    scale = np.where(scale > 1e-3, scale, np.nan)
    if np.all(np.isnan(scale)):
        return None
    out = {"n_frames": float(kps.shape[0]),
           "subsampled": float(kps.shape[0] > max_src_frames)}

    # --- estimator health ---
    out["body_conf"] = float(np.nanmean(conf[:, :17]))
    hconf = np.concatenate([conf[:, L_HAND].mean(1), conf[:, R_HAND].mean(1)])
    out["hand_conf"] = float(np.nanmean(hconf))
    out["hand_conf_p10"] = float(np.nanpercentile(hconf, 10))
    out["conf_std"] = float(np.nanstd(hconf))                      # estimation instability over time
    out["invalid_frac"] = float(np.mean(np.all(conf < 0.1, axis=1)))

    # --- framing / geometry ---
    sizes = [np.linalg.norm(xy[:, hs].max(1) - xy[:, hs].min(1), axis=-1) / scale
             for hs in (L_HAND, R_HAND)]
    out["hand_size_rel"] = float(np.nanmean(np.concatenate(sizes)))
    lh, rh = xy[:, L_HAND].mean(1), xy[:, R_HAND].mean(1)
    face = xy[:, 23:91].mean(1)
    out["hands_overlap_frac"] = float(np.nanmean(np.linalg.norm(lh - rh, axis=-1) / scale < 0.35))
    out["hand_on_face_frac"] = float(np.nanmean(
        np.minimum(np.linalg.norm(lh - face, axis=-1),
                   np.linalg.norm(rh - face, axis=-1)) / scale < 0.35))

    # --- motion ---
    sp = np.linalg.norm(np.diff(lh, axis=0), axis=-1) / scale[:-1]
    out["hand_speed"] = float(np.nanmean(sp))
    out["pause_frac"] = float(np.nanmean(sp < 0.02))               # near-still frames
    torso = xy[:, [5, 6, 11, 12]].mean(1)
    out["body_motion"] = float(np.nanmean(
        np.linalg.norm(np.diff(torso, axis=0), axis=-1) / scale[:-1]))  # signer/camera drift
    return out


def outcomes(hyp: str, ref: str) -> dict:
    """Per-clip quality. Content recall is the project's own measure; chrF is stable on short text."""
    import sacrebleu
    h, r = detokenize(hyp), detokenize(ref)
    cw_r, cw_h = content_words(ref), set(content_words(hyp))
    return {"content_recall": float(np.mean([w in cw_h for w in cw_r])) if cw_r else np.nan,
            "any_content": float(any(w in cw_h for w in cw_r)) if cw_r else np.nan,
            "chrf": float(sacrebleu.sentence_chrf(h, [r]).score),
            "exact": float(h.strip() == r.strip())}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip_dir", default="results/test_clips_dwpose")
    p.add_argument("--manifest", default="comm_run/manifests/test.tsv")
    p.add_argument("--hyps", nargs="+",
                   default=["results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl",
                            "results/semantic_adequacy_grading/hyps_vlp.jsonl",
                            "results/semantic_adequacy_grading/hyps_rgbfusion.jsonl"])
    p.add_argument("--key", default="hyp_beam")
    p.add_argument("--perm", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/translation_predictors.json")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]

    rows = [l.rstrip("\n").split("\t") for l in open(root / a.manifest)]
    names = [Path(r[0]).name for r in rows]
    refs = [r[1] for r in rows]

    feats, keep = [], []
    for i, n in enumerate(names):
        f = root / a.clip_dir / n
        if not f.exists():
            continue
        d = clip_features(np.load(f)["KPS"])
        if d is None:
            continue
        feats.append(d); keep.append(i)
    print(f"clips with features: {len(keep)}/{len(names)}")
    fnames = list(feats[0])
    X = np.array([[f[k] for k in fnames] for f in feats], float)
    videos = np.array(["_".join(names[i].split("_")[:2]) for i in keep])
    # reference-side comparison feature (NOT inference-time available) -- known uninformative
    ref_cw = np.array([len(content_words(refs[i])) for i in keep], float)

    models = {}
    for hp in a.hyps:
        recs = [json.loads(l) for l in open(root / hp)]
        assert len(recs) == len(names), f"{hp}: {len(recs)} vs {len(names)}"
        assert all(r["ref"].strip() == t.strip() for r, t in zip(recs, refs)), f"{hp}: ref mismatch"
        models[Path(hp).stem.replace("hyps_", "")] = [outcomes(recs[i][a.key], refs[i]) for i in keep]

    onames = list(next(iter(models.values()))[0])
    rng = np.random.default_rng(a.seed)
    print(f"\ninput-side features: {len(fnames)}   outcomes: {onames}   models: {list(models)}")
    print(f"\n{'':22}" + "".join(f"{o:>16}" for o in onames))

    report, nulls = {}, {}
    for o in onames:
        ys = np.array([[r[o] for r in models[m]] for m in models], float)
        y = np.nanmean(ys, axis=0)                                  # consensus across the 3 models
        best = 0.0
        for _ in range(a.perm):
            yp = y.copy(); rng.shuffle(yp)
            best = max(best, max(abs(corr(X[:, j], yp)) for j in range(X.shape[1])))
        nulls[o] = best                                             # max-|r| over features, per perm-set
    # null band: p95 of the per-permutation max |r| (family-wise, so it already covers the 12 features)
    for o in onames:
        ys = np.array([[r[o] for r in models[m]] for m in models], float)
        y = np.nanmean(ys, axis=0)
        mx = []
        for _ in range(a.perm):
            yp = y.copy(); rng.shuffle(yp)
            mx.append(max(abs(corr(X[:, j], yp)) for j in range(X.shape[1])))
        nulls[o] = float(np.percentile(mx, 95))

    for j, fn in enumerate(fnames):
        line = f"{fn:22}"
        for o in onames:
            ys = np.array([[r[o] for r in models[m]] for m in models], float)
            r = corr(X[:, j], np.nanmean(ys, axis=0))
            flag = "*" if abs(r) > nulls[o] else " "
            line += f"{r:>15.3f}{flag}"
            report.setdefault(fn, {})[o] = r
        print(line)
    line = f"{'[ref] n_content_words':22}"
    for o in onames:
        ys = np.array([[r[o] for r in models[m]] for m in models], float)
        r = corr(ref_cw, np.nanmean(ys, axis=0))
        line += f"{r:>15.3f}{'*' if abs(r) > nulls[o] else ' '}"
        report.setdefault("[ref] n_content_words", {})[o] = r
    print(line)
    print(f"{'NULL BAND (p95 |r|)':22}" + "".join(f"{nulls[o]:>16.3f}" for o in onames))
    print("  * = exceeds the family-wise shuffled-outcome null band")

    # ---- multivariate, video-grouped CV: are they JOINTLY predictive? ----
    y = np.nanmean(np.array([[r["any_content"] for r in models[m]] for m in models], float), axis=0)
    yb = (y > 0.5).astype(int)
    Xz = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1)
    uv = np.unique(videos); rng.shuffle(uv)
    folds = np.array_split(uv, 5)
    aucs, aucs_shuf = [], []
    yshuf = yb.copy(); rng.shuffle(yshuf)
    for target, store in ((yb, aucs), (yshuf, aucs_shuf)):
        for f in folds:
            te = np.isin(videos, f); tr = ~te
            if te.sum() < 15 or len(np.unique(target[tr])) < 2 or len(np.unique(target[te])) < 2:
                continue
            w = np.zeros(Xz.shape[1] + 1)
            Xt = np.hstack([Xz[tr], np.ones((tr.sum(), 1))])
            for _ in range(600):                                     # plain logistic GD + L2
                pr = 1 / (1 + np.exp(-Xt @ w))
                w -= 0.5 * (Xt.T @ (pr - target[tr]) / tr.sum() + 0.01 * w)
            s = np.hstack([Xz[te], np.ones((te.sum(), 1))]) @ w
            pos, neg = s[target[te] == 1], s[target[te] == 0]
            store.append(float((pos[:, None] > neg[None, :]).mean()
                               + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    print(f"\nMULTIVARIATE (video-grouped 5-fold, predict 'gets >=1 content word'):")
    print(f"  AUC real     = {np.mean(aucs):.3f}  (folds {', '.join(f'{v:.3f}' for v in aucs)})")
    print(f"  AUC shuffled = {np.mean(aucs_shuf):.3f}   [0.5 = no signal]")
    print(f"  base rate    = {yb.mean():.3f}")

    (root / a.out).write_text(json.dumps(
        {"n": len(keep), "features": fnames, "corr": report, "null_band": nulls,
         "auc": float(np.mean(aucs)), "auc_shuffled": float(np.mean(aucs_shuf)),
         "base_rate": float(yb.mean())}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
