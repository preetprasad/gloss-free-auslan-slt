"""Classical-CV + keypoint analysis of WHY hands lose sign identity in continuous signing.

Same playbook that produced the project's largest gain: in August, classical CV showed video
quality was uncorrelated with pose confidence, which eliminated "the videos are bad", redirected
effort to the pose estimator, and the DWPose swap then delivered +4.45 BLEU.

The 2026-09-09 continuous-domain probe found hands carry the most sign identity (isolated 40.0%
top-1) and lose the most of it in continuous signing (18.1%). Two competing explanations, with
completely different fixes:

  (A) IMAGE CONDITIONS -- in conversation the hands are smaller, blurrier, more often occluded or
      overlapping. Fix is at the sensor/estimator/crop level and is actionable.
  (B) ARTICULATORY REDUCTION -- continuous signs are genuinely produced with reduced, assimilated
      handshapes. That is a linguistic property; no estimator can recover it, and it is a hard
      ceiling that must be reported rather than engineered around.

This measures both without training anything.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

L_HAND, R_HAND = slice(91, 112), slice(112, 133)


def hand_stats(kps: np.ndarray) -> dict | None:
    """Per-clip hand visibility from DWPose keypoints alone (no RGB needed)."""
    if kps.shape[0] < 2 or kps.shape[1] < 133:
        return None
    out = {}
    conf = kps[..., 2]
    xy = kps[..., :2]
    body = xy[:, :17]
    # signer scale = shoulder distance, so hand size is comparable across framings
    scale = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
    scale = np.where(scale > 1e-3, scale, np.nan)
    sizes, confs, occl = [], [], []
    for hs in (L_HAND, R_HAND):
        h = xy[:, hs]
        confs.append(conf[:, hs].mean(1))
        span = np.linalg.norm(h.max(1) - h.min(1), axis=-1)
        sizes.append(span / scale)                       # hand extent / shoulder width
    lh, rh = xy[:, L_HAND].mean(1), xy[:, R_HAND].mean(1)
    hh = np.linalg.norm(lh - rh, axis=-1) / scale        # hand-hand separation
    face = xy[:, 23:91].mean(1)
    hf = np.minimum(np.linalg.norm(lh - face, axis=-1),
                    np.linalg.norm(rh - face, axis=-1)) / scale
    out["hand_conf"] = float(np.nanmean(np.concatenate(confs)))
    out["hand_conf_p10"] = float(np.nanpercentile(np.concatenate(confs), 10))
    out["hand_size_rel"] = float(np.nanmean(np.concatenate(sizes)))
    out["hands_overlap_frac"] = float(np.nanmean(hh < 0.35))     # hands close => mutual occlusion
    out["hand_on_face_frac"] = float(np.nanmean(hf < 0.35))
    out["hand_speed"] = float(np.nanmean(np.linalg.norm(np.diff(lh, axis=0), axis=-1) / scale[:-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot_csv", required=True)
    ap.add_argument("--cont_dirs", nargs="+", required=True)
    ap.add_argument("--iso_dirs", nargs="+", required=True)
    ap.add_argument("--iso_labels", nargs="+", required=True)
    ap.add_argument("--classical_cv", default="")
    a = ap.parse_args()

    spot = pd.read_csv(a.spot_csv)
    spot["cid"] = spot.Video_Name_Clip.map(lambda v: re.sub(r"_signer\.mp4$", "", str(v)))
    spot["g"] = spot.Query_Gloss.astype(str).str.strip().str.lower()
    idx = {}
    for d in a.cont_dirs:
        for f in glob.glob(os.path.join(d, "*.npz")):
            idx.setdefault(os.path.basename(f)[:-4], f)

    # ---- continuous: stats over the ANNOTATED SPAN only (not the whole clip) ----
    cont = []
    for r in spot.itertuples():
        p = idx.get(r.cid)
        if not p:
            continue
        with np.load(p) as z:
            k = z["KPS"].astype(np.float32)
        b, e = max(0, int(r.Begin_Frame)), min(len(k), max(int(r.Begin_Frame) + 3, int(r.End_Frame)))
        s = hand_stats(k[b:e])
        if s:
            s.update(cid=r.cid, gloss=r.g)
            cont.append(s)
    cont = pd.DataFrame(cont)

    # ---- isolated: whole citation clip ----
    import json
    lab = {}
    for lp in a.iso_labels:
        lab.update(json.load(open(lp)))
    G = set(spot.g)
    iso = []
    for d in a.iso_dirs:
        for f in glob.glob(os.path.join(d, "*.npz")):
            cid = os.path.basename(f)[:-4]
            g = str(lab.get(cid, "")).strip().lower()
            if g not in G:
                continue
            with np.load(f) as z:
                s = hand_stats(z["KPS"].astype(np.float32))
            if s:
                s.update(cid=cid, gloss=g)
                iso.append(s)
    iso = pd.DataFrame(iso)

    cols = ["hand_conf", "hand_conf_p10", "hand_size_rel", "hands_overlap_frac",
            "hand_on_face_frac", "hand_speed"]
    print("=" * 92)
    print("(A) ARE THE HANDS PHYSICALLY HARDER TO SEE IN CONTINUOUS SIGNING?")
    print(f"    isolated n={len(iso)}   continuous spans n={len(cont)}")
    print("=" * 92)
    print(f"{'measure':<22}{'ISOLATED':>12}{'CONTINUOUS':>13}{'ratio':>9}   interpretation")
    interp = {
        "hand_conf": "estimator certainty on hand keypoints",
        "hand_conf_p10": "worst-decile certainty",
        "hand_size_rel": "hand extent / shoulder width",
        "hands_overlap_frac": "frac of frames with hands close (mutual occlusion)",
        "hand_on_face_frac": "frac of frames with a hand near the face",
        "hand_speed": "hand travel per frame / shoulder width",
    }
    for c in cols:
        i, k = iso[c].mean(), cont[c].mean()
        print(f"{c:<22}{i:>12.3f}{k:>13.3f}{(k/i if i else np.nan):>9.2f}x   {interp[c]}")

    if a.classical_cv and os.path.exists(a.classical_cv):
        cv = pd.read_csv(a.classical_cv)
        m = cont.merge(cv, left_on="cid", right_on="clip", how="inner")
        print()
        print("=" * 92)
        print(f"(B) DOES IMAGE QUALITY PREDICT HAND VISIBILITY IN CONTINUOUS CLIPS?  n={len(m)}")
        print("=" * 92)
        cands = [c for c in ["lap_var_mean", "blurry_frame_frac", "flow_mag_mean", "flow_mag_p95",
                             "megapixels", "hand_blob_area_mean", "hands_together_frac",
                             "brightness_mean", "contrast_mean", "framediff_mean"] if c in m.columns]
        print(f"{'image statistic':<24}{'corr w/ hand_conf':>20}{'corr w/ hand_size':>20}")
        for c in cands:
            v = pd.to_numeric(m[c], errors="coerce")
            if v.notna().sum() < 50 or v.std() == 0:
                continue
            print(f"{c:<24}{v.corr(m.hand_conf):>20.3f}{v.corr(m.hand_size_rel):>20.3f}")


if __name__ == "__main__":
    main()
