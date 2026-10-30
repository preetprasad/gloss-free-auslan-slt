"""EXPLORATORY feasibility probe (not pre-registered): do DWPose face keypoints carry Auslan question marking?

Auslan, like other sign languages, marks yes/no questions with raised brows and wh-questions with furrowed brows. If the
face keypoints the translator already receives carry that signal, sentence-type intonation for the speech stage can be
predicted from the signer's face rather than from the (often wrong) translated text. This probe measures the effect on
TRAIN clips only, so validation and test stay clean for a later pre-registered confirmatory test.

Labels come from the English reference (Communication references carry no '?', so the type is read from word order;
rules are conservative and anything ambiguous is dropped). Features are signer-centred to remove face shape.
Usage: python tools/nmm_question_probe.py [selftest]
"""
import json, random, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
POSE = RES / "nmm_probe" / "tracked_train"
CONF = 0.3
WH = {"what", "where", "who", "whom", "whose", "when", "why", "how", "which"}
AUX = {"do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "should", "have", "has",
       "had", "may", "might", "shall", "am"}
SUBJ = {"i", "you", "we", "they", "he", "she", "it", "that", "this", "there", "these", "those", "not", "possum",
        "sally", "anyone", "anybody", "everyone", "somebody", "someone"}
HOW_Q = {"many", "much", "long", "old", "far", "often", "about"}
# COCO-WholeBody: face = keypoints 23..90 in 300W/dlib order
BROW_R, BROW_L = list(range(23 + 17, 23 + 22)), list(range(23 + 22, 23 + 27))
EYE_R, EYE_L = list(range(23 + 36, 23 + 42)), list(range(23 + 42, 23 + 48))
INNER_R, INNER_L, NOSE_TIP = 23 + 21, 23 + 22, 23 + 30


def label(text):
    w = text.lower().split()
    w = [x for x in w if x not in {".", ","}]
    if not w:
        return None
    if w[0] in WH:
        if len(w) == 1 or (len(w) > 1 and (w[1] in AUX | {"not", w[0]} or (w[0] == "how" and w[1] in HOW_Q)
                                           or (w[0] == "what" and w[1] == "about"))):
            return "wh"
        return None                                           # "what a mess", "what we do is ..." -> ambiguous
    if w[0] in AUX:
        return "yn" if len(w) > 1 and w[1] in SUBJ else None  # "have a look" -> dropped, not called a question
    return "stmt"


def frame_feats(kps):
    """Per frame: brow height and inner-brow gap (both / interocular), head-pitch proxy; NaN where not confident."""
    need = BROW_R + BROW_L + EYE_R + EYE_L + [INNER_R, INNER_L, NOSE_TIP]
    ok = (kps[:, need, 2] >= CONF).all(1)
    er, el = kps[:, EYE_R, :2].mean(1), kps[:, EYE_L, :2].mean(1)
    io = np.linalg.norm(er - el, axis=1)
    ok &= io > 1.0
    io = np.where(ok, io, np.nan)
    brow = ((er[:, 1] - kps[:, BROW_R, 1].mean(1)) + (el[:, 1] - kps[:, BROW_L, 1].mean(1))) / 2 / io
    inner = np.linalg.norm(kps[:, INNER_R, :2] - kps[:, INNER_L, :2], axis=1) / io
    pitch = (kps[:, NOSE_TIP, 1] - (er[:, 1] + el[:, 1]) / 2) / io
    return brow, inner, pitch, ok


def clip_feats(kps):
    brow, inner, pitch, ok = frame_feats(kps)
    if ok.sum() < 5:
        return None
    b, i, p = brow[ok], inner[ok], pitch[ok]
    return {"brow_mean": float(b.mean()), "brow_p90": float(np.percentile(b, 90)), "inner_mean": float(i.mean()),
            "inner_p10": float(np.percentile(i, 10)), "pitch_mean": float(p.mean()), "n_ok": int(ok.sum()),
            "brow_above_eye": float((b > 0).mean())}


def auroc(pos, neg):
    s = np.r_[pos, neg]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    tr = pd.read_excel(ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
    signer, video = dict(zip(tr.Video_Clip_Name, tr.Signer_ID)), dict(zip(tr.Video_Clip_Name, tr.Video_Name))
    rows = [l.rstrip("\n").split("\t") for l in open(RES / "_kaya_train_manifest.tsv") if "\t" in l]
    lab = {Path(p).stem: label(t) for p, t in rows}
    text = {Path(p).stem: t for p, t in rows}
    print("labels (train):", Counter(lab.values()))
    rng = random.Random(0)
    for k in ("yn", "wh"):
        print(f"sample {k}:", rng.sample([text[c] for c in lab if lab[c] == k], 12))
    feats = {}
    for c, l in lab.items():
        if l is None or not (POSE / f"{c}.npz").exists():
            continue
        f = clip_feats(np.load(POSE / f"{c}.npz")["KPS"])
        if f:
            feats[c] = f
    df = pd.DataFrame.from_dict(feats, orient="index")
    df["label"], df["signer"], df["video"] = [lab[c] for c in df.index], [signer[c] for c in df.index], [video[c] for c in df.index]
    print(f"clips with >=5 confident face frames: {len(df)} of {sum(l is not None for l in lab.values())} labelled; "
          f"brows above eyes in {100 * df.brow_above_eye.mean():.1f}% of frames (index sanity check)")
    assert df.brow_above_eye.mean() > 0.95, "face keypoint indices look wrong"
    big = df.signer.map(df.signer.value_counts()) >= 20
    df = df[big].copy()
    cols = ["brow_mean", "brow_p90", "inner_mean", "inner_p10", "pitch_mean"]
    for col in cols:                                          # signer-centred: removes face shape and camera distance
        g = df.groupby("signer")[col]
        df[col + "_z"] = (df[col] - g.transform("mean")) / g.transform("std")
    out = {"n": Counter(df.label), "signers": int(df.signer.nunique())}
    for col in ("brow_mean_z", "inner_mean_z"):              # pitch control: label-blind OLS residual on head pitch
        b = np.polyfit(df.pitch_mean_z, df[col], 1)
        df[col + "_resid_pitch"] = df[col] - np.polyval(b, df.pitch_mean_z)
        out[f"{col}_on_pitch_slope"] = float(b[0])
    tests = [("yn", "brow_mean_z", +1), ("yn", "brow_p90_z", +1), ("wh", "inner_p10_z", -1), ("wh", "inner_mean_z", -1),
             ("wh", "brow_mean_z", -1), ("yn", "pitch_mean_z", +1), ("wh", "pitch_mean_z", +1),
             ("yn", "brow_mean_z_resid_pitch", +1), ("wh", "brow_mean_z_resid_pitch", -1),
             ("wh", "inner_mean_z_resid_pitch", -1)]
    nrng = np.random.default_rng(0)
    for cls, col, sign in tests:
        sub = df[df.label.isin([cls, "stmt"])]
        y, x = (sub.label == cls).to_numpy(), sign * sub[col].to_numpy()
        obs = auroc(x[y], x[~y])
        # null: shuffle labels within signer (keeps each signer's question rate)
        null = []
        groups = [np.flatnonzero(sub.signer.to_numpy() == s) for s in sub.signer.unique()]
        for _ in range(2000):
            yy = y.copy()
            for g in groups:
                yy[g] = nrng.permutation(yy[g])
            null.append(auroc(x[yy], x[~yy]))
        p = float(np.mean(np.abs(np.array(null) - 0.5) >= abs(obs - 0.5)))
        # video-cluster bootstrap CI
        vids = sub.video.unique()
        bv = []
        for _ in range(2000):
            pick = nrng.choice(vids, len(vids))
            idx = np.concatenate([np.flatnonzero(sub.video.to_numpy() == v) for v in pick])
            if y[idx].any() and (~y[idx]).any():
                bv.append(auroc(x[idx][y[idx]], x[idx][~y[idx]]))
        out[f"{cls}:{'+' if sign > 0 else '-'}{col}"] = {"auroc": obs, "ci95": [float(np.percentile(bv, 2.5)),
                                                         float(np.percentile(bv, 97.5))], "perm_p": p,
                                                         "n_pos": int(y.sum()), "n_neg": int((~y).sum())}
        print(f"{cls:>3} vs stmt  {'+' if sign > 0 else '-'}{col:14s} AUROC {obs:.3f}  95% [{out[f'{cls}:{chr(43) if sign > 0 else chr(45)}{col}']['ci95'][0]:.3f}, "
              f"{out[f'{cls}:{chr(43) if sign > 0 else chr(45)}{col}']['ci95'][1]:.3f}]  perm p {p:.4f}  (n {int(y.sum())} vs {int((~y).sum())})")
    # does the translator keep the sentence type? (reads existing test decodes; descriptive, nothing is tuned)
    for name, path, key in (("S6", RES / "final_ensemble_test_decodes/hyps_LP=0.3.jsonl", "hyp_beam"),):
        rs = [json.loads(l) for l in open(path)]
        pairs = [(label(r["ref"]), label(r[key])) for r in rs]
        kept = {cls: (sum(1 for a, b in pairs if a == cls and b == cls), sum(1 for a, _ in pairs if a == cls)) for cls in ("yn", "wh")}
        false_q = sum(1 for a, b in pairs if a == "stmt" and b in ("yn", "wh")), sum(1 for a, _ in pairs if a == "stmt")
        out[f"translator_{name}"] = {"type_kept": kept, "stmt_called_question": false_q}
        print(f"translator {name}: reference questions whose output is the same type {kept}; statements output as a "
              f"question {false_q[0]}/{false_q[1]}")
    (RES / "nmm_probe").mkdir(exist_ok=True)
    json.dump(out, open(RES / "nmm_probe" / "nmm_question_probe.json", "w"), indent=1, default=str)


def selftest():
    assert label("what is it .") == "wh" and label("where is possum .") == "wh" and label("what what .") == "wh"
    assert label("how many are there .") == "wh" and label("what about you .") == "wh"
    assert label("what a mess .") is None and label("what we do is we get paper .") is None
    assert label("would you like to sit down now .") == "yn" and label("did not that sound a bit odd .") == "yn"
    assert label("have a look .") is None and label("i am happy .") == "stmt"
    k = np.zeros((6, 133, 3), np.float32)
    k[..., 2] = 1.0
    for i in EYE_R: k[:, i, :2] = (40, 50)
    for i in EYE_L: k[:, i, :2] = (60, 50)
    for i in BROW_R + BROW_L: k[:, i, 1] = 44                 # 6 px above the eyes, interocular 20 -> 0.3
    k[:, INNER_R, 0], k[:, INNER_L, 0], k[:, NOSE_TIP, 1] = 46, 54, 62
    f = clip_feats(k)
    assert abs(f["brow_mean"] - 0.3) < 1e-6 and abs(f["inner_mean"] - 0.4) < 1e-6 and abs(f["pitch_mean"] - 0.6) < 1e-6
    k[:3, BROW_R + BROW_L, 2] = 0.1                          # low confidence -> dropped, 3 frames left -> None
    assert clip_feats(k) is None
    assert auroc([2, 3], [0, 1]) == 1.0 and auroc([0], [1]) == 0.0
    print("selftest passed")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
