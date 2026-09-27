"""Robustness checks on signing_activity.py's detector (2026-09-26). Reads features_oof.csv; refits only in (c).

  (a) reproduction gate: refitting the 'full' model on features_oof.csv reproduces its stored out-of-fold scores
  (b) identity confound: AUROC of the same out-of-fold score, signers split by Signer_ID (1 / 2 / other) vs all
      non-signers. Signers 1 and 2 hold 92% of clips and appear in every video, so grouping by video cannot keep them
      out of the training folds; if the detector keyed on who they are, "other" signers would score near chance.
  (d) N-way top-1 analogue of Auslan-Daily Active Signer Detection (Shen et al. 2023, Table 5): per clip, is the
      signer the highest-scoring person among {signer + that clip's non-signer crops}? Same for the confidence gate
      (lowest ce wins). Chance = mean of 1/N.
  (c) sensitivity: put back the 42 crops the gate's B3 test flagged as the signer's own skeleton (all 1,678).

    python signing_activity_checks.py selftest
    python signing_activity_checks.py run      # writes checks.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import signing_activity as sa  # noqa: E402

COMM = sa.ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/" \
                 "Auslan-Daily_Communication.xlsx"


def top1(df, score):
    """df rows: parent, y; score higher = more signer-like. Returns per-clip (hit, 1/N, video) for clips with >=1 neg."""
    out = []
    for par, g in df.assign(s=score).groupby("parent"):
        if g.y.sum() != 1 or len(g) < 2:
            continue
        s_sig, s_neg = g.s[g.y == 1].iloc[0], g.s[g.y == 0].to_numpy()
        out.append((float(s_sig > s_neg.max()), 1 / len(g), g.video.iloc[0]))
    return out


def run():
    P = pd.read_csv(HERE / "features_oof.csv")
    y, grp = P.y.to_numpy(), P.video.to_numpy()
    X = P[sa.FULL].to_numpy(float)
    re = sa.oof(X, y, grp, 0)
    assert np.allclose(re, P.oof_full, atol=1e-8), "stored out-of-fold scores do not reproduce"
    out = {"a_reproduced_oof_rows": int(len(P))}

    # (b) Signer_ID strata
    sid = pd.read_excel(COMM).set_index("Video_Clip_Name").Signer_ID
    P["sid"] = P.cid.map(sid).where(P.y == 1)
    assert P.sid[P.y == 1].notna().all()
    neg = P[P.y == 0]
    out["b_by_signer_id"] = {}
    for name, m in (("id1", P.sid == 1), ("id2", P.sid == 2), ("other", (P.y == 1) & ~P.sid.isin([1, 2]))):
        pos = P[m]
        sub = pd.concat([pos, neg])
        out["b_by_signer_id"][name] = {"n_signers": int(len(pos)),
                                       "auroc_full": sa.boot_auc(sub.oof_full.to_numpy(), sub.y.to_numpy(),
                                                                 sub.video.to_numpy()),
                                       "auroc_conf_gate": sa.boot_auc(-sub.ce.to_numpy(), sub.y.to_numpy(),
                                                                      sub.video.to_numpy())}
        print("signer", name, out["b_by_signer_id"][name])

    # (d) N-way top-1 per clip
    out["d_topN"] = {}
    for name, s in (("detector_full", P.oof_full.to_numpy()), ("conf_gate", -P.ce.to_numpy())):
        t = top1(P, s)
        out["d_topN"][name] = {"n_clips": len(t), "top1": sa.boot_mean([h for h, _, _ in t], [v for *_, v in t]),
                               "chance": float(np.mean([c for _, c, _ in t]))}
        print("top1", name, out["d_topN"][name])

    # (c) all 1,678: add the B3-flagged crops back
    df = sa.load()
    back = df[(df.y == 0) & ~df.cid.isin(P.cid)].reset_index(drop=True)
    assert len(back) == 42 and len(P) + len(back) == 3278
    F = pd.DataFrame([sa.features(np.load(p)["KPS"]) for p in back.path])
    A = pd.concat([P[["cid", "y", "video"] + sa.FULL], pd.concat([back[["cid", "y", "video"]], F], axis=1)],
                  ignore_index=True)
    ya, ga, Xa = A.y.to_numpy(), A.video.to_numpy(), A[sa.FULL].to_numpy(float)
    sa_ = sa.oof(Xa, ya, ga, 0)
    acc = sa.nested_accept(Xa, ya, ga, 0)
    out["c_all_1678"] = {"n_non_signers": int((ya == 0).sum()), "auroc_full": sa.boot_auc(sa_, ya, ga),
                         "op95_signer_recall": sa.boot_mean(acc[ya == 1], ga[ya == 1]),
                         "op95_nonsigner_rejected": sa.boot_mean(~acc[ya == 0], ga[ya == 0]),
                         "flagged_crops_accepted_as_signer_op95": float(acc[len(P):].mean())}
    print("all 1678", out["c_all_1678"])
    json.dump(out, open(HERE / "checks.json", "w"), indent=1, default=float)


def selftest():
    d = pd.DataFrame({"parent": ["a", "a", "a", "b", "b", "c"], "y": [1, 0, 0, 1, 0, 1],
                      "video": ["v1", "v1", "v1", "v2", "v2", "v3"]})
    t = top1(d, np.array([3.0, 1.0, 2.0, 0.0, 5.0, 9.0]))       # clip c has no negative -> skipped
    assert [h for h, _, _ in t] == [1.0, 0.0] and [round(c, 3) for _, c, _ in t] == [0.333, 0.5]
    t = top1(d, np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0]))       # a tie is a miss
    assert [h for h, _, _ in t] == [0.0, 1.0]
    print("selftest passed")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
