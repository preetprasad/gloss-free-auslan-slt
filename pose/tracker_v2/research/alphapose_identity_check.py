"""Is the dataset's SHIPPED AlphaPose track free of the person switches found in our DWPose re-extraction?

Why: the Auslan-Daily paper (Auslan-Daily.txt lines 429-434 and footnote 9, line 471) says the authors tracked people
with AlphaPose and had annotators "manually check and modify the tracking results by assigning correct IDs to the
signers" (tracking errors in < 20% of multi-signer clips). Our DWPose worker kept detection 0 per frame instead.
AlphaPose (full-frame pixels) and DWPose (signer-crop pixels) use different coordinate frames, so a box-IoU identity
check is impossible without the crop offset. This uses a frame-invariant proxy instead:

switch-sized jump = between consecutive frames where both shoulders are confident (> 0.3), the shoulder centre moves, or
the shoulder width changes, by more than 40% of the clip's median shoulder width (the 2026-09-22b definition; body
indices 5/6 are the shoulders in both HALPE-136 and COCO-WholeBody-133).

Gate: on the stored DWPose the jump rate must rise steeply with frac_chosen_nonzero (it detects the known switches).
Reading: if AlphaPose's jump rate stays near its switch-free level in the heavy bin, the shipped track follows one person
where our re-extraction switches. Val + test clips only (the ones held locally in both forms).

Usage: alphapose_identity_check.py
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
AP = ROOT / "full_run/clips"                     # shipped AlphaPose HALPE-136, per clip (KPS (T,136,3) full-frame px)
DW = {"val": R / "val_clips_dwpose", "test": R / "test_clips_dwpose"}   # stored DWPose index-0 (T,133,3)
FRAC = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(open(R / "ndet_per_clip.csv"))}
BINS = (("none", lambda f: f == 0), ("light", lambda f: 0 < f <= 0.1),
        ("medium", lambda f: 0.1 < f <= 0.3), ("heavy", lambda f: f > 0.3))


def jumps(kps, thr=0.3, k=0.4):
    """(jumps, valid consecutive transitions) for a (T, K, 3) pose array."""
    s = kps[:, [5, 6]]; ok = (s[:, :, 2] > thr).all(1)
    ctr = s[:, :, :2].mean(1); w = np.linalg.norm(s[:, 0, :2] - s[:, 1, :2], axis=1)
    if ok.sum() < 2:
        return 0, 0
    ref = np.median(w[ok]); pair = ok[1:] & ok[:-1]
    big = (np.linalg.norm(ctr[1:] - ctr[:-1], axis=1) > k * ref) | (np.abs(w[1:] - w[:-1]) > k * ref)
    return int((big & pair).sum()), int(pair.sum())


def selfcheck():
    t = np.zeros((6, 133, 3)); t[:, 5] = [0, 0, 1]; t[:, 6] = [10, 0, 1]
    assert jumps(t) == (0, 5)
    t2 = t.copy(); t2[3:, 5, 0] += 50; t2[3:, 6, 0] += 50          # teleport at transition 2->3
    assert jumps(t2) == (1, 5)
    t3 = t.copy(); t3[:, 6, 2] = 0                                    # no confident shoulders
    assert jumps(t3) == (0, 0)
    print("SELF-CHECK OK")


def main():
    selfcheck()
    rows = []
    for split, d in DW.items():
        for p in sorted(d.glob("*.npz")):
            c = p.stem
            ap = AP / f"{c}.npz"
            assert ap.exists(), f"missing shipped AlphaPose for {c}"
            a = np.load(ap)["KPS"]; w = np.load(p)["KPS"]
            assert a.shape[1] == 136 and w.shape[1] == 133
            rows.append((split, c, FRAC[c], *jumps(w), *jumps(a)))
    assert len(rows) == 1600, len(rows)
    print(f"{len(rows)} val+test clips; per bin: clips with >=1 switch-sized jump, and jumps per 100 transitions")
    print(f"{'bin':7s} {'n':>4s} | {'DWPose idx0 clips':>18s} {'per100':>7s} | {'AlphaPose shipped clips':>24s} {'per100':>7s}")
    out = {}
    for b, f in BINS:
        r = [x for x in rows if f(x[2])]
        dwc = np.mean([x[3] > 0 for x in r]); dwr = 100 * sum(x[3] for x in r) / max(1, sum(x[4] for x in r))
        apc = np.mean([x[5] > 0 for x in r]); apr = 100 * sum(x[5] for x in r) / max(1, sum(x[6] for x in r))
        out[b] = (len(r), dwc, dwr, apc, apr)
        print(f"{b:7s} {len(r):4d} | {dwc:18.1%} {dwr:7.2f} | {apc:24.1%} {apr:7.2f}")
    # gate: the detector must see the known DWPose switches (heavy >> none)
    assert out["heavy"][2] > 3 * out["none"][2], "jump detector does not track known DWPose switches"
    print("GATE OK: DWPose jump rate rises with the tracker-override fraction "
          f"({out['none'][2]:.2f} -> {out['heavy'][2]:.2f} per 100 transitions, none -> heavy)")


if __name__ == "__main__":
    main()
