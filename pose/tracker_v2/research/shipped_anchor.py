"""Does the dataset's own signer annotation already solve signer selection?

The Auslan-Daily paper (local text ./Auslan-Daily.txt, lines 213-217) says the authors ran AlphaPose
tracking on each clip, "manually modify the error of the Alphapose results and labelled the signer". The shipped
per-clip pose (full_run/clips/<clip>.npz: KPS (T,136,3) HALPE-136 in full-video pixels, BBOX (T,4) xywh, zeros where
no signer) is that labelled signer. DWPose ran on the signer-crop video, so its pixels differ from the shipped ones by
the crop offset. This script:
  1. estimates the per-clip offset from the body keypoints (COCO-17 order in both layouts) and checks the scale is 1;
  2. "anchor" tracker: per frame, the DWPose detection whose body skeleton matches the shipped signer after the offset
     (median keypoint distance < TOL shoulder widths), else masked; shipped-absent frames are masked;
  3. scores it with the GT scorer's outcome() on DEV (dev) or describes it on held-out (describe, after V21_FROZEN).
No parameter here is chosen on held-out: TOL is fixed below before any scoring.

Usage: shipped_anchor.py dev|describe
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residuals as R  # noqa: E402

SHIP = Path("./full_run/clips")
BODY = np.arange(17)
TOL = 0.25            # max median body-keypoint distance, in shoulder widths, for "same skeleton"
CONF = 0.3


def body_match(ship_xy, ship_sc, det_xy, det_sc):
    """Offset (det - ship) and per-keypoint differences over body keypoints confident in both."""
    m = (ship_sc[BODY] > CONF) & (det_sc[BODY] > CONF)
    if m.sum() < 4:
        return None
    return det_xy[BODY][m] - ship_xy[BODY][m]


def shoulder(xy, sc):
    return float(np.linalg.norm(xy[5] - xy[6])) if sc[5] > CONF and sc[6] > CONF else np.nan


def fit(P, Q):
    """Least-squares scale + translation (no rotation) mapping P -> Q."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    sc = float((Pc * Qc).sum() / max((Pc ** 2).sum(), 1e-9))
    return sc, Q.mean(0) - sc * P.mean(0)


def anchor(clip, d):
    """Returns (chosen (T,), info). chosen[t] = DWPose detection matching the shipped signer, -1 if none/absent.
    The crop may be both shifted and resized, so a per-clip scale + translation is estimated by consensus: every
    (frame, detection) pair proposes one; the proposal under which the most frames have a matching detection wins."""
    s = np.load(SHIP / f"{clip}.npz"); K, B = s["KPS"], s["BBOX"]
    T = len(d["NDET"])
    if len(K) != T:          # frame correspondence unknown: excluded and reported, never guessed
        return None, dict(shipped_T=len(K), T=T)
    present = (B[:, 2] > 1e-3) & (B[:, 3] > 1e-3)
    KA, SA, ND = d["KP_ALL"], d["SC_ALL"], d["NDET"]
    chosen = np.full(T, -1, int)
    pairs = []            # (t, i, P ship pts, Q det pts, det shoulder width)
    for t in np.flatnonzero(present):
        for i in range(int(ND[t])):
            m = (K[t, BODY, 2] > CONF) & (SA[t, i, BODY] > CONF)
            sw = shoulder(KA[t, i], SA[t, i])
            if m.sum() >= 4 and np.isfinite(sw) and sw > 0:
                pairs.append((t, i, K[t, BODY, :2][m], KA[t, i, BODY][m], sw))
    if not pairs:
        return chosen, dict(present=int(present.sum()), scale=np.nan, matched=0)
    def err(sc, tr, P, Q, sw):
        return float(np.median(np.linalg.norm(sc * P + tr - Q, axis=1))) / sw
    best = None
    for (_, _, P, Q, _) in pairs[:: max(1, len(pairs) // 300)]:          # ponytail: subsample proposals, O(n^2)
        sc, tr = fit(P, Q)
        if sc <= 0:
            continue
        ok = {t for (t, i, P2, Q2, sw) in pairs if err(sc, tr, P2, Q2, sw) < TOL}
        if best is None or len(ok) > best[0]:
            best = (len(ok), sc, tr)
    _, sc, tr = best
    for t in np.flatnonzero(present):
        cand = [(err(sc, tr, P, Q, sw), i) for (tt, i, P, Q, sw) in pairs if tt == t]
        cand = [c for c in cand if c[0] < TOL]
        if cand:
            chosen[t] = min(cand)[1]
    return chosen, dict(present=int(present.sum()), scale=sc, offset=tr, matched=int((chosen >= 0).sum()))


def main(mode):
    if mode == "describe":
        assert R.FROZEN.exists()
    part = "dev" if mode == "dev" else "heldout"
    data, clips, panels = R.load(part)
    lines = [f"## {part.upper()}: anchor to the shipped (manually corrected) AlphaPose signer track, TOL {TOL} SW"]
    tot, cnt_a, cnt_v, both = Counter(), Counter(), Counter(), Counter()
    for c, d in data.items():
        ch, inf = anchor(c, d); v2c = R.trace(d)["chosen"]
        if ch is None:
            lines.append(f"  {c}: EXCLUDED, shipped T {inf['shipped_T']} != DWPose T {inf['T']} ({len(panels[c])} panels)")
            tot["clips excluded"] += 1; tot["panels excluded"] += len(panels[c]); continue
        T = len(ch); tot["frames"] += T; tot["shipped present"] += inf["present"]; tot["anchored"] += inf["matched"]
        tot["anchor == v2 (both on a detection)"] += int(((ch >= 0) & (v2c >= 0) & (ch == v2c)).sum())
        tot["both on a detection"] += int(((ch >= 0) & (v2c >= 0)).sum())
        for f, tr in panels[c]:
            oa, ov = R.outcome(ch[f], tr, d["DET_BOX"][f]), R.outcome(v2c[f], tr, d["DET_BOX"][f])
            cnt_a[oa] += 1; cnt_v[ov] += 1; both[(ov, oa)] += 1
        lines.append(f"  {c}: T {T} shipped-present {inf['present']} anchored {inf['matched']} scale "
                     f"{inf['scale']:.3f} offset {np.round(inf.get('offset'), 1) if inf.get('offset') is not None else None}")
    lines.append(f"totals: {dict(tot)}")
    lines.append(f"labelled panels, anchor : HIT/WRONG/MISS/OKMASK {cnt_a['HIT']}/{cnt_a['WRONG']}/{cnt_a['MISS']}/{cnt_a['OKMASK']}")
    lines.append(f"labelled panels, v2-tuned: HIT/WRONG/MISS/OKMASK {cnt_v['HIT']}/{cnt_v['WRONG']}/{cnt_v['MISS']}/{cnt_v['OKMASK']}")
    lines.append(f"(v2 outcome, anchor outcome) where they differ: "
                 f"{ {f'{a}->{b}': n for (a, b), n in both.items() if a != b} }")
    print("\n".join(lines))
    (Path(__file__).parent / f"shipped_anchor_{mode}_output.txt").write_text("\n".join(lines) + "\n")


def selftest():
    rng = np.random.default_rng(0)
    ship = rng.uniform(100, 400, (133, 2)); sc = np.full(133, 0.9)
    diff = body_match(ship, sc, ship + [50, -20], sc)
    assert np.allclose(np.median(diff, 0), [50, -20])
    assert body_match(ship, np.zeros(133), ship, sc) is None
    k, t = fit(ship[:17], 0.5 * ship[:17] + [30, 40])
    assert abs(k - 0.5) < 1e-9 and np.allclose(t, [30, 40])
    print("shipped_anchor self-test OK")


def diag(clip, lag_range=range(-6, 7)):
    """One DEV clip: per-detection own-fit (scale, translation, residual in shoulder widths) against the shipped signer
    at the same frame and at temporal lags, to tell a moving crop (translation drifts, residual small) from a temporal
    misalignment (residual small only at a nonzero lag)."""
    data, clips, panels = R.load("dev"); assert clips[clip] == "dev"; d = data[clip]
    s = np.load(SHIP / f"{clip}.npz"); K = s["KPS"]; KA, SA, ND = d["KP_ALL"], d["SC_ALL"], d["NDET"]
    truth = dict(panels[clip])
    for lag in lag_range:
        res = []
        for t in range(len(K)):
            u = t + lag
            if not 0 <= u < len(ND) or truth.get(u, "x") in (None, "x"):
                continue
            i = truth[u]; m = (K[t, BODY, 2] > CONF) & (SA[u, i, BODY] > CONF); sw = shoulder(KA[u, i], SA[u, i])
            if m.sum() >= 4 and np.isfinite(sw):
                sc, tr = fit(K[t, BODY, :2][m], KA[u, i, BODY][m])
                res.append((u, round(sc, 2), np.round(tr).tolist(),
                            round(float(np.median(np.linalg.norm(sc * K[t, BODY, :2][m] + tr - KA[u, i, BODY][m], axis=1))) / sw, 3)))
        if res:
            print(f"lag {lag:+d}: median residual {np.median([r[3] for r in res]):.3f} SW | " +
                  "; ".join(f"f{u} s{sc} t{tr} r{r}" for u, sc, tr, r in res[:6]))


if __name__ == "__main__":
    selftest()
    diag(sys.argv[2]) if sys.argv[1] == "diag" else main(sys.argv[1])
