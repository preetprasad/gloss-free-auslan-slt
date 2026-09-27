"""Are v2's rejected boundary joins camera cuts? Scene change from the local signer-crop video (no download):
mean absolute difference of consecutive 96x96 greyscale frames (the 2026-09-22 switch-cause measure, where a
difference >= 18 marked a camera cut). Reports the value at every boundary link of the join tables and every cut in
the 60 GT clips, then (DEV only) a per-shot v2: split the clip at cuts and let v2 choose a signer in each shot.

Usage: cuts.py dev|describe     (describe = held-out description; refuses to run before V21_FROZEN.json exists)
"""
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residuals as R  # noqa: E402

VID = Path("./results/dwpose_experiment/rgb_clips/Signer")
CUT = 18.0
CLEAR = 40.0     # an unambiguous cut: >= 7x the median transition (4-6) in both parts


def frame_diffs(clip, T):
    cap = cv2.VideoCapture(str(VID / f"{clip}_signer.mp4")); g = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA).astype(float))
    assert len(g) == T, (clip, len(g), T)       # pose frame t == video frame t
    return np.array([np.abs(g[t + 1] - g[t]).mean() for t in range(T - 1)])   # diff[t]: frame t -> t+1


def per_shot(d, cuts):
    """v2 run separately on each shot (cuts = frame indices t where t -> t+1 is a cut)."""
    T = len(d["NDET"]); edges = [0] + [t + 1 for t in cuts] + [T]; out = np.full(T, -1, int)
    for a, b in zip(edges[:-1], edges[1:]):
        seg = {k: d[k][a:b] for k in ("KP_ALL", "SC_ALL", "DET_BOX", "NDET")}; seg["WH"] = d["WH"]
        out[a:b] = R.v2.choose(seg, R.v2.TUNED)[1]
    return out


def main(mode):
    if mode == "describe":
        assert R.FROZEN.exists()
    part = "dev" if mode == "dev" else "heldout"
    data, clips, panels = R.load(part)
    diffs = {c: frame_diffs(c, len(d["NDET"])) for c, d in data.items()}
    cuts = {c: [int(t) for t in np.flatnonzero(x >= CUT)] for c, x in diffs.items()}
    lines = [f"## {part.upper()}: camera cuts (grey-frame difference >= {CUT})",
             f"clips with >= 1 cut: {sum(bool(v) for v in cuts.values())} / {len(cuts)}; "
             f"median frame difference over all transitions {np.median(np.concatenate(list(diffs.values()))):.2f}"]
    for c in sorted(cuts):
        if cuts[c]:
            lines.append(f"  {c}: cuts after frames {cuts[c]} (diff {[round(float(diffs[c][t]), 1) for t in cuts[c]]})")
    lines.append("boundary links v2 tested (all tracklets, labelled or not): max frame difference across the link")
    tally = Counter()
    for c, d in data.items():
        info = R.trace(d)
        for k, dc in info["dec"].items():
            if dc["kind"] != "boundary" or not dc["link"]:
                continue
            a, b, v = dc["link"]; lo, hi = min(a, b), max(a, b)
            m = float(diffs[c][lo:hi].max()); cut = m >= CUT
            tally[(dc["joined"], cut)] += 1
            lines.append(f"  {c} f{a}->f{b} IoU {v:.2f} joined={dc['joined']} len {len(info['tr'][k])} "
                         f"max diff {m:.1f}{'  CUT' if cut else ''}")
    lines.append(f"tally (joined, across a cut): {dict(tally)}")
    lines.append(f"\nper-shot view of clips with a clear cut (diff >= {CLEAR}): the person v2 would pick in each shot, "
                 "their hand-motion feature (v2.features 'hands'), and what the labels say that person is")
    for c, d in data.items():
        cc = [int(t) for t in np.flatnonzero(diffs[c] >= CLEAR)]
        if not cc:
            continue
        T = len(d["NDET"]); edges = [0] + [t + 1 for t in cc] + [T]; W, H = (int(x) for x in d["WH"])
        for a, b in zip(edges[:-1], edges[1:]):
            seg = {k: d[k][a:b] for k in ("KP_ALL", "SC_ALL", "DET_BOX", "NDET")}; seg["WH"] = d["WH"]
            info = R.trace(seg)
            if info["best"] is None:
                lines.append(f"  {c} shot f{a}-{b - 1}: no usable detection"); continue
            f_ = R.v2.features(info["tr"][info["best"]], seg["KP_ALL"], seg["SC_ALL"], seg["DET_BOX"], W, H, b - a)
            lab = Counter()
            for f, tr in panels[c]:
                if a <= f < b and info["chosen"][f - a] >= 0:
                    lab[R.outcome(info["chosen"][f - a], tr, d["DET_BOX"][f])] += 1
            lines.append(f"  {c} shot f{a}-{b - 1} ({b - a} fr): hands {f_['hands']:.3f} area {f_['area']:.3f} "
                         f"cov {f_['cov']:.2f} | labelled picks {dict(lab)}")
    if mode == "dev":
        lines.append("\n## DEV only: per-shot v2 (split at cuts) vs v2-tuned, labelled panels HIT/WRONG/MISS/OKMASK")
        for name, fn in (("v2-tuned", lambda c, d: R.trace(d)["chosen"]), ("per-shot v2", lambda c, d: per_shot(d, cuts[c]))):
            cnt, masked = Counter(), 0
            for c, d in data.items():
                ch = fn(c, d); masked += int((ch < 0).sum())
                for f, tr in panels[c]:
                    cnt[R.outcome(ch[f], tr, d["DET_BOX"][f])] += 1
            lines.append(f"  {name:12s} {cnt['HIT']}/{cnt['WRONG']}/{cnt['MISS']}/{cnt['OKMASK']} | masked frames {masked}")
    print("\n".join(lines))
    (Path(__file__).parent / f"cuts_{mode}_output.txt").write_text("\n".join(lines) + "\n")


def selftest():
    d = dict(NDET=np.ones(6, int), WH=np.array([100, 100]), KP_ALL=np.zeros((6, 1, 133, 2), np.float32),
             SC_ALL=np.full((6, 1, 133), 0.9, np.float32), DET_BOX=np.tile(np.array([10, 10, 60, 90], np.float32), (6, 1, 1)))
    d["DET_BOX"][3:] = [[40, 5, 95, 95]]            # the person jumps at a cut after frame 2
    assert list(R.v2.choose(d, R.v2.TUNED)[1]).count(-1) == 3      # whole-clip v2 keeps one side
    assert list(per_shot(d, [2])) == [0] * 6                        # per-shot keeps both
    print("cuts self-test OK")


if __name__ == "__main__":
    selftest()
    main(sys.argv[1])
