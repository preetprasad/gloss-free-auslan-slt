"""Score trackers against the ground-truth labels (results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md).

Per labelled panel the truth is the signer's detection index, or NONE (the signer is not among the detections).
A tracker's choice counts as the signer if it is that detection or a near-duplicate of it (detector-box IoU >= 0.6).
Outcomes: HIT (signer chosen), WRONG (another person chosen, including when the signer is not detected),
MISS (masked although the signer was detected), OKMASK (masked and the signer not detected).
Primary metric: wrong-person rate = WRONG / labelled panels. Unclear panels are excluded and counted.
Usage: score_gt.py ALLDET_DIR KEY.json LABELS.json [--part dev|heldout] [--grid]
"""
import argparse, itertools, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import tracker_v2 as v2


def outcome(choice, truth, DB_t):
    if truth is None:
        return "OKMASK" if choice < 0 else "WRONG"
    if choice < 0:
        return "MISS"
    return "HIT" if choice == truth or v2.iou(DB_t[choice], DB_t[truth]) >= 0.6 else "WRONG"


def evaluate(trackers, key, labels, src, part):
    res = {name: Counter() for name in trackers}; cache = {}
    for pid, k in key.items():
        if k["part"] != part:
            continue
        lab = labels.get(pid)
        if lab is None or lab == "unclear":
            for name in trackers: res[name]["UNCLEAR"] += 1
            continue
        truth = None if lab == "none" else k["letters"][lab]
        if k["clip"] not in cache:
            cache[k["clip"]] = dict(np.load(src / f"{k['clip']}.npz"))
        d = cache[k["clip"]]
        for name, fn in trackers.items():
            ck = (name, k["clip"])
            if ck not in cache:
                cache[ck] = fn(d)
            res[name][outcome(int(cache[ck][k["frame"]]), truth, d["DET_BOX"][k["frame"]])] += 1
    return res


def summary(c):
    n = sum(c[x] for x in ("HIT", "WRONG", "MISS", "OKMASK"))
    return (f"n={n} HIT {c['HIT']} ({c['HIT'] / max(n, 1):.1%}) WRONG {c['WRONG']} ({c['WRONG'] / max(n, 1):.1%}) "
            f"MISS {c['MISS']} OKMASK {c['OKMASK']} | unclear {c['UNCLEAR']}")


def base_trackers():
    return {"index0": lambda d: np.where(d["NDET"] > 0, 0, -1),
            "v1": lambda d: d["CHOSEN"].astype(int),
            "v2": lambda d: v2.choose(d)[1].astype(int),
            "v2-tuned": lambda d: v2.choose(d, v2.TUNED)[1].astype(int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("key"); ap.add_argument("labels")
    ap.add_argument("--part", default="dev"); ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    src, key, labels = Path(a.src), json.load(open(a.key)), json.load(open(a.labels))
    for name, c in evaluate(base_trackers(), key, labels, src, a.part).items():
        print(f"[{a.part}] {name:7s} {summary(c)}")
    if a.grid:
        assert a.part == "dev", "weights are tuned on DEV only"
        grid = []
        for wc, wn, wa, wh in itertools.product((0, 0.5, 1, 2), (0, 0.5, 1, 2, 4), (0, 0.5, 1, 2), (0, 0.5, 1, 2, 4)):
            w = dict(w_cov=wc, w_centre=wn, w_area=wa, w_hands=wh)
            c = evaluate({"v2": lambda d, w=w: v2.choose(d, w)[1].astype(int)}, key, labels, src, "dev")["v2"]
            n = sum(c[x] for x in ("HIT", "WRONG", "MISS", "OKMASK"))
            grid.append(((c["WRONG"] / n, -c["HIT"] / n), w, summary(c)))
        grid.sort(key=lambda g: g[0])
        for g in grid[:8]:
            print("[dev grid]", g[1], g[2])


if __name__ == "__main__":
    main()
