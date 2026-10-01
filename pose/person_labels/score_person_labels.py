"""Score the roster + activity rule against the two-rater ground truth (results/person_labels/PREREG_PERSON_LABELS_2026-09-30.md).

Compares, on the labelled panels of results/tracker_v2/, index-0, the shipped online tracker (v1), tracker v2 (box area),
and the JavaScript roster in `online` and `activity` modes (results/person_labels/eval_roster.mjs output). Outcomes and the
duplicate-detection rule (box IoU >= 0.6) are score_gt.py's own. Where the roster makes no choice (fewer than two people
counted, or nobody eligible) the engine keeps the shipped single-signer stream, so the panel is scored as v1.

    python score_person_labels.py ALLDET_DIR ROSTER.json --part dev
    python score_person_labels.py ALLDET_DIR ROSTER.json --part heldout     # once: writes heldout_scored.lock, then refuses
    python score_person_labels.py --selftest
"""
import argparse, json, sys, tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TV2 = HERE.parents[0] / "tracker_v2"
sys.path.insert(0, str(TV2))
import score_gt as sg          # noqa: E402  (outcome, summary, base_trackers)

MODES = ("online", "activity")


def roster_choice(mode, roster, clip, d):
    r = roster[clip].get(mode)
    return np.array(r, int) if r is not None else d["CHOSEN"].astype(int)


def trackers_for(roster):
    base = sg.base_trackers()
    out = {"index0": lambda clip, d: base["index0"](d), "v1": lambda clip, d: base["v1"](d),
           "v2-tuned": lambda clip, d: base["v2-tuned"](d)}
    for m in MODES:
        out[f"roster-{m}"] = lambda clip, d, m=m: roster_choice(m, roster, clip, d)
    return out


def evaluate(trackers, key, labels, src, part):
    res = {n: Counter() for n in trackers}
    per_panel = {n: {} for n in trackers}
    cache = {}
    for pid, k in key.items():
        if k["part"] != part:
            continue
        lab = labels.get(pid)
        if lab is None or lab == "unclear":
            for n in trackers:
                res[n]["UNCLEAR"] += 1
            continue
        truth = None if lab == "none" else k["letters"][lab]
        clip = k["clip"]
        if clip not in cache:
            cache[clip] = dict(np.load(Path(src) / f"{clip}.npz"))
        d = cache[clip]
        for name, fn in trackers.items():
            if (name, clip) not in cache:
                cache[(name, clip)] = fn(clip, d)
            o = sg.outcome(int(cache[(name, clip)][k["frame"]]), truth, d["DET_BOX"][k["frame"]])
            res[name][o] += 1
            per_panel[name][pid] = o
    return res, per_panel


def rates(c):
    n = sum(c[x] for x in ("HIT", "WRONG", "MISS", "OKMASK"))
    return c["HIT"] / max(n, 1) * 100, c["WRONG"] / max(n, 1) * 100, n


def rule(res):
    """The pre-registered inequalities for one part: WRONG(activity) <= WRONG(v1) + 2.0 and HIT(activity) >= HIT(v1) - 5.0."""
    h_a, w_a, _ = rates(res["roster-activity"]); h_v, w_v, _ = rates(res["v1"])
    return w_a <= w_v + 2.0 and h_a >= h_v - 5.0, (w_a - w_v, h_a - h_v)


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); T = 4
        db = np.zeros((T, 2, 4), "f4"); db[:, 0] = [0, 0, 100, 100]; db[:, 1] = [200, 0, 300, 100]
        np.savez(tmp / "c.npz", DET_BOX=db, CHOSEN=np.array([0, 0, 1, 1], "i2"), NDET=np.full(T, 2, "i2"),
                 KP_ALL=np.zeros((T, 2, 133, 2), "f4"), SC_ALL=np.zeros((T, 2, 133), "f4"), KPS=np.zeros((T, 133, 3), "f4"),
                 BBOX=np.zeros((T, 4), "f4"), WH=np.array([640, 480]))
        key = {f"g{t}": {"part": "dev", "clip": "c", "frame": t, "letters": {"A": 1, "B": 0}} for t in range(T)}
        labels = {f"g{t}": "A" for t in range(T)}                       # the signer is detection 1 on every frame
        roster = {"c": {"online": None, "activity": [1, 1, 1, -1]}}     # activity: right, right, right, masked (signer missed)
        res, _ = evaluate(trackers_for(roster), key, labels, tmp, "dev")
        assert res["v1"]["HIT"] == 2 and res["v1"]["WRONG"] == 2, res["v1"]
        assert res["roster-online"] == res["v1"], "no choice -> scored as v1"
        assert res["roster-activity"]["HIT"] == 3 and res["roster-activity"]["MISS"] == 1, res["roster-activity"]
        ok, (dw, dh) = rule(res)
        assert ok and dw < 0 and dh > 0
        bad = {"c": {"online": None, "activity": [0, 0, 0, 0]}}
        assert not rule(evaluate(trackers_for(bad), key, labels, tmp, "dev")[0])[0]
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("roster", nargs="?"); ap.add_argument("--part", default="dev")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    lock = HERE / "heldout_scored.lock"
    if a.part == "heldout":
        assert not lock.exists(), f"HELD-OUT was already scored ({lock.read_text().strip()}); it is scored once"
    roster = json.load(open(a.roster))
    key, labels = json.load(open(TV2 / "gt_key.json")), json.load(open(TV2 / "gt_labels.json"))
    res, per = evaluate(trackers_for(roster), key, labels, a.src, a.part)
    for name, c in res.items():
        print(f"[{a.part}] {name:16s} {sg.summary(c)}")
    ok, (dw, dh) = rule(res)
    multi = sum(1 for k, v in roster.items() if v["multi"] and key and any(x["clip"] == k and x["part"] == a.part for x in key.values()))
    changed = sum(per["roster-activity"][p] != per["v1"][p] for p in per["v1"])
    fell = sum(1 for p in per["v1"] if roster[key[p]["clip"]].get("online_by") == "activity")
    print(f"[{a.part}] panels in clips where roster-online fell back to the activity ranking (tracker followed nobody eligible): {fell}")
    print(f"[{a.part}] clips where two or more people counted: {multi}; panels where roster-activity's outcome differs from v1: {changed}")
    print(f"[{a.part}] rule: WRONG(activity) - WRONG(v1) = {dw:+.1f} points (<= +2.0), HIT(activity) - HIT(v1) = {dh:+.1f} points (>= -5.0): "
          f"{'satisfied' if ok else 'NOT satisfied'}")
    if a.part == "heldout":
        lock.write_text(f"scored once with roster {a.roster}\n")


if __name__ == "__main__":
    main()
