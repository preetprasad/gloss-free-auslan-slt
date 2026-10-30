"""Gate C score (results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md, last addendum): v2 tracklets + activity choice
(eval_variant.mjs output) against index-0, v1, v2-tuned and the shipped roster-activity rule, on the two-rater ground truth.
Outcomes are score_gt.outcome's; the roster rows are score_person_labels.py's own (no choice -> scored as v1). The variant
always decides (a decline masks the clip), so it is scored exactly as written.

Rule, per part: WRONG(variant) <= WRONG(roster-activity) and HIT(variant) >= HIT(roster-activity) - 2 points.
    python score_variant.py ALLDET_DIR VARIANT.json --part dev
    python score_variant.py ALLDET_DIR VARIANT.json --part heldout   # once: writes heldout_scored.lock, then refuses
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TV2, PL = HERE.parent, HERE.parents[1] / "person_labels"
sys.path.insert(0, str(PL))
import score_person_labels as spl   # noqa: E402  (evaluate, rates, trackers_for; imports score_gt)

CUT_CLIP = "video_10_134"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("variant"); ap.add_argument("--part", required=True, choices=("dev", "heldout"))
    ap.add_argument("--name", default="variant")
    a = ap.parse_args()
    lock = HERE / "heldout_scored.lock"
    if a.part == "heldout":
        assert not lock.exists(), f"HELD-OUT was already scored ({lock.read_text().strip()}); it is scored once"
    variant = json.load(open(a.variant))
    roster = json.load(open(PL / "outcome" / "roster.json"))
    key, labels = json.load(open(TV2 / "gt_key.json")), json.load(open(TV2 / "gt_labels.json"))
    trackers = {n: fn for n, fn in spl.trackers_for(roster).items() if n != "roster-online"}
    trackers[a.name] = lambda clip, d: np.array(variant[clip]["chosen"], int)

    res, per = spl.evaluate(trackers, key, labels, a.src, a.part)
    for n, c in res.items():
        print(f"[{a.part}] {n:16s} {spl.sg.summary(c)}")
    h_v, w_v, _ = spl.rates(res[a.name]); h_r, w_r, _ = spl.rates(res["roster-activity"])
    ok = w_v <= w_r and h_v >= h_r - 2.0
    clips = {k["clip"] for k in key.values() if k["part"] == a.part}
    by = {}
    for c in sorted(clips):
        by[variant[c]["by"]] = by.get(variant[c]["by"], 0) + 1
    print(f"[{a.part}] variant choice made by (clips): {by}")
    print(f"[{a.part}] panels where {a.name} differs from roster-activity: "
          f"{sum(per[a.name][p] != per['roster-activity'][p] for p in per[a.name])}")
    print(f"[{a.part}] Gate C: WRONG {w_v:.1f}% vs roster-activity {w_r:.1f}% (<=), HIT {h_v:.1f}% vs {h_r:.1f}% - 2 "
          f"(>= {h_r - 2:.1f}): {'MET' if ok else 'NOT MET'}")
    if CUT_CLIP in clips:
        cut = {p: k for p, k in key.items() if k["clip"] == CUT_CLIP}
        r2, _ = spl.evaluate(trackers, cut, labels, a.src, a.part)
        for n, c in r2.items():
            print(f"[{a.part}] {CUT_CLIP} (camera cut) {n:16s} {spl.sg.summary(c)}")
    if a.part == "heldout":
        lock.write_text(f"scored once with {a.variant}\n")


if __name__ == "__main__":
    main()
