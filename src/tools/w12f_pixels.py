"""Wave 12f: are the isolated-sign clips the SAME PIXELS as their located frame range, at full resolution?

tools/w12f_locate.py finds the source of each isolated clip on 16 x 16 grey thumbnails.  Gate E of the experiment
(the DWPose of an isolated clip must be the production DWPose of the frames it was cut from, within 1 pixel) can
only hold if the pose model is given the same picture twice, so this checks the full frames: for every located
clip, the largest absolute difference of any colour value between frame i of the isolated clip and frame
Begin_Frame + i of `<Video_Name_Clip>_signer.mp4`.  OpenCV calls only (no array arithmetic), no pose, no accuracy.
A local check before submission; it is not part of the Kaya job.

  python tools/w12f_pixels.py --located located.csv --bank <dir of <Video_ID>.mp4> --signer <dir of *_signer.mp4> [--workers 8]
  python tools/w12f_pixels.py --self_test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def max_diff(bank, signer, begin):
    """Largest absolute colour difference over the clip, or None if a frame is missing or the sizes differ."""
    import cv2
    if not bank or begin < 0 or begin + len(bank) > len(signer):
        return None
    worst = 0.0
    for i, fr in enumerate(bank):
        s = signer[begin + i]
        if fr.shape != s.shape:
            return None
        worst = max(worst, float(cv2.norm(fr, s, cv2.NORM_INF)))
    return worst


def frames(path):
    import cv2
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def _one(job):
    vid, bank, signer, begin = job
    return vid, max_diff(frames(bank), frames(signer), begin)


def self_test():
    import numpy as np
    rng = np.random.default_rng(0)
    sig = [rng.integers(0, 255, (12, 10, 3), dtype=np.uint8) for _ in range(20)]
    assert max_diff(sig[5:9], sig, 5) == 0.0
    assert max_diff(sig[5:9], sig, 6) > 0                       # one frame off is not the same picture
    off = [f.copy() for f in sig[5:9]]
    off[2][3, 4, 1] ^= 4                                          # one colour value changed by 4
    assert max_diff(off, sig, 5) == 4.0
    assert max_diff(sig[5:9], sig, 18) is None and max_diff([], sig, 0) is None
    assert max_diff([np.zeros((3, 3, 3), np.uint8)], sig, 0) is None
    print("w12f_pixels self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--located"); ap.add_argument("--bank"); ap.add_argument("--signer")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    from multiprocessing import Pool
    with open(a.located, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "located"]
    jobs = [(r["Video_ID"], os.path.join(a.bank, r["Video_ID"] + ".mp4"),
             os.path.join(a.signer, r["Video_Name_Clip"] + "_signer.mp4"), int(r["Begin_Frame"])) for r in rows]
    missing = [j[0] for j in jobs if not (os.path.exists(j[1]) and os.path.exists(j[2]))]
    assert not missing, f"{len(missing)} located clips without both videos, e.g. {missing[:5]}"
    with Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(_one, jobs, chunksize=16))
    split = {r["Video_ID"]: r["comm_split"] for r in rows}
    bad = sorted((v for v, d in res.items() if d is None), key=int)
    diff = sorted(((d, v) for v, d in res.items() if d), reverse=True)
    same = sum(d == 0.0 for d in res.values())
    print(f"located isolated clips {len(res)} | every colour value identical to the located frame range: {same} | "
          f"different: {len(diff)} | not comparable (missing frame or other size): {len(bad)}")
    print("identical, by split of the source clip: " +
          ", ".join(f"{s} {sum(res[v] == 0.0 for v in res if split[v] == s)}/{sum(split[v] == s for v in res)}"
                    for s in sorted(set(split.values()))))
    if diff:
        print("largest differences (grey levels, Video_ID): " + ", ".join(f"{d:.0f} ({v})" for d, v in diff[:10]))
    if bad:
        print("not comparable: " + ", ".join(bad[:20]))


if __name__ == "__main__":
    sys.exit(main())
