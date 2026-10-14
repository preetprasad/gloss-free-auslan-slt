"""Where does each clip of Auslan-Daily's isolated-sign set come from?  (wave 12f, measurement 1)

Auslan-Daily ships 3,209 "isolated" sign clips (`New_Gloss_RGB_Data.zip`, index
`Isolated_Sign_Language_Recognition.xlsx` with Gloss / Split / Video_ID only: no source clip, no frame
bounds).  The paper says they were annotated as temporal boundaries inside the continuous videos, and three
clips compared by hand on 13 Oct 2026 were frame ranges of `<clip>_signer.mp4` with identical pixels.  If a
reference clip is cut from the query's own footage, matching against it is a self-match, so the probe needs
the source clip of every isolated clip.  This tool finds it from the pixels alone:

  thumbs  decode videos and keep one SxS grey thumbnail per frame (OpenCV calls only, no array arithmetic)
  match   for each isolated clip: the signer clip of the same frame size and the frame offset whose
          thumbnails are closest (mean absolute grey-level difference over the whole isolated clip)
  self_test

Output of `match`: a CSV with one row per isolated clip.  Its columns Video_Name_Clip / Query_Gloss /
Begin_Frame / End_Frame are the ones `tools/continuous_id_probe.py --spot_csv` reads.
status: located   d_min <= TAU and no OTHER signer clip within MARGIN of it
        ambiguous d_min <= TAU but another signer clip is as close (duplicated footage)
        unlocated no signer clip of that frame size, or d_min > TAU
        empty     the isolated clip has no decodable frame
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

TAU = 1.0       # grey levels on the thumbnails. 20 test clips: true source exactly 0; one frame away 0.8 to 5.1; another clip >= 35
MARGIN = 1.0    # a second signer clip closer than d_min + MARGIN makes the source ambiguous


def _one(job):
    """(id, width, height, thumbs or None).  Header size first; decode only if the size is wanted."""
    import cv2
    path, vid, size, wanted = job
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if wanted is not None and (w, h) not in wanted:
        cap.release()
        return vid, w, h, None
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).tobytes())
    cap.release()
    return vid, w, h, np.frombuffer(b"".join(out), np.uint8).reshape(len(out), size * size)


def save_thumbs(path, res, size):
    res = sorted(res, key=lambda r: r[0])
    th = [r[3] for r in res if r[3] is not None and len(r[3])]
    np.savez_compressed(
        path, ids=np.array([r[0] for r in res]), wh=np.array([[r[1], r[2]] for r in res], np.int32).reshape(-1, 2),
        n=np.array([-1 if r[3] is None else len(r[3]) for r in res], np.int32),
        thumbs=np.concatenate(th) if th else np.zeros((0, size * size), np.uint8))


def load_thumbs(path):
    with np.load(path) as z:
        ids, whs, ns, th = z["ids"], z["wh"], z["n"], z["thumbs"]   # read ONCE: every z[...] decompresses the whole array again
    off = np.concatenate([[0], np.cumsum(np.maximum(ns, 0))])
    return {str(i): (tuple(int(v) for v in wh), th[off[k]:off[k + 1]] if n >= 0 else None)
            for k, (i, wh, n) in enumerate(zip(ids, whs, ns))}


def cmd_thumbs(a):
    from multiprocessing import Pool
    wanted = None
    if a.only_sizes:
        wanted = {wh for wh, _ in load_thumbs(a.only_sizes).values()}
    jobs = []
    for root, _, files in os.walk(a.videos):
        for f in files:
            if f.endswith(a.suffix):
                jobs.append((os.path.join(root, f), f[:-len(a.suffix)], a.size, wanted))
    ids = [j[1] for j in jobs]
    assert len(ids) == len(set(ids)), "two videos share an id"
    with Pool(a.workers) as pool:
        res = list(pool.imap_unordered(_one, sorted(jobs), chunksize=16))
    save_thumbs(a.out, res, a.size)
    dec = [r for r in res if r[3] is not None]
    print(f"videos {len(res)} | decoded {len(dec)} ({sum(len(r[3]) for r in dec)} frames) | "
          f"skipped by frame size {len(res) - len(dec)} | without a frame {sum(len(r[3]) == 0 for r in dec)} -> {a.out}")


def locate(bank_t: np.ndarray, cands: dict) -> tuple:
    """(d_min, clip, offset, d_other_clip, n_candidates) for one isolated clip against {clip: thumbs}."""
    n = len(bank_t)
    b = bank_t.astype(np.int16)
    best = (np.inf, "", -1)
    per_clip = {}
    for cid, t in cands.items():
        if t is None or len(t) < n:
            continue
        t = t.astype(np.int16)
        d0 = np.abs(t[:len(t) - n + 1] - b[0]).mean(1)               # first frame against every admissible offset
        d = min((float(np.abs(t[o:o + n] - b).mean()), int(o)) for o in np.argsort(d0, kind="stable")[:3])
        per_clip[cid] = d
        if d[0] < best[0]:
            best = (d[0], cid, d[1])
    other = min((v[0] for c, v in per_clip.items() if c != best[1]), default=np.inf)
    return best[0], best[1], best[2], other, len(per_clip)


def status_of(n, d_min, d_other):
    if n == 0:
        return "empty"
    if not d_min <= TAU:
        return "unlocated"
    return "located" if d_other - d_min >= MARGIN else "ambiguous"


def read_index(xlsx):
    """Video_ID -> (gloss lower-cased, split) from Isolated_Sign_Language_Recognition.xlsx."""
    import openpyxl
    rows = list(openpyxl.load_workbook(xlsx, read_only=True).active.iter_rows(values_only=True))
    col = {str(c): i for i, c in enumerate(rows[0]) if c is not None}
    return {str(r[col["Video_ID"]]).strip(): (str(r[col["Gloss"]]).strip().lower(), str(r[col["Split"]]).strip().lower())
            for r in rows[1:] if r[col["Video_ID"]] is not None}


def manifest_ids(path):
    return {os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(path) if "\t" in l}


def cmd_match(a):
    bank, sig = load_thumbs(a.bank), load_thumbs(a.signer)
    index = read_index(a.xlsx)
    assert set(bank) == set(index), f"isolated videos {len(bank)} and index rows {len(index)} do not name the same ids"
    split = {}
    for name, path in (("train", a.train), ("val", a.val), ("test", a.test)):
        for c in manifest_ids(path):
            assert c not in split, f"{c} is in two manifests"
            split[c] = name
    by_size = {}
    for cid, (wh, t) in sig.items():
        by_size.setdefault(wh, {})[cid] = t
    rows, tally = [], {}
    for vid in sorted(bank, key=int):
        wh, t = bank[vid]
        gloss, isp = index[vid]
        n = 0 if t is None else len(t)
        d, clip, off, other, nc = locate(t, by_size.get(wh, {})) if n else (np.inf, "", -1, np.inf, 0)
        st = status_of(n, d, other)
        if st != "located":
            clip_out, b, e = (clip if st == "ambiguous" else ""), -1, -1
        else:
            clip_out, b, e = clip, off, off + n
        cs = split.get(clip, "none") if st == "located" else ""
        tally[(st, cs)] = tally.get((st, cs), 0) + 1
        rows.append([vid, gloss, isp, n, wh[0], wh[1], clip_out, b, e,
                     "" if not np.isfinite(d) else f"{d:.4f}", "" if not np.isfinite(other) else f"{other:.4f}", nc, st, cs])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Video_ID", "Query_Gloss", "islr_split", "frames", "width", "height", "Video_Name_Clip", "Begin_Frame",
                    "End_Frame", "d_min", "d_other", "n_candidates", "status", "comm_split"])
        w.writerows(rows)
    print(f"isolated clips {len(rows)} | glosses {len({r[1] for r in rows})} -> {a.out}")
    for (st, cs), k in sorted(tally.items()):
        print(f"  {st:<9} {('source clip in Communication ' + cs) if cs else '':<40} {k:>5}")
    dm = np.array([float(r[9]) for r in rows if r[9] != ""])
    if len(dm):
        print("  d_min over clips with a same-size candidate: " +
              " | ".join(f"{q}% {np.percentile(dm, q):.3f}" for q in (50, 90, 99, 100)) +
              f" | exactly 0: {int((dm == 0).sum())} | in ({TAU}, 5]: {int(((dm > TAU) & (dm <= 5)).sum())}")
    same_video = sum(1 for r in rows if r[12] == "located" and not re.match(r"video_\d+_\d+$", r[6]))
    assert same_video == 0, "a located clip id is not of the form video_<n>_<m>"


def self_test():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (60, 256), dtype=np.uint8)
    decoy = rng.integers(0, 255, (60, 256), dtype=np.uint8)
    cut = a[17:30]
    d, clip, off, other, nc = locate(cut, {"video_1_1": a, "video_1_2": decoy, "video_1_3": a[:5]})
    assert (d, clip, off, nc) == (0.0, "video_1_1", 17, 2) and other > 50, (d, clip, off, other, nc)
    assert status_of(13, d, other) == "located"
    # the same footage in two signer clips cannot be attributed to one of them
    d, clip, off, other, _ = locate(cut, {"video_1_1": a, "video_2_9": a.copy()})
    assert d == 0.0 and status_of(13, d, other) == "ambiguous"
    # a clip that is nowhere, a clip with no candidate, an empty clip
    d, _, _, other, _ = locate(rng.integers(0, 255, (9, 256), dtype=np.uint8), {"video_1_1": a})
    assert status_of(9, d, other) == "unlocated"
    assert status_of(9, *locate(cut, {})[0:4:3]) == "unlocated" and status_of(0, np.inf, np.inf) == "empty"
    # codec noise below TAU is still located, one frame off is not the minimum
    noisy = np.clip(cut.astype(np.int16) + rng.integers(-1, 2, cut.shape), 0, 255).astype(np.uint8)
    d, clip, off, other, _ = locate(noisy, {"video_1_1": a, "video_1_2": decoy})
    assert 0 < d <= TAU and (clip, off) == ("video_1_1", 17) and status_of(13, d, other) == "located"
    # thumbnail file round trip, including a video skipped by size and one without frames
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.npz")
        save_thumbs(p, [("b", 4, 6, a[:3]), ("a", 8, 8, None), ("c", 4, 6, np.zeros((0, 256), np.uint8))], 16)
        t = load_thumbs(p)
        assert t["a"] == ((8, 8), None) and t["b"][0] == (4, 6) and np.array_equal(t["b"][1], a[:3]) and len(t["c"][1]) == 0
    print("w12f_locate self-test passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thumbs")
    t.add_argument("--videos", required=True); t.add_argument("--suffix", default=".mp4")
    t.add_argument("--out", required=True); t.add_argument("--size", type=int, default=16)
    t.add_argument("--only_sizes", default="", help="thumbnail file: decode only videos whose frame size occurs in it")
    t.add_argument("--workers", type=int, default=8)
    m = sub.add_parser("match")
    for k in ("bank", "signer", "xlsx", "train", "val", "test", "out"):
        m.add_argument("--" + k, required=True)
    sub.add_parser("self_test")
    a = ap.parse_args()
    {"thumbs": cmd_thumbs, "match": cmd_match, "self_test": lambda _: self_test()}[a.cmd](a)


if __name__ == "__main__":
    main()
