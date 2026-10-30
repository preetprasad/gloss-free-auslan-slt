"""Wave 9 prep: turn the all-detections extraction into three pose arms and their manifests.

From ONE extraction (dwpose_alldet), per manifest clip:
  C0 = detection 0 each frame (what the original extractor saved; the current pipeline)
  V1 = the v1 tracker's choice (KPS field, computed by the imported v1 functions during extraction)
  V2 = tracker_v2.choose(d, TUNED); frames it masks are all-zero (the loader drops them); a clip it masks ENTIRELY falls
       back to V1 (counted).
Each arm file holds KPS (T,133,3) and an all-zero BBOX (T,4), the stored format, so arms differ only in whose pose it is.
Manifests: <arm>_{train,val,test}.tsv keep the official row order and English text, with the path swapped;
joint_<arm>.tsv = <arm>_train (12,439 rows) + the adopted synthetic rows (13,760) copied verbatim from a wave-6
adopted-recipe joint manifest, so the synthetic half is byte-identical across arms.
Usage: materialise_arms.py ALLDET_DIR OFFICIAL_MANIFEST_DIR ADOPTED_JOINT_TSV OUT_POSE_DIR OUT_MANIFEST_DIR [--workers N]
       materialise_arms.py --selftest
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import tracker_v2 as v2

ARMS = ("c0", "v1", "v2")


def arms_for(d):
    T = len(d["NDET"])
    c0 = np.zeros((T, 133, 3), np.float32)
    for t in range(T):
        if d["NDET"][t] > 0:
            c0[t, :, :2] = d["KP_ALL"][t, 0]; c0[t, :, 2] = d["SC_ALL"][t, 0]
    k1 = d["KPS"].astype(np.float32)
    k2, ch = v2.choose(d, v2.TUNED)
    fallback = not (ch >= 0).any()
    if fallback:
        k2 = k1.copy()
    stats = dict(T=T, v2_masked=int((ch < 0).sum()) if not fallback else 0, fallback=int(fallback),
                 c0_ne_v2=int(sum(not np.allclose(c0[t], k2[t]) for t in range(T))),
                 c0_ne_v1=int(sum(not np.allclose(c0[t], k1[t]) for t in range(T))))
    return {"c0": c0, "v1": k1, "v2": k2}, stats


def work(job):
    clip, src, out = job
    d = dict(np.load(os.path.join(src, clip + ".npz")))
    arms, stats = arms_for(d)
    for a in ARMS:
        p = os.path.join(out, a, clip + ".npz")
        np.savez_compressed(p[:-4] + ".writing", KPS=arms[a], BBOX=np.zeros((len(arms[a]), 4), np.float32))
        os.replace(p[:-4] + ".writing.npz", p)
    stats["clip"] = clip
    return stats


def rows(p):
    return [l.rstrip("\n").split("\t", 1) for l in open(p) if "\t" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alldet", nargs="?"); ap.add_argument("mandir", nargs="?"); ap.add_argument("joint", nargs="?")
    ap.add_argument("outpose", nargs="?"); ap.add_argument("outman", nargs="?")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    split_rows = {s: rows(os.path.join(a.mandir, f"{s}.tsv")) for s in ("train", "val", "test")}
    assert [len(split_rows[s]) for s in ("train", "val", "test")] == [12439, 800, 800], {s: len(r) for s, r in split_rows.items()}
    clips = [r[0].rsplit("/", 1)[1][:-4] for s in ("train", "val", "test") for r in split_rows[s]]
    miss = [c for c in clips if not os.path.exists(os.path.join(a.alldet, c + ".npz"))]
    assert not miss, f"{len(miss)} clips missing from the all-detections extraction, e.g. {miss[:5]}"
    for arm in ARMS:
        os.makedirs(os.path.join(a.outpose, arm), exist_ok=True)
    os.makedirs(a.outman, exist_ok=True)
    with Pool(a.workers) as pool:
        stats = pool.map(work, [(c, a.alldet, a.outpose) for c in clips], chunksize=20)
    json.dump(stats, open(os.path.join(a.outman, "arm_stats.json"), "w"))
    split_of = {r[0].rsplit("/", 1)[1][:-4]: s for s in ("train", "val", "test") for r in split_rows[s]}
    for s in ("train", "val", "test"):
        st = [x for x in stats if split_of[x["clip"]] == s]; F = sum(x["T"] for x in st)
        print(f"{s}: clips {len(st)} frames {F} | C0 != V2 on {sum(x['c0_ne_v2'] for x in st) / F:.2%} of frames "
              f"(C0 != V1 {sum(x['c0_ne_v1'] for x in st) / F:.2%}) | V2 masked {sum(x['v2_masked'] for x in st) / F:.2%} "
              f"| V2 fell back to V1 on {sum(x['fallback'] for x in st)} clips")
    joint = rows(a.joint)
    assert len(joint) == 26199 and [r[1] for r in joint[:12439]] == [r[1] for r in split_rows["train"]], "adopted joint manifest mismatch"
    syn = joint[12439:]
    assert all("/scratch_backup/dwpose/" not in r[0] for r in syn), "synthetic half contains real clips"
    for arm in ARMS:
        for s in ("train", "val", "test"):
            with open(os.path.join(a.outman, f"{arm}_{s}.tsv"), "w") as f:
                for p, t in split_rows[s]:
                    f.write(os.path.join(a.outpose, arm, p.rsplit("/", 1)[1]) + "\t" + t + "\n")
        with open(os.path.join(a.outman, f"joint_{arm}.tsv"), "w") as f:
            for p, t in split_rows["train"]:
                f.write(os.path.join(a.outpose, arm, p.rsplit("/", 1)[1]) + "\t" + t + "\n")
            for p, t in syn:
                f.write(p + "\t" + t + "\n")
    texts = {arm: [r[1] for r in rows(os.path.join(a.outman, f"joint_{arm}.tsv"))] for arm in ARMS}
    assert texts["c0"] == texts["v1"] == texts["v2"] and len(texts["c0"]) == 26199
    print("GATE OK: 3 arms x 14,039 clips written; joint manifests 26,199 rows each, English text identical across arms, "
          "synthetic half verbatim from the adopted manifest")


def selftest():
    """Detection 0 is a bystander; the signer is detection 1 in frames 0-9 and undetected in 10-14. C0 must take the
    bystander, V2 the signer, masking 10-14; a clip with no usable detection falls back to V1."""
    T = 15; KA = np.zeros((T, 2, 133, 2), np.float32); SA = np.zeros((T, 2, 133), np.float32)
    DB = np.zeros((T, 2, 4), np.float32); ND = np.zeros(T, np.int16)
    for t in range(T):
        KA[t, 0] = 50.0; SA[t, 0] = 0.9; DB[t, 0] = [0, 300, 120, 850]; ND[t] = 1
        if t < 10:
            KA[t, 1] = 300.0 + t; SA[t, 1] = 0.9; DB[t, 1] = [200, 100, 420, 850]; ND[t] = 2
    k1 = np.concatenate([KA[:, 1], SA[:, 1, :, None]], axis=2)
    d = dict(KP_ALL=KA, SC_ALL=SA, DET_BOX=DB, NDET=ND, WH=np.array([600, 900]), KPS=k1, CHOSEN=np.ones(T, np.int16))
    arms, st = arms_for(d)
    assert np.allclose(arms["c0"][:, :, 0], 50.0)
    assert np.allclose(arms["v2"][:10, 0, 0], 300 + np.arange(10)) and not arms["v2"][10:].any(), arms["v2"][:, 0, 0]
    assert st["v2_masked"] == 5 and st["fallback"] == 0
    d0 = dict(d, SC_ALL=np.zeros_like(SA))                       # nothing usable -> V2 masks all -> falls back to V1
    arms0, st0 = arms_for(d0)
    assert st0["fallback"] == 1 and np.allclose(arms0["v2"], k1)
    print("self-test OK: C0 = detection 0, V2 follows the signer and masks lost frames, full-mask clips fall back to V1")


if __name__ == "__main__":
    main()
