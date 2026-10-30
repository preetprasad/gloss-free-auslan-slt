"""Export all-detections npz files (results/dwpose_experiment/dwpose_extract_worker_alldet.py) for the Node evaluation.

Per clip writes <out>/<clip>.json  {T, W, H, NDET, CHOSEN}  and  <out>/<clip>.kp.f32 (T x N x 133 x 2)  and
<out>/<clip>.sc.f32 (T x N x 133), little-endian float32, exactly the arrays the npz holds (detections zero-padded).
    python export_alldet.py ALLDET_DIR OUT_DIR [--clips results/tracker_v2/gt_clips.json]
    python export_alldet.py --selftest
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def export(npz, out):
    d = np.load(npz)
    kp, sc = d["KP_ALL"].astype("<f4"), d["SC_ALL"].astype("<f4")
    T, N = kp.shape[:2]
    stem = Path(npz).stem
    (out / f"{stem}.kp.f32").write_bytes(kp.tobytes())
    (out / f"{stem}.sc.f32").write_bytes(sc.tobytes())
    W, H = (int(x) for x in d["WH"])
    json.dump({"T": int(T), "N": int(N), "W": W, "H": H, "NDET": d["NDET"].astype(int).tolist(),
               "CHOSEN": d["CHOSEN"].astype(int).tolist()}, open(out / f"{stem}.json", "w"))
    return stem


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); T, N = 5, 2
        np.savez(tmp / "c.npz", KP_ALL=np.random.rand(T, N, 133, 2).astype("f4"), SC_ALL=np.random.rand(T, N, 133).astype("f4"),
                 DET_BOX=np.zeros((T, N, 4), "f4"), NDET=np.array([2, 2, 1, 0, 2], "i2"), CHOSEN=np.array([0, 1, 0, -1, 1], "i2"),
                 WH=np.array([640, 480]))
        export(tmp / "c.npz", tmp)
        j = json.load(open(tmp / "c.json"))
        assert j["T"] == 5 and j["N"] == 2 and j["W"] == 640 and j["NDET"] == [2, 2, 1, 0, 2]
        assert len(np.fromfile(tmp / "c.kp.f32", "<f4")) == T * N * 133 * 2 and len(np.fromfile(tmp / "c.sc.f32", "<f4")) == T * N * 133
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--clips", default=str(ROOT / "results/tracker_v2/gt_clips.json")); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = [c for part in json.load(open(a.clips)).values() for c, _ in part]
    for c in clips:
        export(Path(a.src) / f"{c}.npz", out)
    print(f"exported {len(clips)} clips to {out}")
