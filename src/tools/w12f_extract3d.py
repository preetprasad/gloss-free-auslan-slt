"""Wave 12f, measurement 2: whole-body pose WITH an estimated depth for a list of signer clips.

rtmlib's `Wholebody3d` (the same YOLOX-m person detector as the production DWPose extraction, then RTMW3D-x) gives,
per frame and detected person, 133 COCO-WholeBody keypoints as image x/y, a confidence, and a root-relative depth z.
As in the production worker (results/dwpose_experiment/dwpose_extract_worker.py) the FIRST detection of every frame
is kept, a frame without output is all zeros, and every decoded frame gives one row.

  <out>/<clip>.npz   KPS (T, 133, 3) float32 = x, y (pixels), confidence      Z (T, 133) float32 = estimated depth
                     FAILED int32 = frames on which the model raised (written as zeros); a resumed run counts them again

The pose weights are third-party and are NOT fetched here: --pose must be an existing file (the job script holds the
download switch).  Pre-registration: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md.

  python tools/w12f_extract3d.py --rgb_dir D --list ids.txt --out_dir O --pose model.onnx [--device cuda]
  python tools/w12f_extract3d.py --self_test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

K = 133


def first_person(keypoints, scores, keypoints_2d):
    """One frame's (133, 3) x/y/confidence and (133,) depth from Wholebody3d's output; zeros if nobody was returned."""
    if scores is None or len(scores) == 0:
        return np.zeros((K, 3), np.float32), np.zeros(K, np.float32)
    xy = np.asarray(keypoints_2d[0], np.float32)[:, :2]
    sc = np.asarray(scores[0], np.float32).reshape(K, 1)
    z = np.asarray(keypoints[0], np.float32)[:, 2]
    assert xy.shape == (K, 2) and z.shape == (K,), (xy.shape, z.shape)
    return np.concatenate([xy, sc], 1), z


def extract(frames, model):
    """frames: iterable of BGR images.  Returns KPS (T, 133, 3), Z (T, 133) and the number of frames that failed."""
    kps, zs, failed = [], [], 0
    for fr in frames:
        try:
            keypoints, scores, _, keypoints_2d = model(fr)
            k, z = first_person(keypoints, scores, keypoints_2d)
        except Exception as e:                                  # one unreadable frame must not lose the clip
            failed += 1
            if failed <= 3:
                print(f"  frame failed: {type(e).__name__}: {e}", flush=True)
            k, z = np.zeros((K, 3), np.float32), np.zeros(K, np.float32)
        kps.append(k); zs.append(z)
    if not kps:
        kps, zs = [np.zeros((K, 3), np.float32)], [np.zeros(K, np.float32)]
    return np.stack(kps), np.stack(zs), failed


def prior(path):
    """(frames, failed frames) of a clip written by an earlier run, so that the 1% limit holds across resumed runs."""
    with np.load(path) as z:
        return len(z["KPS"]), int(z["FAILED"])


def read_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def self_test():
    rng = np.random.default_rng(0)

    class Fake:                                   # two people; person 0 must be the one kept
        def __init__(self): self.n = 0
        def __call__(self, fr):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            if self.n == 3:
                return np.zeros((0, K, 3)), np.zeros((0, K)), None, np.zeros((0, K, 2))
            kp = np.stack([np.full((K, 3), 7.0), np.full((K, 3), 9.0)])
            return kp, np.full((2, K), 0.5), None, np.stack([np.full((K, 2), 11.0), np.full((K, 2), 13.0)])

    kps, z, failed = extract([rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(4)], Fake())
    assert kps.shape == (4, K, 3) and z.shape == (4, K) and failed == 1
    assert np.all(kps[0, :, :2] == 11.0) and np.all(kps[0, :, 2] == 0.5) and np.all(z[0] == 7.0)   # person 0: x/y from the 2-D output, z from the 3-D one
    assert not kps[1].any() and not z[1].any() and not kps[2].any()                                # failed frame, empty frame
    assert np.all(z[3] == 7.0)
    kps, z, failed = extract([], Fake())
    assert kps.shape == (1, K, 3) and z.shape == (1, K) and not kps.any()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:    # what a resumed run reads back from a file written by an earlier run
        p = os.path.join(tmp, "c.npz")
        np.savez_compressed(p, KPS=np.zeros((7, K, 3), np.float32), Z=np.zeros((7, K), np.float32), FAILED=np.int32(2))
        assert prior(p) == (7, 2)
    print("w12f_extract3d self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--rgb_dir"); ap.add_argument("--list"); ap.add_argument("--out_dir"); ap.add_argument("--pose")
    ap.add_argument("--suffix", default="_signer.mp4")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    assert a.pose and os.path.isfile(a.pose), f"3-D pose weights not found at {a.pose} (they are not downloaded here)"
    ids = [l.strip() for l in open(a.list) if l.strip()]
    assert len(ids) == len(set(ids)) and ids, "empty or repeated clip list"
    video = {f[:-len(a.suffix)]: os.path.join(r, f) for r, _, fs in os.walk(a.rgb_dir) for f in fs if f.endswith(a.suffix)}
    missing = [c for c in ids if c not in video]
    assert not missing, f"{len(missing)} listed clips have no video, e.g. {missing[:5]}"
    os.makedirs(a.out_dir, exist_ok=True)

    from rtmlib import Wholebody3d
    model = Wholebody3d(pose=a.pose, mode="balanced", backend="onnxruntime", device=a.device)
    if a.device.startswith("cuda"):                # a silent CPU fallback would turn two hours into two days
        for m in (model.det_model, model.pose_model):
            assert m.session.get_providers()[0] == "CUDAExecutionProvider", f"not on the GPU: {m.session.get_providers()}"
    done = skip = frames = failed = 0
    for cid in ids:
        outp = os.path.join(a.out_dir, cid + ".npz")
        if os.path.exists(outp):                    # a resumed run: its failed frames still count (they are zero poses on disk)
            n0, f0 = prior(outp)
            skip += 1; frames += n0; failed += f0
            continue
        kps, z, f = extract(read_frames(video[cid]), model)
        tmp = os.path.join(a.out_dir, cid + ".writing")
        np.savez_compressed(tmp, KPS=kps, Z=z, FAILED=np.int32(f))
        os.replace(tmp + ".npz", outp)
        done += 1; frames += len(kps); failed += f
        if done % 50 == 0:
            print(f"done {done} skip {skip} / {len(ids)} | frames {frames} | failed frames {failed}", flush=True)
    print(f"FINISHED done {done} skip {skip} of {len(ids)} | frames {frames} | failed frames {failed}", flush=True)
    assert failed <= 0.01 * max(frames, 1), "more than 1% of frames failed in the 3-D model"


if __name__ == "__main__":
    sys.exit(main())
