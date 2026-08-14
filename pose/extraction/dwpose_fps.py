"""DWPose front-end FPS/latency benchmark (research gap G1 — the real-time bottleneck).

Times the DWPose (RTMW whole-body) pose estimator per frame, which the earlier latency work
showed is the dominant cost (the translator is ~24 ms/clip, negligible). Reports ms/frame and FPS.
"""
import os, glob, time, argparse, numpy as np, cv2
cv2.setNumThreads(1)
import onnxruntime as ort  # noqa
from rtmlib import Wholebody

ap = argparse.ArgumentParser()
ap.add_argument("--rgb_dir", default="/scratch/pmc099/pprasad/dwpose/rgb_clips")
ap.add_argument("--device", default="cuda")
ap.add_argument("--mode", default="balanced")
ap.add_argument("--clips", type=int, default=10)
a = ap.parse_args()

mp4s = sorted(glob.glob(a.rgb_dir + "/**/*.mp4", recursive=True))[:a.clips]
model = Wholebody(mode=a.mode, backend="onnxruntime", device=a.device)
lat = []
for v in mp4s:
    cap = cv2.VideoCapture(v)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    for i, fr in enumerate(frames):
        t0 = time.perf_counter()
        try: model(fr)
        except Exception: continue
        dt = (time.perf_counter() - t0) * 1000.0
        if len(lat) < 3 and i < 3: continue   # warmup
        lat.append(dt)
import statistics
n = len(lat)
print(f"===== DWPose FRONT-END LATENCY (gap G1) mode={a.mode} device={a.device} =====")
print(f"frames={n}  ms/frame: mean={statistics.mean(lat):.1f} median={statistics.median(lat):.1f} "
      f"p90={sorted(lat)[int(0.9*n)]:.1f}")
print(f"throughput: {1000.0/statistics.mean(lat):.1f} FPS  (video is ~25-30 fps; "
      f"{'REAL-TIME' if 1000.0/statistics.mean(lat)>=25 else 'BELOW real-time'})")
