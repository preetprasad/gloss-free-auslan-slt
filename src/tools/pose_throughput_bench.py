#!/usr/bin/env python3
"""Laptop throughput of the exact DWPose front end and of the translator (deployment evidence, RQ4).

Pose front end: rtmlib Wholebody in the configuration every Auslan-Daily and MM-WLAuslan extraction
used (`--mode balanced` = YOLOX-m detector at 640x640 + RTMW-DW-x-l at 256x192; see
results/dwpose_experiment/dwpose_extract_worker.py and jobs/slurm/extract_dwpose.sbatch). Both ONNX models
are loaded from the local rtmlib cache by explicit path, so nothing is downloaded. Frames are decoded
before timing, so video decoding is excluded. Three measurements:

  1. full pipeline (detector + pose on every frame), exactly as the worker calls it, on onnxruntime's
     CPU provider; the CoreML provider is attempted too and any failure is recorded, not hidden;
  2. pose model only, on every frame, with the detector run once per clip on the first frame that has
     a person (a single signer stays in frame, so this is a realistic live configuration) -- CPU;
  3. the same pose-only loop on the CoreML provider, with keypoints checked against CPU on identical
     boxes before its speed is reported.

Translator: a saved checkpoint decodes real test clips one at a time (batch 1), greedy and beam-5
(length penalty 0.6, as in eval_beam.py), on CPU and MPS, after a warm-up. Feature loading is excluded.

Usage:
  python tools/pose_throughput_bench.py --videos a.mp4 b.mp4 --ckpt best.pt --manifest test.tsv --out res.json
"""
import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
CACHE = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints")


def summarise(ms):
    ms = np.asarray(ms, dtype=np.float64)
    return {"n": int(ms.size), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "fps_from_mean": float(1000.0 / ms.mean())}


def load_frames(videos, max_frames):
    per_video, meta = [], []
    for v in videos:
        cap = cv2.VideoCapture(v)
        meta.append({"video": os.path.basename(v), "fps": cap.get(cv2.CAP_PROP_FPS),
                     "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]})
        frames = []
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        per_video.append(frames)
    return per_video, meta


def providers_bound(model):
    out = {}
    for name, obj in vars(model).items():
        sess = getattr(obj, "session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            out[name] = sess.get_providers()
    return out


def first_person(kp, sc):
    return (np.asarray(kp[0], np.float32), np.asarray(sc[0], np.float32)) if sc is not None and len(sc) else None


def time_loop(call, per_video, warmup):
    flat = [fr for frames in per_video for fr in frames]
    for fr in flat[:warmup]:
        call(fr, None)
    times, outs = [], []
    for vi, frames in enumerate(per_video):
        for fr in frames:
            t0 = time.perf_counter()
            kp, sc = call(fr, vi)
            times.append((time.perf_counter() - t0) * 1000.0)
            outs.append(first_person(kp, sc))
    return times, outs


def agreement(ref_outs, cand_outs, conf=0.3):
    """Median keypoint displacement of candidate vs reference, as a fraction of the reference body extent."""
    per_frame = []
    for r, c in zip(ref_outs, cand_outs):
        if r is None or c is None:
            continue
        ok = r[1] > conf
        if ok.sum() < 10:
            continue
        span = float(np.linalg.norm(r[0][ok].max(axis=0) - r[0][ok].min(axis=0))) or 1.0
        per_frame.append(float(np.median(np.linalg.norm(r[0][ok] - c[0][ok], axis=1))) / span)
    return {"frames_compared": len(per_frame),
            "median_disp_over_body_extent": float(np.median(per_frame)) if per_frame else None,
            "p95_disp_over_body_extent": float(np.percentile(per_frame, 95)) if per_frame else None}


def bench_translator(device, ckpt, manifest, n_clips, warmup):
    import torch
    sys.path.insert(0, HERE)
    from config import Config
    from slt.data import AuslanDailyDataset, Vocab, collate
    from slt.reference_model import build_model
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization, sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(manifest, vocab, cfg)
    ds.samples = ds.samples[:n_clips + warmup]
    dev = torch.device(device)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    greedy, beam, frames = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            src, src_pad, _tgt, _texts = collate([ds[i]])
            src, src_pad = src.to(dev), src_pad.to(dev)
            t0 = time.perf_counter(); model.greedy_decode(src, src_pad, max_len=mx); sync()
            g = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter(); model.beam_decode(src, src_pad, max_len=mx, beam=5, length_penalty=0.6); sync()
            b = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                greedy.append(g); beam.append(b); frames.append(int(src.shape[1]))
    return {"greedy": summarise(greedy), "beam5": summarise(beam), "mean_input_frames": float(np.mean(frames))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--max_frames", type=int, default=150, help="frames per video")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--det", default=os.path.join(CACHE, "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"))
    ap.add_argument("--pose", default=os.path.join(CACHE, "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--n_clips", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in (a.det, a.pose):
        if not os.path.exists(p):
            sys.exit(f"FATAL: {p} not in the local rtmlib cache -- refusing to download")

    import onnxruntime
    from rtmlib import Wholebody
    res = {"machine": {"platform": platform.platform(), "machine": platform.machine(),
                       "mac_ver": platform.mac_ver()[0], "cpu_count": os.cpu_count(),
                       "onnxruntime": onnxruntime.__version__,
                       "available_providers": onnxruntime.get_available_providers()},
           "pose_config": {"det": os.path.basename(a.det), "det_input": [640, 640],
                           "pose": os.path.basename(a.pose), "pose_input": [192, 256]}}
    per_video, meta = load_frames(a.videos, a.max_frames)
    res["videos"] = meta
    n_frames = sum(len(f) for f in per_video)
    print(f"[pose] {n_frames} frames from {len(meta)} videos", flush=True)

    build = lambda device: Wholebody(det=a.det, det_input_size=(640, 640), pose=a.pose, pose_input_size=(192, 256),
                                     backend="onnxruntime", device=device)
    wb_cpu = build("cpu")
    res["providers_cpu"] = providers_bound(wb_cpu)

    # 1. full pipeline, as the extraction worker runs it
    t, full_cpu = time_loop(lambda fr, vi: wb_cpu(fr), per_video, a.warmup)
    res["full_pipeline_cpu"] = {**summarise(t), "frames_without_person": sum(o is None for o in full_cpu)}
    print(f"[full cpu] {res['full_pipeline_cpu']}", flush=True)
    try:
        wb_mps = build("mps")
        res["providers_mps"] = providers_bound(wb_mps)
        t, _ = time_loop(lambda fr, vi: wb_mps(fr), per_video, a.warmup)
        res["full_pipeline_coreml"] = summarise(t)
    except Exception as e:  # recorded, not hidden
        res["full_pipeline_coreml"] = {"failed": f"{type(e).__name__}: {str(e)[:400]}"}
        wb_mps = None
    print(f"[full coreml] {res['full_pipeline_coreml']}", flush=True)

    # 2./3. pose model only, detector once per clip (CPU detector, first frame with a person)
    boxes = []
    for frames in per_video:
        bb = []
        for fr in frames:
            bb = wb_cpu.det_model(fr)
            if len(bb):
                break
        boxes.append(bb[:1] if len(bb) else bb)
    res["clips_without_detection"] = sum(len(b) == 0 for b in boxes)
    t, pose_cpu = time_loop(lambda fr, vi: wb_cpu.pose_model(fr, bboxes=boxes[vi if vi is not None else 0]),
                            per_video, a.warmup)
    res["pose_only_cpu"] = summarise(t)
    print(f"[pose-only cpu] {res['pose_only_cpu']}", flush=True)
    if wb_mps is None:
        wb_mps = build("mps")
        res["providers_mps"] = providers_bound(wb_mps)
    try:
        t, pose_mps = time_loop(lambda fr, vi: wb_mps.pose_model(fr, bboxes=boxes[vi if vi is not None else 0]),
                                per_video, a.warmup)
        res["pose_only_coreml"] = {**summarise(t), "agreement_with_cpu": agreement(pose_cpu, pose_mps)}
    except Exception as e:
        res["pose_only_coreml"] = {"failed": f"{type(e).__name__}: {str(e)[:400]}"}
    print(f"[pose-only coreml] {res['pose_only_coreml']}", flush=True)

    if a.ckpt and a.manifest:
        import torch
        for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
            res[f"translator_{device}"] = bench_translator(device, a.ckpt, a.manifest, a.n_clips, a.warmup)
            print(f"[translator {device}] {res[f'translator_{device}']}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
