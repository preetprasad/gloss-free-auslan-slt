"""Inference-latency / throughput benchmark for the translator (research gap G1: real-time).

Measures single-stream (batch=1) latency of the pose->text model: encode + greedy decode per
clip. Reports mean/median/p90 ms per clip, clips/sec, and ms-per-input-frame (the deployment-
relevant number, since a live signer streams frames). This is the TRANSLATOR stage only; the
full real-time pipeline is pose-extraction (DWPose front-end) + this + TTS.

  python bench_latency.py --init_ckpt <best.pt> --test_manifest <test.tsv> \
      --num_keypoints 133 --keypoint_subset all --coord_norm frame --device cpu --limit 60
"""
from __future__ import annotations
import argparse, os, sys, time, statistics
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.cli import strict_bool
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.reference_model import build_model
from slt.engine import resolve_device


def main():
    ap = argparse.ArgumentParser()
    cfg0 = Config()
    for name, val in cfg0.to_dict().items():
        if isinstance(val, bool):
            ap.add_argument(f"--{name}", type=strict_bool, default=val)
        elif val is None:
            ap.add_argument(f"--{name}", type=str, default=None)
        else:
            ap.add_argument(f"--{name}", type=type(val), default=val)
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--beam", type=int, default=0, help="0=greedy; else beam width")
    a = ap.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in ("init_ckpt", "limit", "beam")})
    dev = resolve_device(cfg.device)

    ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
    vocab = Vocab(ck["vocab"])
    is_fusion = bool(cfg.rgb_feat_dir)
    if is_fusion:
        from slt.rgb_fusion import RGBFusionSLT, RGBDataset, collate_rgb
        model = RGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg).to(dev).eval()
        model.load_state_dict(ck["model"])
        ds = RGBDataset(a.test_manifest, vocab, cfg, "")
        collate_fn = collate_rgb
    else:
        model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev).eval()
        model.load_state_dict(ck["model"])
        ds = AuslanDailyDataset(a.test_manifest, vocab, cfg, "")
        collate_fn = collate
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    mx = cfg.decode_max_len or cfg.max_tgt_tokens

    ds.samples = ds.samples[:a.limit]
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    lat, frames, outs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if is_fusion:
                src, src_pad, rgb, rgb_pad, _t, _txt = batch
                src, src_pad, rgb, rgb_pad = src.to(dev), src_pad.to(dev), rgb.to(dev), rgb_pad.to(dev)
            else:
                src, src_pad, _t, _txt = batch
                src, src_pad = src.to(dev), src_pad.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if a.beam and a.beam > 1:
                out = (model.beam_decode(src, src_pad, rgb, rgb_pad, max_len=mx, beam=a.beam)
                       if is_fusion else model.beam_decode(src, src_pad, max_len=mx, beam=a.beam))
                nout = len(out[0])
            else:
                out = (model.greedy_decode(src, src_pad, rgb, rgb_pad, max_len=mx)
                       if is_fusion else model.greedy_decode(src, src_pad, max_len=mx))
                nout = out.shape[1]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if i < 3:            # warmup
                continue
            lat.append(dt); frames.append(int((~src_pad).sum().item())); outs.append(nout)

    n = len(lat)
    tot_frames = sum(frames)
    mode = f"beam{a.beam}" if a.beam > 1 else "greedy"
    print("===== TRANSLATOR LATENCY (gap G1) =====")
    print(f"device={dev} arch={cfg.model_arch} d_model={cfg.d_model} params={nparam:.2f}M "
          f"decode={mode} clips={n}")
    print(f"per-clip ms : mean={statistics.mean(lat):.1f}  median={statistics.median(lat):.1f}  "
          f"p90={sorted(lat)[int(0.9*n)]:.1f}  min={min(lat):.1f}  max={max(lat):.1f}")
    print(f"throughput  : {1000.0/statistics.mean(lat):.2f} clips/sec")
    print(f"input frames: mean={statistics.mean(frames):.1f}/clip  "
          f"ms/frame={sum(lat)/max(1,tot_frames):.2f}  "
          f"=> pose->text keeps up with ~{tot_frames/ (sum(lat)/1000.0):.0f} fps of signing")
    print(f"output len  : mean={statistics.mean(outs):.1f} tokens")


if __name__ == "__main__":
    main()
