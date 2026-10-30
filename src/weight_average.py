"""Weight averaging ('model soup' / SWA) of same-architecture, same-vocab checkpoints.

Averages the model tensors of N checkpoints into ONE model — a single forward pass at inference
(unlike logit-ensembling which runs all N). Standard NMT/SLT free booster. Members MUST share
architecture + vocab (e.g. our 3 DWPose+CTC seeds).

  python weight_average.py --ckpts a.pt,b.pt,c.pt --out souped.pt
"""
from __future__ import annotations
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = [p.strip() for p in a.ckpts.split(",") if p.strip()]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for c in cks[1:]:
        assert c["vocab"] == cks[0]["vocab"], "soup members must share vocab"
    sds = [c["model"] for c in cks]
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        v0 = sds[0][k]
        if torch.is_floating_point(v0):
            avg[k] = sum(sd[k].float() for sd in sds) / len(sds)
            avg[k] = avg[k].to(v0.dtype)
        else:
            avg[k] = v0            # buffers / int tensors: take first
    torch.save({"model": avg, "vocab": cks[0]["vocab"], "cfg": cks[0].get("cfg", {})}, a.out)
    print(f"[soup] averaged {len(sds)} checkpoints -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
