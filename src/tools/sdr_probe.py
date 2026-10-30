"""Representation-density probe — the cheap GATE on reopening SignCL.

SignCL (Ye et al., NeurIPS 2024, arXiv 2405.14312) argues gloss-free SLT is limited by
"representation density": features of semantically DISTINCT signs sit as close together as
features of ADJACENT frames within one sign, so the decoder cannot tell them apart. Their fix
is a contrastive term, and they report PHOENIX VLP 10.73 -> 13.51 and CSL-Daily GFSLT-VLP
11.00 -> 16.16.

This project's SignCL attempt is recorded as a negative (1.775) but it was CONFOUNDED: margin 1
against the paper's 64, loss weight 0.5 against 0.01, on normalised features, batch 16, in the
pre-DWPose wrong-bbox era. That tells us almost nothing about the method -- the same situation
as the graph-encoder adjacency init, which we re-ran fairly and only then closed on evidence.

Rather than spend a training run on a re-run, this probe measures whether the headroom EXISTS,
in the style of the sign-spotting probe that saved a multi-day build by returning 5% recall.
Forward passes only, no training.

METRIC (an explicit OPERATIONALISATION -- the paper's exact formula is not reproduced here):
for each clip we take the encoder's frame features and compare
  * POSITIVE scale = mean distance between temporally ADJACENT frames (|dt| = 1), i.e. the
    within-sign scale, and
  * NEGATIVE pairs = frames far enough apart to be different signs, |dt| > 2 * margin, with
    margin = max(10, T / n_text_tokens * 2.3) exactly as SignCL defines it.
`density` = the fraction of those DISTANT pairs whose distance is BELOW the clip's adjacent-frame
mean -- i.e. how often two frames that must belong to different signs are as close together as
two frames inside the same sign. HIGH density = entangled = SignCL has headroom.
LOW density = pose features are already well separated = close the lever cheaply and honestly.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.reference_model import build_model


@torch.no_grad()
def probe(ckpt_path, manifest, device, n_clips, seed=0):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(manifest, vocab, cfg)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device).eval()
    sd = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_clips, len(ds)), replace=False)
    dens, adj_mu, far_mu, margins, kept = [], [], [], [], 0
    for i in idx:
        src, src_pad, tgt, text = collate([ds[int(i)]])
        src, src_pad = src.to(device), src_pad.to(device)
        mem = model.encode(src, src_pad)[0]                  # (T, d)
        T = int((~src_pad[0]).sum().item())
        mem = mem[:T].float()
        n_tok = max(1, len(str(text[0]).split()))
        margin = max(10.0, T / n_tok * 2.3)
        if T < 2 * margin + 4:                               # too short to contain a far pair
            continue
        d = torch.cdist(mem, mem)                            # (T, T) euclidean
        ar = torch.arange(T, device=d.device)
        dt = (ar[:, None] - ar[None, :]).abs()
        adj = d[dt == 1]
        far = d[dt > 2 * margin]
        if adj.numel() == 0 or far.numel() == 0:
            continue
        a_mu = adj.mean()
        dens.append(float((far < a_mu).float().mean()))
        adj_mu.append(float(a_mu)); far_mu.append(float(far.mean()))
        margins.append(margin); kept += 1
    return dict(n=kept, density=float(np.mean(dens)) if dens else float("nan"),
                adj=float(np.mean(adj_mu)) if adj_mu else float("nan"),
                far=float(np.mean(far_mu)) if far_mu else float("nan"),
                ratio=float(np.mean(far_mu) / np.mean(adj_mu)) if adj_mu else float("nan"),
                margin=float(np.mean(margins)) if margins else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_clips", type=int, default=300)
    a = ap.parse_args()
    print(f"{'checkpoint':38} {'n':>4} {'density':>8} {'adj_d':>8} {'far_d':>8} {'far/adj':>8} {'margin':>7}")
    for c in a.ckpts:
        try:
            r = probe(c, a.manifest, a.device, a.n_clips)
            print(f"{os.path.basename(os.path.dirname(c)):38} {r['n']:4d} {r['density']:8.3f} "
                  f"{r['adj']:8.2f} {r['far']:8.2f} {r['ratio']:8.3f} {r['margin']:7.1f}")
        except Exception as e:
            print(f"{c}: FAILED {type(e).__name__}: {e}")
    print("\ndensity = fraction of DISTANT frame pairs closer together than the mean ADJACENT pair.")
    print("HIGH (>~0.4) => distinct signs are entangled => SignCL has headroom, a faithful re-run is justified.")
    print("LOW  (<~0.15) => pose features already separated => close the lever cheaply, as the spotting probe did.")


if __name__ == "__main__":
    main()
