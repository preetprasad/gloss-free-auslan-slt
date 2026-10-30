"""Test-time augmentation over TEMPORAL VIEWS — a decode-time lever this project has never tried.

Motivation is measured, not assumed. The classical-CV round (2026-09-09d) subsampled source frames
and found sign identification does not degrade -- it slightly IMPROVES: top-1 24.9% @25fps -> 25.2%
@12.5 -> 26.5% @8.3 -> 26.9% @6.2fps. Sign identity lives in ~6 frames, implying roughly 4x temporal
redundancy in what the encoder consumes. Two consequences follow:

  * a subsampled view is not a degraded view -- it is a legitimate one, so averaging over views is
    not trading accuracy for diversity;
  * different PHASE OFFSETS of the same stride see genuinely different frames, so the views decorrelate
    while every one stays faithful to the clip.

That is the standard precondition for TTA, and it is satisfied here by measurement rather than hope.
Implementation reuses `ensemble_decode.ensemble_beam` unchanged: it sums per-step VOCAB log-probs, so
members may hold memories of different LENGTHS. Passing one model K times with K view-memories is
therefore exactly an ensemble over views, with no new decoding code.

Selection is on VALIDATION. Nothing here touches the official test split unless --manifest says so.
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config                                            # noqa: E402
from slt.data import AuslanDailyDataset, Vocab                       # noqa: E402
from slt.metrics import corpus_bleu4, corpus_wer, detokenize         # noqa: E402
from ensemble_decode import Member, ensemble_beam                    # noqa: E402


def views(T: int, spec: str) -> list[np.ndarray]:
    """Temporal views of a T-frame clip. 's1' = full rate; 's2p0' = stride 2, phase 0; etc."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "s1":
            out.append(np.arange(T)); continue
        s, p = tok.split("p"); s = int(s[1:]); p = int(p)
        idx = np.arange(p, T, s)
        if idx.size < 4:                       # too short to be a usable view -> fall back to full rate
            idx = np.arange(T)
        out.append(idx)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", default="s1")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    ck = torch.load(root / a.ckpt if not os.path.isabs(a.ckpt) else a.ckpt,
                    map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    dev = torch.device(a.device)
    member = Member(ck, cfg, vocab, dev)
    ds = AuslanDailyDataset(str(root / a.manifest), vocab, cfg)
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    spec = a.views
    print(f"{a.ckpt} | {n} clips | views='{spec}' | beam {a.beam} lp {a.length_penalty}")

    hyps, refs = [], []
    for i in range(n):
        feats, _tgt, text = ds[i][0], ds[i][1], ds[i][2]
        T = feats.shape[0]
        vw = views(T, spec)
        mems = []
        for idx in vw:
            src = feats[torch.as_tensor(idx, dtype=torch.long)].unsqueeze(0).to(dev)
            pad = torch.zeros(1, src.shape[1], dtype=torch.bool, device=dev)
            mems.append(member.memory(src, pad, None, None))
        out = ensemble_beam([member] * len(vw), mems, cfg.max_tgt_tokens,
                            a.beam, a.length_penalty, dev)
        hyps.append(vocab.decode(out[0])); refs.append(text)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}", flush=True)

    h = [detokenize(x) for x in hyps]; r = [detokenize(x) for x in refs]
    b = corpus_bleu4(h, r, tokenize="13a"); w = corpus_wer(h, r)
    print(f"\nviews={spec}  n_views={len(views(100, spec))}  BLEU-4 = {b:.2f}  WER = {w:.4f}  "
          f"unique = {len(set(hyps))}")
    if a.dump:
        import json
        with open(root / a.dump, "w") as f:
            for i, (hh, rr) in enumerate(zip(hyps, refs)):
                f.write(json.dumps({"idx": i, "ref": rr, "hyp_beam": hh}) + "\n")
        print(f"wrote {a.dump}")


if __name__ == "__main__":
    main()
