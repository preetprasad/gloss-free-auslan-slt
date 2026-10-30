"""Diagnose a trained SLT checkpoint: what is it actually predicting, and where
does BLEU collapse? Decodes the test manifest with the saved best.pt, then reports
per-corpus BLEU (Communication vs News), hypothesis-degeneracy stats, and example
hyp/ref pairs. Read-only — writes nothing except stdout.
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.metrics import corpus_bleu4, detokenize
from slt.reference_model import build_model


def corpus_of(clip: str) -> str:
    return "Communication" if os.path.basename(clip).startswith("video_") else "News"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    device = torch.device(dev)

    ds = AuslanDailyDataset(a.manifest, vocab, cfg)
    if a.limit:
        ds.samples = ds.samples[: a.limit]
    clips = [c for c, _ in ds.samples]
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    hyps, refs = [], []
    with torch.no_grad():
        for bi, (src, src_pad, _tgt, texts) in enumerate(dl):
            src, src_pad = src.to(device), src_pad.to(device)
            ys = model.greedy_decode(src, src_pad,
                                     max_len=cfg.decode_max_len or cfg.max_tgt_tokens)
            for row in ys.tolist():
                hyps.append(vocab.decode(row))
            refs.extend(texts)
            print(f"  ...decoded {len(hyps)}/{len(ds)}", flush=True)

    H = [detokenize(x) for x in hyps] if cfg.detokenize else list(hyps)
    R = [detokenize(x) for x in refs] if cfg.detokenize else list(refs)
    idx = {"Communication": [], "News": []}
    for i, c in enumerate(clips):
        idx[corpus_of(c)].append(i)

    print("\n================ DIAGNOSIS ================")
    print(f"checkpoint: {a.ckpt}")
    print(f"cfg: d_model={cfg.d_model} layers={cfg.num_encoder_layers}/{cfg.num_decoder_layers} "
          f"vocab={len(vocab)} epochs={cfg.epochs} lr={cfg.lr} bleu={cfg.bleu_tokenize}+detok{cfg.detokenize}")
    print(f"\noverall  n={len(H)}  BLEU-4={corpus_bleu4(H, R, cfg.bleu_tokenize):.2f}")
    for name in ("Communication", "News"):
        ii = idx[name]
        if ii:
            b = corpus_bleu4([H[i] for i in ii], [R[i] for i in ii], cfg.bleu_tokenize)
            print(f"  {name:14s} n={len(ii):4d}  BLEU-4={b:.2f}")

    hl = [len(h.split()) for h in hyps]
    rl = [len(r.split()) for r in refs]
    uniq = collections.Counter(hyps)
    top = uniq.most_common(1)[0]
    print(f"\nhyp length  mean={statistics.mean(hl):.1f}  (ref mean={statistics.mean(rl):.1f})")
    print(f"unique hypotheses = {len(uniq)}/{len(hyps)}  ({100*len(uniq)/len(hyps):.1f}% unique)")
    print(f"most common single output covers {100*top[1]/len(hyps):.1f}% of test: {top[0][:70]!r}")
    print("top-8 outputs:")
    for h, c in uniq.most_common(8):
        print(f"  {c:4d} | {h[:72]!r}")

    for name in ("Communication", "News"):
        print(f"\n--- examples: {name} ---")
        for i in idx[name][:6]:
            print(f"  REF: {refs[i][:95]}")
            print(f"  HYP: {hyps[i][:95]}")


if __name__ == "__main__":
    main()
