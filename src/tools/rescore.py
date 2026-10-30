"""Re-score a checkpoint's greedy outputs under several BLEU protocols.

Separates the *measurement* gap (sacreBLEU 13a+detok, our strict protocol, vs
tokenize=none which is closer to what the Auslan-Daily paper reports on already-
tokenised text) from the *modelling* gap. Also reports unique-output count as a
collapse check.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from slt.reference_model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="mps")
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
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    hyps, refs = [], []
    with torch.no_grad():
        for src, src_pad, _t, texts in dl:
            src, src_pad = src.to(device), src_pad.to(device)
            ys = model.greedy_decode(src, src_pad,
                                     max_len=cfg.decode_max_len or cfg.max_tgt_tokens)
            hyps += [vocab.decode(r) for r in ys.tolist()]
            refs += list(texts)

    print(f"n={len(hyps)}  unique_hyps={len(set(hyps))} ({100*len(set(hyps))/len(hyps):.1f}%)")
    print("greedy decode | BLEU-4 under each protocol:")
    for tok in ("none", "13a"):
        for det in (False, True):
            H = [detokenize(x) for x in hyps] if det else hyps
            R = [detokenize(x) for x in refs] if det else refs
            b = corpus_bleu4(H, R, tokenize=tok)
            tag = " <- our reported protocol" if (tok == "13a" and det) else (
                  " <- paper-like (already tokenised)" if (tok == "none" and not det) else "")
            print(f"  tokenize={tok:4s} detok={str(det):5s}  BLEU-4={b:5.2f}{tag}")
    w = corpus_wer([detokenize(x) for x in hyps], [detokenize(x) for x in refs])
    print(f"WER (detok) = {w:.3f}")


if __name__ == "__main__":
    main()
