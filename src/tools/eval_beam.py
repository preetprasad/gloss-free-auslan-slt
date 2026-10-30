"""Evaluate a checkpoint with greedy vs beam-search decoding.

Per-epoch training stays greedy (fast); this applies beam search to a finished
checkpoint (slow, one-off) to recover the BLEU that greedy leaves on the table.
Reports BLEU-4 under our protocol (13a+detok) and paper-like (none), for both
decoders, plus unique-output counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.metrics import corpus_bleu4, detokenize
from slt.reference_model import build_model


def report(hyps, refs, tag):
    payload = {
        "tag": tag,
        "n": len(hyps),
        "unique_hypotheses": len(set(hyps)),
        "metrics": {},
    }
    print(f"[{tag}] n={len(hyps)}  unique={payload['unique_hypotheses']} "
          f"({100*payload['unique_hypotheses']/max(1,len(hyps)):.1f}%)")
    for tok, det in (("13a", True), ("none", False)):
        H = [detokenize(x) for x in hyps] if det else hyps
        R = [detokenize(x) for x in refs] if det else refs
        bleu4 = corpus_bleu4(H, R, tok)
        payload["metrics"][f"tokenize={tok};detok={det}"] = bleu4
        print(f"    tokenize={tok:4s} detok={str(det):5s}  BLEU-4={bleu4:5.2f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--legacy_abs_bbox_validity", action="store_true")
    ap.add_argument("--output_json", default="",
                    help="optional structured result path; safe for selected test decode")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    if a.legacy_abs_bbox_validity:
        cfg.legacy_abs_bbox_validity = True
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    device = torch.device(dev)

    ds = AuslanDailyDataset(a.manifest, vocab, cfg)
    if a.limit:
        ds.samples = ds.samples[: a.limit]
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    g_h, b_h, refs = [], [], []
    with torch.no_grad():
        for src, src_pad, _t, texts in dl:
            src, src_pad = src.to(device), src_pad.to(device)
            decode_len = cfg.decode_max_len or cfg.max_tgt_tokens
            gy = model.greedy_decode(src, src_pad, max_len=decode_len)
            g_h += [vocab.decode(r) for r in gy.tolist()]
            by = model.beam_decode(src, src_pad, max_len=decode_len,
                                   beam=a.beam, length_penalty=a.length_penalty)
            b_h += [vocab.decode(r) for r in by]
            refs += list(texts)
            print(f"  ...{len(refs)}", flush=True)

    print()
    greedy = report(g_h, refs, "greedy")
    beam = report(b_h, refs, f"beam{a.beam}")
    if a.output_json:
        payload = {
            "checkpoint": os.path.abspath(a.ckpt),
            "manifest": os.path.abspath(a.manifest),
            "split": "test_or_requested_manifest",
            "beam": a.beam,
            "length_penalty": a.length_penalty,
            "greedy": greedy,
            "beam_result": beam,
            # Hypotheses ride along so ONE decode can answer mechanism questions -- e.g. whether a
            # subword model ever emits a word outside the word-level vocabulary. Metrics alone cannot,
            # and dump_hyps.py builds Vocab(itos) without tokenisation args, so it garbles
            # SentencePiece output. Additive keys only; existing readers look up named fields.
            "greedy_hyps": g_h,
            "beam_hyps": b_h,
            "refs": refs,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.output_json)), exist_ok=True)
        temporary = a.output_json + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, a.output_json)


if __name__ == "__main__":
    main()
