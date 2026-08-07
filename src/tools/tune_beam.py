"""Tune beam width and length penalty on validation without touching test.

For each beam width, decoding is performed once and the raw finished N-best
hypotheses are rescored for every requested length penalty. This makes a useful
search several times cheaper than invoking eval_beam.py for every pair.
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
from slt.data import AuslanDailyDataset, BOS, EOS, Vocab, collate
from slt.metrics import corpus_bleu4, detokenize
from slt.reference_model import build_model


@torch.no_grad()
def decode_nbest(model, src, src_pad, max_len: int, beam: int):
    """Return raw-log-probability finished hypotheses for each sample."""
    model.eval()
    output = []
    for batch_index in range(src.size(0)):
        memory = model.encode(src[batch_index:batch_index + 1],
                              src_pad[batch_index:batch_index + 1])
        memory = memory.expand(beam, -1, -1).contiguous()
        source_pad = src_pad[batch_index:batch_index + 1].expand(
            beam, -1).contiguous()
        tokens = torch.full((beam, 1), BOS, dtype=torch.long, device=src.device)
        scores = torch.full((beam,), float("-inf"), device=src.device)
        scores[0] = 0.0
        finished = []
        for _ in range(max_len - 1):
            log_probability = torch.log_softmax(
                model.decode_step(tokens, memory, source_pad, None)[:, -1], dim=-1)
            vocab_size = log_probability.size(-1)
            top_scores, indices = (scores[:, None] + log_probability).reshape(
                -1).topk(beam)
            parents = torch.div(indices, vocab_size, rounding_mode="floor")
            next_tokens = indices % vocab_size
            tokens = torch.cat((tokens[parents], next_tokens[:, None]), dim=1)
            scores = top_scores.clone()
            for index in range(beam):
                if next_tokens[index].item() == EOS:
                    finished.append((scores[index].item(), tokens[index].tolist()))
                    scores[index] = float("-inf")
            if len(finished) >= beam:
                break
        if not finished:
            best = int(scores.argmax())
            finished.append((scores[best].item(), tokens[best].tolist()))
        output.append(finished)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--beams", default="3,5,8")
    parser.add_argument("--length_penalties", default="0,0.3,0.6,0.9,1.2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--legacy_abs_bbox_validity", action="store_true")
    args = parser.parse_args()

    beams = [int(value) for value in args.beams.split(",")]
    penalties = [float(value) for value in args.length_penalties.split(",")]
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**checkpoint["cfg"])
    if args.legacy_abs_bbox_validity:
        cfg.legacy_abs_bbox_validity = True
    vocab = Vocab(checkpoint["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = AuslanDailyDataset(args.manifest, vocab, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    hypotheses = {(beam, penalty): [] for beam in beams for penalty in penalties}
    references = []
    seen = 0
    for src, src_pad, _target, texts in loader:
        src, src_pad = src.to(device), src_pad.to(device)
        max_len = cfg.decode_max_len or cfg.max_tgt_tokens
        for beam in beams:
            candidates = decode_nbest(model, src, src_pad, max_len, beam)
            for penalty in penalties:
                selected = [
                    max(sample, key=lambda pair: pair[0] /
                        (max(1, len(pair[1])) ** penalty))[1]
                    for sample in candidates
                ]
                hypotheses[(beam, penalty)].extend(
                    vocab.decode(sequence) for sequence in selected)
        references.extend(texts)
        seen += len(texts)
        print(f"...{seen}/{len(dataset)}", flush=True)

    detokenized_references = [detokenize(text) for text in references]
    results = []
    for beam in beams:
        for penalty in penalties:
            output = hypotheses[(beam, penalty)]
            score = corpus_bleu4(
                [detokenize(text) for text in output], detokenized_references, "13a")
            result = {
                "beam": beam,
                "length_penalty": penalty,
                "bleu4": score,
                "unique_hypotheses": len(set(output)),
            }
            results.append(result)
            print(f"beam={beam} length_penalty={penalty:g} BLEU-4={score:.4f} "
                  f"unique={len(set(output))}/{len(output)}")
    results.sort(key=lambda item: item["bleu4"], reverse=True)
    payload = {
        "checkpoint": os.path.abspath(args.ckpt),
        "manifest": os.path.abspath(args.manifest),
        "selection_split": "validation",
        "best": results[0],
        "results": results,
    }
    print("BEST " + json.dumps(results[0], sort_keys=True))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        temporary = args.output_json + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, args.output_json)


if __name__ == "__main__":
    main()
