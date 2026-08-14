"""Dump per-clip (reference, hypothesis) pairs from a trained checkpoint for semantic grading.

Handles both the plain pose translator (train_ctc / VLP checkpoints) and the RGB-fusion model.
Decodes the official test split (greedy, + beam if --beam>1) in manifest order and writes a
jsonl: {"idx", "ref", "hyp_greedy", "hyp_beam"}. BLEU punishes valid paraphrase, so these pairs
are then graded for semantic adequacy by a local LLM (grade_semantic.py) to show whether the
BLEU number understates real translation quality.

Run on Kaya (CPU `work` is fine — decoding is light):
  python dump_hyps.py --ckpt <best.pt> --test_manifest <test.tsv> --out hyps.jsonl \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame [--beam 5] [--rgb_feat_dir <dir>]
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/src on path via caller
# caller sets cwd to src; import from there
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, PAD
from slt.reference_model import build_model
from slt.engine import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--num_keypoints", type=int, default=133)
    ap.add_argument("--keypoint_subset", default="all")
    ap.add_argument("--coord_norm", default="frame")
    ap.add_argument("--rgb_feat_dir", default="")
    ap.add_argument("--rgb_feature_dim", type=int, default=1152)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = resolve_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    itos = ck["vocab"]
    vocab = Vocab(itos)
    cfgd = ck.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfgd.items() if k in Config().to_dict()})
    cfg.num_keypoints = a.num_keypoints; cfg.keypoint_subset = a.keypoint_subset
    cfg.coord_norm = a.coord_norm; cfg.test_manifest = a.test_manifest
    is_fusion = bool(a.rgb_feat_dir) or bool(cfgd.get("rgb_feat_dir"))
    mx = cfg.decode_max_len or cfg.max_tgt_tokens

    if is_fusion:
        from slt.rgb_fusion import RGBDataset, collate_rgb, RGBFusionSLT
        cfg.rgb_feat_dir = a.rgb_feat_dir or cfgd.get("rgb_feat_dir")
        cfg.rgb_feature_dim = a.rgb_feature_dim or cfgd.get("rgb_feature_dim", 1152)
        model = RGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
        ds = RGBDataset(a.test_manifest, vocab, cfg, "")
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_rgb)
        rows = []
        with torch.no_grad():
            for src, src_pad, rgb, rgb_pad, _tgt, texts in dl:
                src, src_pad, rgb, rgb_pad = src.to(dev), src_pad.to(dev), rgb.to(dev), rgb_pad.to(dev)
                g = model.greedy_decode(src, src_pad, rgb, rgb_pad, max_len=mx).tolist()
                b = (model.beam_decode(src, src_pad, rgb, rgb_pad, max_len=mx, beam=a.beam)
                     if a.beam > 1 else [None] * len(texts))
                for gr, be, ref in zip(g, b, texts):
                    rows.append({"ref": ref, "hyp_greedy": vocab.decode(gr),
                                 "hyp_beam": vocab.decode(be) if be else ""})
    else:
        model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
        ds = AuslanDailyDataset(a.test_manifest, vocab, cfg, "")
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
        rows = []
        with torch.no_grad():
            for src, src_pad, _tgt, texts in dl:
                src, src_pad = src.to(dev), src_pad.to(dev)
                g = model.greedy_decode(src, src_pad, max_len=mx).tolist()
                b = (model.beam_decode(src, src_pad, max_len=mx, beam=a.beam)
                     if a.beam > 1 else [None] * len(texts))
                for gr, be, ref in zip(g, b, texts):
                    rows.append({"ref": ref, "hyp_greedy": vocab.decode(gr),
                                 "hyp_beam": vocab.decode(be) if be else ""})

    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"idx": i, **r}) + "\n")
    print(f"[dump] wrote {len(rows)} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
