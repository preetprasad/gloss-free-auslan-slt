"""Clean ensemble ablation — incremental (1..N members) + leave-one-out — in ONE decode pass.

Loads all members once, encodes each member's memory once per batch, then decodes every ablation
subset reusing those encodings (cheap). Reports val BLEU-4 / WER for:
  * incremental: strongest single, then +2nd, +3rd, ... +Nth  (marginal value of each member)
  * leave-one-out: full set minus each member                 (per-member contribution)
Validation-only (the 6-member set was already selected; this ablation is descriptive, not a new
selection). Reuses ensemble_decode's Member + ensemble_beam for identical decoding behaviour.

  python tools/ensemble_ablation.py --ckpts a.pt,b.pt,... --names VLP80,CTCs202,... \
      --rgb_feat_dir <dir> --eval_manifest val.tsv --num_keypoints 133 --coord_norm frame \
      --beam 5 --length_penalty 1.0 --device cpu
"""
from __future__ import annotations
import argparse, os, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.cli import strict_bool
from slt.data import Vocab
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from slt.engine import resolve_device
from ensemble_decode import Member, ensemble_beam


def main():
    ap = argparse.ArgumentParser()
    cfg0 = Config()
    for name, val in cfg0.to_dict().items():
        if isinstance(val, bool): ap.add_argument(f"--{name}", type=strict_bool, default=val)
        elif val is None: ap.add_argument(f"--{name}", type=str, default=None)
        else: ap.add_argument(f"--{name}", type=type(val), default=val)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--eval_manifest", default="")
    a = ap.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in ("ckpts", "names", "beam", "eval_manifest")})
    if a.eval_manifest: cfg.test_manifest = a.eval_manifest
    dev = resolve_device(cfg.device)
    paths = [p.strip() for p in a.ckpts.split(",") if p.strip()]
    names = [n.strip() for n in a.names.split(",")] if a.names else [f"m{i}" for i in range(len(paths))]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    vocab = Vocab(cks[0]["vocab"])
    members = [Member(c, cfg, vocab, dev) for c in cks]
    any_fusion = any(m.fusion for m in members)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    print(f"[abla] {len(members)} members: {names} | fusion={[m.fusion for m in members]} beam={a.beam}", flush=True)

    if any_fusion:
        from slt.rgb_fusion import RGBDataset, collate_rgb
        ds = RGBDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate_rgb
    else:
        from slt.data import AuslanDailyDataset, collate
        ds = AuslanDailyDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # subsets to evaluate (indices into members). Incremental assumes ckpts are passed strongest-first.
    N = len(members)
    subsets = {}
    for k in range(1, N + 1):
        subsets[f"top{k}: " + "+".join(names[:k])] = list(range(k))
    for i in range(N):
        subsets[f"LOO -{names[i]}"] = [j for j in range(N) if j != i]

    # decode: encode each member once per batch, decode every subset reusing the encodings
    hyps = {tag: [] for tag in subsets}; refs = []
    for batch in dl:
        if any_fusion:
            src, src_pad, rgb, rgb_pad, _t, texts = batch
            rgb, rgb_pad = rgb.to(dev), rgb_pad.to(dev)
        else:
            src, src_pad, _t, texts = batch; rgb = rgb_pad = None
        src, src_pad = src.to(dev), src_pad.to(dev)
        mems_all = [m.memory(src, src_pad, rgb, rgb_pad) for m in members]   # encode ONCE
        refs += list(texts)
        for tag, idx in subsets.items():
            sub_m = [members[i] for i in idx]; sub_mem = [mems_all[i] for i in idx]
            rows = ensemble_beam(sub_m, sub_mem, mx, a.beam, cfg.length_penalty, dev, None)
            hyps[tag] += [vocab.decode(r) for r in rows]

    r = [detokenize(x) for x in refs]
    print(f"\n[abla] {os.path.basename(cfg.test_manifest)}  beam={a.beam} LP={cfg.length_penalty}", flush=True)
    print(f"{'config':38s} {'BLEU-4':>8s} {'WER':>7s} {'uniq':>6s}")
    full = None
    for tag, _ in subsets.items():
        h = [detokenize(x) for x in hyps[tag]]
        bleu = corpus_bleu4(h, r, tokenize=cfg.bleu_tokenize); wer = corpus_wer(h, r)
        if tag.startswith(f"top{N}"): full = bleu
        delta = f"  (Δ{bleu-full:+.2f})" if (full is not None and tag.startswith("LOO")) else ""
        print(f"{tag:38s} {bleu:8.3f} {wer:7.3f} {len(set(hyps[tag])):6d}{delta}", flush=True)


if __name__ == "__main__":
    main()
