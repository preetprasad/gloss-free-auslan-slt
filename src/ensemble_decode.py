"""Decode-time ensemble of trained SLT checkpoints (training-free BLEU booster).

Averages per-step output log-probabilities across multiple models that share the SAME vocab,
then beam/greedy decodes. Handles heterogeneous members: plain pose models (PoseSLT / reference)
AND the RGB-fusion model — each contributes its own memory, we only ensemble the vocab logits.
This is the standard NMT/SLT ensemble trick (STMC-Transformer, UPC WMT-SLT). No retraining.

  python ensemble_decode.py --ckpts a.pt,b.pt,c.pt --rgb_feat_dir <dir> \
      --test_manifest test.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame \
      --beam 5 --length_penalty 0.6 --device cpu
"""
from __future__ import annotations
import argparse, os, sys
import jiwer
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.cli import strict_bool
from slt.preproc_guard import assert_preproc_matches_checkpoints
from slt.data import Vocab, PAD, BOS, EOS
from slt.reference_model import build_model
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from slt.engine import resolve_device


def is_fusion_sd(sd):
    return any(k.startswith(("rgb_proj", "rgb_encoder")) for k in sd)


def is_temporal_fusion_sd(sd):
    """Return whether this RGB member has P3's learnable temporal adapter."""
    return any(k.startswith("temporal_adapter.") for k in sd)


def is_gated_temporal_fusion_sd(sd):
    """Return whether a temporal-fusion checkpoint has P6's lateral gates."""
    return any(k.startswith(("pose_from_rgb.", "rgb_from_pose.", "pose_gate.", "rgb_gate.")) for k in sd)


class Member:
    """One ensemble member: holds the model + a fn to produce (memory, mem_pad) for a batch."""
    def __init__(self, ckpt, cfg, vocab, dev):
        sd = ckpt["model"]
        self.fusion = is_fusion_sd(sd)
        self.temporal_fusion = self.fusion and is_temporal_fusion_sd(sd)
        self.gated_temporal_fusion = self.temporal_fusion and is_gated_temporal_fusion_sd(sd)
        if self.fusion:
            if self.temporal_fusion:
                if self.gated_temporal_fusion:
                    from slt.gated_temporal_rgb_fusion import GatedTemporalRGBFusionSLT
                    m = GatedTemporalRGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg)
                else:
                    from slt.temporal_rgb_fusion import TemporalRGBFusionSLT
                    m = TemporalRGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg,
                                             adapter="pyramid")
            else:
                from slt.rgb_fusion import RGBFusionSLT
                m = RGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg)
        else:
            m = build_model(cfg.feature_dim(), len(vocab), cfg)
        m.load_state_dict(sd)
        self.model = m.to(dev).eval()

    @torch.no_grad()
    def memory(self, src, src_pad, rgb, rgb_pad):
        if self.fusion:
            mem_p = self.model.base.encode(src, src_pad)
            mem_r = self.model.encode_rgb(rgb, rgb_pad)
            return self.model._fuse(mem_p, src_pad, mem_r, rgb_pad)   # (mem, pad)
        return self.model.encode(src, src_pad), src_pad

    @torch.no_grad()
    def step_logp(self, ys, mem, pad):
        base = self.model.base if self.fusion else self.model
        return F.log_softmax(base.decode_step(ys, mem, pad, None)[:, -1], dim=-1)


@torch.no_grad()
def ensemble_beam(members, mems, max_len, beam, lp, dev, weights=None):
    """Per-sample beam over the WEIGHTED-AVERAGED log-probs of all members."""
    w = weights if weights is not None else [1.0] * len(members)
    wsum = sum(w)
    B = mems[0][0].size(0)
    outs = []
    for b in range(B):
        mem_b = [(m[b:b + 1].expand(beam, -1, -1).contiguous(),
                  p[b:b + 1].expand(beam, -1).contiguous()) for (m, p) in mems]
        ys = torch.full((beam, 1), BOS, dtype=torch.long, device=dev)
        scores = torch.full((beam,), float("-inf"), device=dev); scores[0] = 0.0
        fin = []
        for _ in range(max_len - 1):
            lp_sum = None
            for wi, (mem, (mm, pp)) in zip(w, zip(members, mem_b)):
                l = wi * mem.step_logp(ys, mm, pp)
                lp_sum = l if lp_sum is None else lp_sum + l
            logp = lp_sum / wsum                              # weighted mean log-prob
            V = logp.size(-1)
            cand = (scores.unsqueeze(1) + logp).view(-1)
            top, idx = cand.topk(beam)
            bi = torch.div(idx, V, rounding_mode="floor"); ti = idx % V
            ys = torch.cat([ys[bi], ti.unsqueeze(1)], 1); scores = top.clone()
            for i in range(beam):
                if ti[i].item() == EOS:
                    seq = ys[i].tolist()
                    fin.append((scores[i].item() / (max(1, len(seq)) ** lp), seq))
                    scores[i] = float("-inf")
            if len(fin) >= beam:
                break
        if not fin:
            fin.append((0.0, ys[int(scores.argmax())].tolist()))
        outs.append(max(fin, key=lambda x: x[0])[1])
    return outs


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
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_manifest", default="", help="override manifest (e.g. val for tuning)")
    ap.add_argument("--weights", default="", help="comma per-member weights (default uniform)")
    ap.add_argument("--allow_preproc_mismatch", action="store_true",
                    help="downgrade the preprocessing-agreement check to a warning (deliberate probes only)")
    a = ap.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items()
                    if k not in ("ckpts", "beam", "limit", "eval_manifest", "weights", "allow_preproc_mismatch")})
    if a.eval_manifest:
        cfg.test_manifest = a.eval_manifest
    dev = resolve_device(cfg.device)
    paths = [p.strip() for p in a.ckpts.split(",") if p.strip()]

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    vocab = Vocab(cks[0]["vocab"])
    for c in cks[1:]:
        assert c["vocab"] == cks[0]["vocab"], "ensemble members must share vocab"
    assert_preproc_matches_checkpoints(cfg, cks, a.allow_preproc_mismatch)
    members = [Member(c, cfg, vocab, dev) for c in cks]
    any_fusion = any(m.fusion for m in members)
    weights = None
    if a.weights:
        weights = [float(x) for x in a.weights.split(",")]
        assert len(weights) == len(members), "weights must match #ckpts"
    print(f"[ensemble] {len(members)} members (fusion={[m.fusion for m in members]}, "
          f"temporal={[m.temporal_fusion for m in members]} "
          f"gated={[m.gated_temporal_fusion for m in members]}) "
          f"weights={weights or 'uniform'} vocab={len(vocab)} beam={a.beam} dev={dev}", flush=True)

    if any_fusion:
        # cfg is built from ARGPARSE DEFAULTS, not from the checkpoints, so rgb_feat_dir is ""
        # unless passed. Without it RGBDataset feeds every fusion member a single ZERO frame and
        # the run silently degrades to pose-only (~ -0.4 BLEU). Same guard as tools/ensemble_mbr_wer.py:222.
        if not cfg.rgb_feat_dir or not os.path.isdir(cfg.rgb_feat_dir):
            raise SystemExit(
                f"[ensemble] {sum(m.fusion for m in members)} of {len(members)} members are RGB-fusion but "
                f"--rgb_feat_dir is {cfg.rgb_feat_dir!r}. Without it every fusion member is fed a single "
                f"ZERO frame and silently degrades to pose-only. Pass --rgb_feat_dir <dir>."
            )
        import glob as _glob
        _n = len(_glob.glob(os.path.join(cfg.rgb_feat_dir, "rgb_*.npz")))
        print(f"[ensemble] rgb_feat_dir={cfg.rgb_feat_dir} ({_n} feature files)", flush=True)
        from slt.rgb_fusion import RGBDataset, collate_rgb
        ds = RGBDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate_rgb
    else:
        from slt.data import AuslanDailyDataset, collate
        ds = AuslanDailyDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate
    if a.limit:
        ds.samples = ds.samples[:a.limit]
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens

    hyps, refs = [], []
    for batch in dl:
        if any_fusion:
            src, src_pad, rgb, rgb_pad, _t, texts = batch
            rgb, rgb_pad = rgb.to(dev), rgb_pad.to(dev)
        else:
            src, src_pad, _t, texts = batch
            rgb = rgb_pad = None
        src, src_pad = src.to(dev), src_pad.to(dev)
        mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in members]
        rows = ensemble_beam(members, mems, mx, a.beam, cfg.length_penalty, dev, weights)
        for r, ref in zip(rows, texts):
            hyps.append(vocab.decode(r)); refs.append(ref)
    h = [detokenize(x) for x in hyps]; r = [detokenize(x) for x in refs]
    bleu = corpus_bleu4(h, r, tokenize=cfg.bleu_tokenize)
    wer = corpus_wer(h, r)
    # Keep the scalar WER used throughout the project, but also emit its audit
    # components.  This makes it clear whether a change comes from wrong words,
    # missing words, or extra words (rather than masking them in one ratio).
    wer_detail = jiwer.process_words(r, h)
    substitutions = wer_detail.substitutions
    deletions = wer_detail.deletions
    insertions = wer_detail.insertions
    reference_words = wer_detail.hits + substitutions + deletions
    print(
        f"[ENSEMBLE TEST] beam{a.beam}_bleu4={bleu:.3f} wer={wer:.3f} "
        f"S={substitutions} D={deletions} I={insertions} N={reference_words} "
        f"uniq={len(set(hyps))} members={len(members)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
