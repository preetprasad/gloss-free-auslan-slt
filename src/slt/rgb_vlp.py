"""RGB Visual-Language Pretraining (GFSLT-VLP style, on the RGB appearance stream).

Our earlier VLP aligned the *pose* encoder to text (-> 10.27). The literature's route to
15-21 (GFSLT-VLP) does CLIP-style pretraining on the RGB stream. This aligns the precomputed
DINOv2 crop features (the same rgb_<clip>.npz used by rgb_fusion) to text, producing an RGB
encoder that RGBFusionSLT warm-starts its rgb stream from -- so the appearance branch starts
text-grounded instead of cold.

Module names (rgb_proj / rgb_pos / rgb_encoder) are IDENTICAL to RGBFusionSLT so the pretrained
weights load straight in via prefix match.
"""
from __future__ import annotations
import math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import Config
from slt.data import Vocab, read_manifest, PAD
from slt.reference_model import SinusoidalPosition
from slt.vlp import TextEncoder, _masked_mean
from slt.rgb_fusion import _rgb_path
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device


# ── data: (rgb features, text) ────────────────────────────────────────────────
class RGBTextDataset(Dataset):
    def __init__(self, manifest, vocab, cfg):
        self.rows = read_manifest(manifest)
        self.vocab = vocab
        self.cfg = cfg
        self.rgb_dir = cfg.rgb_feat_dir
        self.rgb_dim = int(cfg.rgb_feature_dim)

    def __len__(self):
        return len(self.rows)

    def _load_rgb(self, clip):
        p = _rgb_path(clip, self.rgb_dir)
        if os.path.exists(p):
            try:
                with np.load(p) as z:
                    f = np.asarray(z["FEATURES"], dtype=np.float32)
                if f.ndim == 2 and f.shape[1] == self.rgb_dim and f.shape[0] >= 1 and np.isfinite(f).all():
                    if f.shape[0] > self.cfg.max_src_frames:
                        sel = np.linspace(0, f.shape[0] - 1, self.cfg.max_src_frames).round().astype(int)
                        f = f[sel]
                    return f
            except Exception:
                pass
        return np.zeros((1, self.rgb_dim), np.float32)

    def __getitem__(self, i):
        clip, text = self.rows[i]
        rgb = torch.from_numpy(self._load_rgb(clip))
        tgt = torch.tensor(self.vocab.encode(text, self.cfg.max_tgt_tokens), dtype=torch.long)
        return rgb, tgt


def collate_rgbtext(batch):
    rgbs, tgts = zip(*batch)
    B = len(rgbs)
    Tr = max(f.shape[0] for f in rgbs); Fr = rgbs[0].shape[1]
    rgb = torch.zeros(B, Tr, Fr); rgb_pad = torch.ones(B, Tr, dtype=torch.bool)
    for i, f in enumerate(rgbs):
        rgb[i, :f.shape[0]] = f; rgb_pad[i, :f.shape[0]] = False
    L = max(t.shape[0] for t in tgts)
    tgt = torch.full((B, L), PAD, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt[i, :t.shape[0]] = t
    return rgb, rgb_pad, tgt


# ── model ─────────────────────────────────────────────────────────────────────
class RGBVLPModel(nn.Module):
    def __init__(self, rgb_dim, vocab, cfg, proj=256, temp=0.07):
        super().__init__()
        d = cfg.d_model
        self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, d), nn.LayerNorm(d))
        self.rgb_pos = SinusoidalPosition(d, cfg.position_max_len)
        layer = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.rgb_encoder = nn.TransformerEncoder(layer, int(getattr(cfg, "rgb_encoder_layers", 2)))
        for attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(self.rgb_encoder, attr):
                setattr(self.rgb_encoder, attr, False)
        self.rgb_head = nn.Linear(d, proj)
        self.text = TextEncoder(vocab, d, cfg.nhead, max(2, cfg.num_encoder_layers // 2),
                                cfg.dim_feedforward, cfg.dropout, proj)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def rgb_embed(self, rgb, rgb_pad):
        x = self.rgb_proj(rgb)
        x = self.rgb_pos(x)
        h = self.rgb_encoder(x, src_key_padding_mask=rgb_pad)
        return F.normalize(self.rgb_head(_masked_mean(h, rgb_pad)), dim=-1)

    def forward(self, rgb, rgb_pad, tgt):
        r = self.rgb_embed(rgb, rgb_pad)
        t = self.text(tgt)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (r @ t.t())
        labels = torch.arange(r.size(0), device=r.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        acc = (logits.argmax(1) == labels).float().mean()
        return loss, acc


def train_rgb_vlp(cfg: Config, proj=256, temp=0.07):
    dev = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = read_manifest(cfg.train_manifest)
    vocab = Vocab.build([t for _, t in rows], cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    ds = RGBTextDataset(cfg.train_manifest, vocab, cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, collate_fn=collate_rgbtext)
    model = RGBVLPModel(int(cfg.rgb_feature_dim), len(vocab), cfg, proj, temp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9,
                           weight_decay=cfg.weight_decay)
    sched = (WarmupCosine(opt, cfg.lr, cfg.warmup_steps, cfg.epochs * max(1, len(dl)))
             if cfg.lr_schedule == "cosine" else WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps))
    print(f"[rgbvlp] dev={dev} vocab={len(vocab)} rgb_dim={cfg.rgb_feature_dim} "
          f"proj={proj} steps/ep={len(dl)}", flush=True)

    def save(tag):
        # save the full model AND an rgb-stream-only slice for RGBFusionSLT warm-start
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "vlp_proj": proj},
                   os.path.join(cfg.out_dir, f"rgb_vlp_{tag}.pt"))

    best = -1.0
    for ep in range(1, cfg.epochs + 1):
        model.train(); tl = ta = n = 0
        for rgb, rgb_pad, tgt in dl:
            rgb, rgb_pad, tgt = rgb.to(dev), rgb_pad.to(dev), tgt.to(dev)
            loss, acc = model(rgb, rgb_pad, tgt)
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); sched.step()
            b = rgb.size(0); tl += loss.item() * b; ta += acc.item() * b; n += b
        a = ta / max(1, n)
        print(f"[rgbvlp ep {ep:03d}] loss={tl/max(1,n):.4f} train_retr_acc={a:.3f}", flush=True)
        if a >= best:
            best = a; save("best")
    save("final")
    print(f"[rgbvlp done] best in-batch retrieval={best:.3f} -> {cfg.out_dir}/rgb_vlp_best.pt", flush=True)
