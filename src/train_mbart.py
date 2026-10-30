"""Train the pose-encoder -> adapter -> mBART (BPE) decoder (GFSLT-VLP-style).

Warm-starts the pose encoder from our VLP/SWA checkpoint (--init_encoder), tokenises targets with
mBART's subword tokenizer, fine-tunes adapter + pose encoder + mBART-LoRA. Reports sacreBLEU 13a.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import read_manifest, load_pose, normalize_clip
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from slt.engine import resolve_device, WarmupCosine
from slt.mbart_decoder import build_pose_mbart


class PoseTextDS(Dataset):
    def __init__(self, manifest, cfg, root=""):
        self.rows = read_manifest(manifest); self.cfg = cfg; self.root = root
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        clip, text = self.rows[i]
        path = os.path.join(self.root, clip) if self.root else clip
        kps, bbox = load_pose(path)
        feats, _ = normalize_clip(kps, bbox, self.cfg)
        return torch.from_numpy(feats), text


def make_collate(tokenizer, max_tgt):
    def collate(batch):
        feats, texts = zip(*batch)
        B = len(feats); T = max(f.shape[0] for f in feats); F = feats[0].shape[1]
        src = torch.zeros(B, T, F); pad = torch.ones(B, T, dtype=torch.bool)
        for i, f in enumerate(feats):
            src[i, :f.shape[0]] = f; pad[i, :f.shape[0]] = False
        lab = tokenizer(list(texts), max_length=max_tgt, truncation=True,
                        padding="longest", return_tensors="pt").input_ids
        lab[lab == tokenizer.pad_token_id] = -100          # ignore pad in loss
        return src, pad, lab
    return collate


def parse():
    cfg = Config(); p = argparse.ArgumentParser()
    for n, v in cfg.to_dict().items():
        if isinstance(v, bool): p.add_argument(f"--{n}", type=lambda x: x.lower() == "true", default=v)
        elif v is None: p.add_argument(f"--{n}", type=str, default=None)
        else: p.add_argument(f"--{n}", type=type(v), default=v)
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=2)
    a = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in ("limit_samples", "grad_accum")})
    return cfg, a


@torch.no_grad()
def evaluate(model, dl, tokenizer, dev, mx, beam):
    model.eval(); hyps, refs = [], []
    for src, pad, _lab in dl:
        src, pad = src.to(dev), pad.to(dev)
        gen = model.generate(src, pad, max_len=mx, num_beams=beam)
        hyps += tokenizer.batch_decode(gen, skip_special_tokens=True)
    # refs from manifest order
    return hyps


def main():
    cfg, a = parse()
    dev = resolve_device(cfg.device); os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.mbart_name, src_lang="en_XX", tgt_lang="en_XX")

    def loader(m, shuf, limit=0):
        ds = PoseTextDS(m, cfg)
        if limit: ds.rows = ds.rows[:limit]
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuf, num_workers=cfg.num_workers,
                          collate_fn=make_collate(tok, cfg.max_tgt_tokens))
    train_dl = loader(cfg.train_manifest, True, a.limit_samples)
    val_rows = read_manifest(cfg.val_manifest)
    val_dl = loader(cfg.val_manifest, False)

    model = build_pose_mbart(cfg.feature_dim(), cfg, tok).to(dev)
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):        # warm-start pose encoder
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck); ref = model.base.state_dict()
        pref = ("input_proj", "pos_enc", "transformer.encoder", "pose_adapter",
                "spatial_embedding", "source_position", "encoder_layers", "encoder_norm")
        enc = {k: v for k, v in sd.items() if k.startswith(pref) and k in ref and v.shape == ref[k].shape}
        model.base.load_state_dict(enc, strict=False)
        print(f"[mbart] warm-started pose encoder: {len(enc)} tensors from {cfg.init_encoder}", flush=True)
    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):     # Stage 2b: continue from ALIGN phase
        rk = torch.load(cfg.resume_checkpoint, map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(rk.get("model", rk), strict=False)
        print(f"[mbart] resumed from {cfg.resume_checkpoint} (missing {len(miss)}, unexpected {len(unexp)})", flush=True)
    ntr = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"[mbart] {cfg.mbart_name} d={model.d_mbart} trainable={ntr:.1f}M device={dev}", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.llm_lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = WarmupCosine(opt, cfg.llm_lr, cfg.warmup_steps, steps)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    best = {"bleu4": -1, "epoch": 0}

    for ep in range(1, cfg.epochs + 1):
        model.train(); tot = n = 0; opt.zero_grad()
        for step, (src, pad, lab) in enumerate(train_dl, 1):
            src, pad, lab = src.to(dev), pad.to(dev), lab.to(dev)
            loss = model(src, pad, lab).loss / a.grad_accum
            loss.backward()
            if step % a.grad_accum == 0 or step == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
                sched.step(); opt.zero_grad()
            tot += loss.item() * a.grad_accum * src.size(0); n += src.size(0)
        hyps = evaluate(model, val_dl, tok, dev, mx, cfg.num_beams)
        refs = [detokenize(t) for _, t in val_rows]
        hyp_d = [detokenize(h) for h in hyps]
        bleu = corpus_bleu4(hyp_d, refs, tokenize=cfg.bleu_tokenize)
        print(f"[mbart ep {ep:02d}] loss={tot/max(1,n):.3f} val_bleu4={bleu:.3f} "
              f"uniq={len(set(hyps))}", flush=True)
        if bleu > best["bleu4"]:
            best = {"bleu4": bleu, "epoch": ep}
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
                       os.path.join(cfg.out_dir, "best.pt"))
        json.dump({"best": best, "epoch": ep}, open(os.path.join(cfg.out_dir, "progress.json"), "w"))

    # test on best
    if cfg.test_manifest:
        ck = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        test_rows = read_manifest(cfg.test_manifest); test_dl = loader(cfg.test_manifest, False)
        hyps = evaluate(model, test_dl, tok, dev, mx, cfg.num_beams)
        refs = [detokenize(t) for _, t in test_rows]; hyp_d = [detokenize(h) for h in hyps]
        bleu = corpus_bleu4(hyp_d, refs, tokenize=cfg.bleu_tokenize); wer = corpus_wer(hyp_d, refs)
        print(f"[mbart TEST] beam{cfg.num_beams}_bleu4={bleu:.3f} wer={wer:.3f} "
              f"uniq={len(set(hyps))} from_ep={best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
