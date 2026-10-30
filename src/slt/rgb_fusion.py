"""RGB fusion — Stage B: two-stream pose + RGB translator (self-contained).

Adds an RGB appearance stream to the DWPose pose translator and fuses it into the
existing SignJoey/generic decoder, keeping the pseudo-gloss CTC head on the POSE
stream (same anti-collapse objective as train_ctc.py). Everything RGB-specific
lives here so slt/data.py, slt/engine.py and the live pose path stay untouched;
train_ctc.py only dispatches here when --rgb_feat_dir is set.

Pose stream : (T,399) DWPose feats -> base.encode           -> mem_p (B,Tp,d)
RGB stream  : (Tr,1152) DINOv2 crop feats -> proj+2L Transf  -> mem_r (B,Tr,d)
Fusion      : + learned modality tag, concat along time      -> mem_f (B,Tp+Tr,d)
Decode      : base.decode_step(tgt, mem_f, fused_pad, ...)    -> logits
CTC         : ctc_head(mem_p) supervised by content-word pseudo-glosses.

Loss = CE(decoder over fused memory) + ctc_weight * CTC(pose encoder).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from slt.data import AuslanDailyDataset, Vocab, PAD, BOS, EOS, tokenize
from slt.reference_model import build_model, SinusoidalPosition
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device
from slt.metrics import corpus_bleu4, corpus_wer, detokenize


# ── pseudo-gloss (identical rule to train_ctc.py; duplicated to stay self-contained) ──
STOP = set("a an the is are am was were be been being to of in on at and or but i you he she "
           "it we they this that these those my your his her its our their do does did not have "
           "has had will would can could should may might must shall there here then than so as "
           "if for with from up out about into over after . , ? ! ; :".split())


def to_gloss(text):
    ws = [w for w in tokenize(text, "whitespace") if w.isalpha() and w not in STOP]
    out = []
    for w in ws:
        if not out or out[-1] != w:
            out.append(w)
    return out or ["<g>"]


def build_gloss_vocab(texts):
    toks = set(["<g>"])
    for t in texts:
        toks.update(to_gloss(t))
    return {g: i + 1 for i, g in enumerate(sorted(toks))}   # 0 = CTC blank


# ── RGB-aware dataset / collate ───────────────────────────────────────────────
def _rgb_path(clip, rgb_dir):
    cid = os.path.basename(clip)
    if cid.endswith(".npz"):
        cid = cid[:-4]
    return os.path.join(rgb_dir, f"rgb_{cid}.npz")


def _face_path(clip, face_dir):
    cid = os.path.basename(clip)
    if cid.endswith(".npz"):
        cid = cid[:-4]
    return os.path.join(face_dir, f"face_{cid}.npz")


class RGBDataset(AuslanDailyDataset):
    """AuslanDailyDataset + per-clip precomputed RGB features (rgb_<clip>.npz FEATURES)."""

    def __init__(self, manifest, vocab, cfg, root=""):
        super().__init__(manifest, vocab, cfg, root)
        self.rgb_dir = cfg.rgb_feat_dir
        self.rgb_dim = int(cfg.rgb_feature_dim)
        self.face_dir = str(getattr(cfg, "face_feat_dir", ""))
        self.face_dim = int(getattr(cfg, "face_feature_dim", 0))

    def _load_feature(self, path, dimension):
        if os.path.exists(path):
            try:
                with np.load(path) as z:
                    feature = np.asarray(z["FEATURES"], dtype=np.float32)
                if feature.ndim == 2 and feature.shape[1] == dimension and feature.shape[0] >= 1 \
                        and np.isfinite(feature).all():
                    if feature.shape[0] > self.cfg.max_src_frames:
                        selected = np.linspace(0, feature.shape[0] - 1, self.cfg.max_src_frames).round().astype(int)
                        feature = feature[selected]
                    return feature
            except Exception:
                pass
        return None

    def _load_rgb(self, clip):
        rgb = self._load_feature(_rgb_path(clip, self.rgb_dir), self.rgb_dim)
        if rgb is not None:
            if self.face_dir and self.face_dim:
                face = self._load_feature(_face_path(clip, self.face_dir), self.face_dim)
                # Both banks are sampled at 24 frames, but align defensively so a
                # shorter source clip or a recoverable missing face file does not
                # change the temporal contract of the established RGB stream.
                if face is None:
                    face = np.zeros((rgb.shape[0], self.face_dim), np.float32)
                elif face.shape[0] != rgb.shape[0]:
                    selected = np.linspace(0, face.shape[0] - 1, rgb.shape[0]).round().astype(int)
                    face = face[selected]
                return np.concatenate([rgb, face], axis=1)
            return rgb
        # missing/invalid RGB: one zero frame so attention memory is never empty
        width = self.rgb_dim + (self.face_dim if self.face_dir else 0)
        return np.zeros((1, width), np.float32)

    def __getitem__(self, i):
        pose_feats, tgt, text = super().__getitem__(i)
        clip = self.samples[i][0]
        rgb = torch.from_numpy(self._load_rgb(clip))
        return pose_feats, rgb, tgt, text


def collate_rgb(batch, pad_id=PAD):
    poses, rgbs, tgts, texts = zip(*batch)
    B = len(poses)
    # pose
    Tp = max(f.shape[0] for f in poses)
    Fp = poses[0].shape[1]
    src = torch.zeros(B, Tp, Fp, dtype=torch.float32)
    src_pad = torch.ones(B, Tp, dtype=torch.bool)
    for i, f in enumerate(poses):
        t = f.shape[0]
        src[i, :t] = f
        src_pad[i, :t] = False
    # rgb
    Tr = max(f.shape[0] for f in rgbs)
    Fr = rgbs[0].shape[1]
    rgb = torch.zeros(B, Tr, Fr, dtype=torch.float32)
    rgb_pad = torch.ones(B, Tr, dtype=torch.bool)
    for i, f in enumerate(rgbs):
        t = f.shape[0]
        rgb[i, :t] = f
        rgb_pad[i, :t] = False
    # target
    Lmax = max(t.shape[0] for t in tgts)
    tgt = torch.full((B, Lmax), pad_id, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt[i, :t.shape[0]] = t
    return src, src_pad, rgb, rgb_pad, tgt, list(texts)


# ── fusion model ──────────────────────────────────────────────────────────────
class RGBFusionSLT(nn.Module):
    def __init__(self, feat_dim, rgb_dim, vocab_size, cfg):
        super().__init__()
        self.base = build_model(feat_dim, vocab_size, cfg)   # PoseSLT or ReferencePoseSLT
        d = cfg.d_model
        self.d_model = d
        self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, d), nn.LayerNorm(d))
        self.rgb_pos = SinusoidalPosition(d, cfg.position_max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        n_rgb = int(getattr(cfg, "rgb_encoder_layers", 2))
        self.rgb_encoder = nn.TransformerEncoder(layer, num_layers=n_rgb)
        for _attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(self.rgb_encoder, _attr):
                setattr(self.rgb_encoder, _attr, False)
        self.modality = nn.Parameter(torch.zeros(2, d))      # [pose, rgb] learned tags
        nn.init.normal_(self.modality, std=0.02)

    def encode_rgb(self, rgb, rgb_pad):
        x = self.rgb_proj(rgb)
        x = self.rgb_pos(x)
        return self.rgb_encoder(x, src_key_padding_mask=rgb_pad)

    def _fuse(self, mem_p, src_pad, mem_r, rgb_pad):
        mem_p = mem_p + self.modality[0].view(1, 1, -1)
        mem_r = mem_r + self.modality[1].view(1, 1, -1)
        mem = torch.cat([mem_p, mem_r], dim=1)               # (B, Tp+Tr, d)
        pad = torch.cat([src_pad, rgb_pad], dim=1)           # True = ignore
        return mem, pad

    def forward(self, src, src_pad, rgb, rgb_pad, tgt_in, tgt_pad, return_memory=False):
        mem_p = self.base.encode(src, src_pad)
        mem_r = self.encode_rgb(rgb, rgb_pad)
        mem_f, pad_f = self._fuse(mem_p, src_pad, mem_r, rgb_pad)
        logits = self.base.decode_step(tgt_in, mem_f, pad_f, tgt_pad)
        return (logits, mem_p) if return_memory else logits   # CTC stays on pose memory

    @torch.no_grad()
    def greedy_decode(self, src, src_pad, rgb, rgb_pad, max_len=64):
        self.eval()
        B = src.size(0)
        mem_p = self.base.encode(src, src_pad)
        mem_r = self.encode_rgb(rgb, rgb_pad)
        mem_f, pad_f = self._fuse(mem_p, src_pad, mem_r, rgb_pad)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.base.decode_step(ys, mem_f, pad_f, None)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            nxt[done] = PAD
            ys = torch.cat([ys, nxt], dim=1)
            done |= nxt.squeeze(1).eq(EOS)
            if bool(done.all()):
                break
        return ys

    @torch.no_grad()
    def beam_decode(self, src, src_pad, rgb, rgb_pad, max_len=64, beam=5, length_penalty=0.6):
        self.eval()
        device = src.device
        out = []
        for b in range(src.size(0)):
            mem_p = self.base.encode(src[b:b + 1], src_pad[b:b + 1])
            mem_r = self.encode_rgb(rgb[b:b + 1], rgb_pad[b:b + 1])
            mem_f, pad_f = self._fuse(mem_p, src_pad[b:b + 1], mem_r, rgb_pad[b:b + 1])
            mem_f = mem_f.expand(beam, -1, -1).contiguous()
            pad_f = pad_f.expand(beam, -1).contiguous()
            ys = torch.full((beam, 1), BOS, dtype=torch.long, device=device)
            scores = torch.full((beam,), float("-inf"), device=device)
            scores[0] = 0.0
            finished = []
            for _ in range(max_len - 1):
                logp = torch.log_softmax(self.base.decode_step(ys, mem_f, pad_f, None)[:, -1], dim=-1)
                V = logp.size(-1)
                cand = (scores.unsqueeze(1) + logp).view(-1)
                top_scores, top_idx = cand.topk(beam)
                bidx = torch.div(top_idx, V, rounding_mode="floor")
                tidx = top_idx % V
                ys = torch.cat([ys[bidx], tidx.unsqueeze(1)], dim=1)
                scores = top_scores.clone()
                for i in range(beam):
                    if tidx[i].item() == EOS:
                        seq = ys[i].tolist()
                        finished.append((scores[i].item() / (max(1, len(seq)) ** length_penalty), seq))
                        scores[i] = float("-inf")
                if len(finished) >= beam:
                    break
            if not finished:
                finished.append((0.0, ys[int(scores.argmax())].tolist()))
            finished.sort(key=lambda x: x[0], reverse=True)
            out.append(finished[0][1])
        return out


# ── eval / train ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_fusion(model, loader, vocab, device, max_len=64, bleu_tokenize="13a",
                    detok=True, beam=0, length_penalty=0.6):
    model.eval()
    hyps, refs = [], []
    for src, src_pad, rgb, rgb_pad, _tgt, texts in loader:
        src, src_pad, rgb, rgb_pad = src.to(device), src_pad.to(device), rgb.to(device), rgb_pad.to(device)
        if beam and beam > 1:
            rows = model.beam_decode(src, src_pad, rgb, rgb_pad, max_len=max_len,
                                     beam=beam, length_penalty=length_penalty)
        else:
            rows = model.greedy_decode(src, src_pad, rgb, rgb_pad, max_len=max_len).tolist()
        for row, ref in zip(rows, texts):
            hyps.append(vocab.decode(row))
            refs.append(ref)
    h = [detokenize(x) for x in hyps] if detok else hyps
    r = [detokenize(x) for x in refs] if detok else refs
    return corpus_bleu4(h, r, tokenize=bleu_tokenize), corpus_wer(h, r), hyps, refs


def train_rgb_fusion(cfg: Config, a):
    dev = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    train_txt = [ln.rstrip("\n").split("\t")[1] for ln in open(cfg.train_manifest) if ln.strip()]
    vocab = Vocab.build(train_txt, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    gloss2idx = build_gloss_vocab(train_txt)
    print(f"[rgbfuse] vocab={len(vocab)} gloss={len(gloss2idx)} dev={dev} "
          f"pose_feat={cfg.feature_dim()} rgb_feat={cfg.rgb_feature_dim} rgb_dir={cfg.rgb_feat_dir}",
          flush=True)

    def loader(manifest, shuffle, limit=0):
        ds = RGBDataset(manifest, vocab, cfg, "")
        if limit:
            ds.samples = ds.samples[:limit]
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate_rgb)

    limit = getattr(a, "limit_samples", 0)
    ctc_weight = getattr(a, "ctc_weight", 0.3)
    train_dl = loader(cfg.train_manifest, True, limit)
    val_dl = loader(cfg.val_manifest, False)

    model = RGBFusionSLT(cfg.feature_dim(), int(cfg.rgb_feature_dim), len(vocab), cfg).to(dev)
    # Warm-start the pose+decoder stack from a trained pose checkpoint (e.g. the
    # 9.6 DWPose+CTC best): RGB stream then only has to add appearance detail.
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        ref = model.base.state_dict()
        keep = {k: v for k, v in sd.items() if k in ref and v.shape == ref[k].shape}
        model.base.load_state_dict(keep, strict=False)
        print(f"[init] warm-started base with {len(keep)}/{len(ref)} tensors from {cfg.init_encoder}",
              flush=True)
    # Warm-start the RGB stream from an RGB-VLP checkpoint (text-grounded appearance encoder).
    if getattr(cfg, "rgb_vlp_init", "") and os.path.exists(cfg.rgb_vlp_init):
        rck = torch.load(cfg.rgb_vlp_init, map_location="cpu", weights_only=False)
        rsd = rck.get("model", rck); mref = model.state_dict()
        rpref = ("rgb_proj", "rgb_pos", "rgb_encoder")
        rkeep = {k: v for k, v in rsd.items() if k.startswith(rpref) and k in mref and v.shape == mref[k].shape}
        model.load_state_dict(rkeep, strict=False)
        print(f"[init] warm-started RGB stream with {len(rkeep)} tensors from {cfg.rgb_vlp_init}", flush=True)

    ctc_head = nn.Linear(cfg.d_model, len(gloss2idx) + 1).to(dev)
    params = list(model.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = (WarmupCosine(opt, cfg.lr, cfg.warmup_steps, steps) if cfg.lr_schedule == "cosine"
             else WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps))
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    print(f"[rgbfuse] params={sum(p.numel() for p in params)/1e6:.2f}M batches={len(train_dl)}", flush=True)

    def gloss_targets(texts):
        ids, lens = [], []
        for t in texts:
            g = [gloss2idx.get(x, gloss2idx["<g>"]) for x in to_gloss(t)]
            ids.extend(g); lens.append(len(g))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lens, dtype=torch.long)

    best = {"bleu4": -1, "wer": 9, "epoch": 0}; history = []
    for ep in range(1, cfg.epochs + 1):
        model.train(); ctc_head.train(); tot = tce = tctc = n = 0
        for src, src_pad, rgb, rgb_pad, tgt, texts in train_dl:
            src, src_pad = src.to(dev), src_pad.to(dev)
            rgb, rgb_pad, tgt = rgb.to(dev), rgb_pad.to(dev), tgt.to(dev)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_pad = tgt_in.eq(PAD)
            logits, mem_p = model(src, src_pad, rgb, rgb_pad, tgt_in, tgt_pad, return_memory=True)
            loss_ce = ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            logp = torch.log_softmax(ctc_head(mem_p), dim=-1).transpose(0, 1).float().cpu()
            in_len = (~src_pad).sum(1).cpu()
            tgt_ids, tgt_len = gloss_targets(texts)
            loss_ctc = ctc(logp, tgt_ids, in_len, tgt_len)
            loss = loss_ce + ctc_weight * loss_ctc.to(dev)
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); sched.step()
            b = src.size(0); tot += loss.item() * b; tce += loss_ce.item() * b
            tctc += float(loss_ctc.detach()) * b; n += b
        bleu, wer, hyps, _ = evaluate_fusion(model, val_dl, vocab, dev, max_len=mx,
                                             bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        uniq = len(set(hyps))
        print(f"[rgbfuse ep {ep:03d}] loss={tot/n:.3f} ce={tce/n:.3f} ctc={tctc/n:.3f} "
              f"val_bleu4={bleu:.3f} wer={wer:.3f} uniq={uniq}", flush=True)
        history.append({"epoch": ep, "bleu4": bleu, "wer": wer, "uniq": uniq})
        if bleu > best["bleu4"]:
            best = {"bleu4": bleu, "wer": wer, "epoch": ep}
            torch.save({"model": model.state_dict(), "ctc_head": ctc_head.state_dict(),
                        "vocab": vocab.itos, "cfg": cfg.to_dict(), "gloss2idx": gloss2idx},
                       os.path.join(cfg.out_dir, "best.pt"))
        if getattr(a, "swa_last_k", 0) and ep > cfg.epochs - a.swa_last_k:   # SWA last-K ckpts
            torch.save({"model": model.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
                       os.path.join(cfg.out_dir, f"swa_ep{ep}.pt"))
        json.dump({"status": "running", "epoch": ep, "epochs": cfg.epochs,
                   "best": best, "history": history},
                  open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)

    # test on best (greedy + beam-5)
    ck = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    if cfg.test_manifest:
        test_dl = loader(cfg.test_manifest, False)
        g_bleu, g_wer, g_hyps, _ = evaluate_fusion(model, test_dl, vocab, dev, max_len=mx,
                                                   bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        b_bleu, b_wer, b_hyps, _ = evaluate_fusion(model, test_dl, vocab, dev, max_len=mx,
                                                   bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize,
                                                   beam=cfg.num_beams, length_penalty=cfg.length_penalty)
        test = {"greedy_bleu4": g_bleu, "greedy_wer": g_wer, "greedy_uniq": len(set(g_hyps)),
                "beam_bleu4": b_bleu, "beam_wer": b_wer, "beam_uniq": len(set(b_hyps)),
                "beam": cfg.num_beams, "from_epoch": best["epoch"]}
        print(f"[rgbfuse TEST] greedy_bleu4={g_bleu:.3f} (uniq {len(set(g_hyps))}) | "
              f"beam{cfg.num_beams}_bleu4={b_bleu:.3f} (uniq {len(set(b_hyps))}) wer={b_wer:.3f}", flush=True)
        prog = json.load(open(os.path.join(cfg.out_dir, "progress.json")))
        prog["status"] = "done"; prog["test"] = test
        json.dump(prog, open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)
    return best
