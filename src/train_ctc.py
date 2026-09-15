"""Pseudo-gloss CTC anti-collapse experiment (standalone; does not touch shared files).

Adds a CTC head on the encoder, supervised by rule-based content-word pseudo-glosses,
to the exact control translator:  L = CE(decoder) + ctc_weight * CTC(encoder).
CTC forces each encoder timestep to align to a gloss, so the decoder cannot ignore
the encoder (the posterior-collapse failure diagnosed in the deep-dive). Reports test
BLEU + distinct-output count vs the from-scratch control (1.76 greedy / 2.87 beam, 53 unique).
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, collections
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, tokenize, PAD
from slt.reference_model import build_model
from slt.engine import evaluate, resolve_device, WarmupInverseSqrt, WarmupCosine

STOP = set("a an the is are am was were be been being to of in on at and or but i you he she "
           "it we they this that these those my your his her its our their do does did not have "
           "has had will would can could should may might must shall there here then than so as "
           "if for with from up out about into over after . , ? ! ; :".split())


GLOSS_MAP = {}   # PGG-SLT: normalised-text -> [gloss tokens] (LLM pseudo-glosses), if loaded


def _normkey(t):
    return " ".join(t.lower().split())


def to_gloss(text):
    if GLOSS_MAP:                                    # PGG-SLT LLM pseudo-glosses override the rule
        g = GLOSS_MAP.get(_normkey(text))
        if g:
            return g
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


def parse():
    cfg = Config()
    p = argparse.ArgumentParser()
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--ctc_weight", type=float, default=0.3)
    p.add_argument("--pseudo_token_drop", type=float, default=0.0,
                   help="deterministically drop this fraction of rule-based pseudo-content tokens per training epoch")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--swa_last_k", type=int, default=0, help="save last-K epoch ckpts for SWA averaging")
    a = p.parse_args()
    if not 0.0 <= a.pseudo_token_drop < 1.0:
        p.error("--pseudo_token_drop must be in [0, 1)")
    extra = {"ctc_weight", "pseudo_token_drop", "limit_samples", "swa_last_k"}
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in extra})
    return cfg, a


def main():
    cfg, a = parse()
    if getattr(cfg, "rgb_feat_dir", ""):          # Stage B: two-stream pose+RGB fusion
        from slt.rgb_fusion import train_rgb_fusion
        return train_rgb_fusion(cfg, a)
    dev = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    if getattr(cfg, "gloss_map", "") and os.path.exists(cfg.gloss_map):   # PGG-SLT LLM glosses
        for ln in open(cfg.gloss_map, encoding="utf-8"):
            if "\t" in ln:
                k, v = ln.rstrip("\n").split("\t", 1)
                toks = v.split()
                if toks:
                    GLOSS_MAP[_normkey(k)] = toks
        print(f"[pgg] loaded {len(GLOSS_MAP)} LLM pseudo-glosses from {cfg.gloss_map}", flush=True)

    train_txt = [ln.rstrip("\n").split("\t")[1] for ln in open(cfg.train_manifest) if ln.strip()]
    vocab = Vocab.build(train_txt, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    gloss2idx = build_gloss_vocab(train_txt)
    print(f"[ctc] vocab={len(vocab)} gloss_vocab={len(gloss2idx)} device={dev} feat_dim={cfg.feature_dim()}", flush=True)
    # Frequency-aware weighting: upweight pseudo-gloss targets by inverse corpus frequency so the
    # model attends to rare/long-tail content words (targets the unseen-target ~6 ceiling). Off by default.
    gwt = None
    if cfg.ctc_freq_weight > 0:
        from collections import Counter
        gfreq = Counter(g for t in train_txt for g in to_gloss(t))
        gwt = {g: c ** (-cfg.ctc_freq_weight) for g, c in gfreq.items()}
        print(f"[ctc] freq-aware weighting ON (alpha={cfg.ctc_freq_weight}); "
              f"gloss freq {min(gfreq.values())}..{max(gfreq.values())} over {len(gfreq)} glosses", flush=True)

    # Multi-reference paraphrase augmentation: keep only paraphrases whose tokens are
    # all already in the reference vocabulary, so the output space (and therefore the
    # comparison against the control) is unchanged and the decoder never learns to emit UNK.
    PARA_MAP = {}
    if getattr(cfg, "paraphrase_map", "") and os.path.exists(cfg.paraphrase_map):
        known = set(vocab.itos)
        kept = dropped = 0
        for ln in open(cfg.paraphrase_map, encoding="utf-8"):
            if "\t" not in ln:
                continue
            k, v = ln.rstrip("\n").split("\t", 1)
            v = v.strip()
            if not v or _normkey(v) == _normkey(k):
                continue
            if all(t in known for t in tokenize(v, cfg.text_tokenization)):
                PARA_MAP.setdefault(_normkey(k), []).append(v)
                kept += 1
            else:
                dropped += 1
        print(f"[para] {kept} in-vocab paraphrases over {len(PARA_MAP)} sentences "
              f"({dropped} dropped as out-of-vocab), prob={cfg.paraphrase_prob}", flush=True)

    def loader(manifest, shuffle, limit=0, paraphrase=False):
        ds = AuslanDailyDataset(manifest, vocab, cfg, "")
        if limit:
            ds.samples = ds.samples[:limit]
        if paraphrase and PARA_MAP and cfg.paraphrase_prob > 0:
            ds.paraphrases = PARA_MAP
            ds.paraphrase_prob = float(cfg.paraphrase_prob)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate)

    train_dl = loader(cfg.train_manifest, True, a.limit_samples, paraphrase=True)
    val_dl = loader(cfg.val_manifest, False)

    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):   # optional VLP/ISLR pretrained encoder
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck); ref = model.state_dict()
        pref = ("input_proj", "pos_enc", "transformer.encoder", "pose_adapter",
                "spatial_embedding", "source_position", "encoder_layers", "encoder_norm")
        enc = {k: v for k, v in sd.items() if k.startswith(pref) and k in ref and v.shape == ref[k].shape}
        model.load_state_dict(enc, strict=False)
        print(f"[init] loaded {len(enc)} pretrained encoder tensors from {cfg.init_encoder}", flush=True)
    if getattr(cfg, "resume_checkpoint", "") and os.path.exists(cfg.resume_checkpoint):
        # Full-model warm-start (e.g. PoseStitch pre-train -> fine-tune). Shape-filter so a different
        # pre-training vocab (different decoder embed/output dims) doesn't crash the load: the encoder
        # + vocab-independent decoder body transfer; only the vocab-tied layers re-initialise.
        rk = torch.load(cfg.resume_checkpoint, map_location="cpu", weights_only=False)
        sd = rk.get("model", rk); ref = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in ref and v.shape == ref[k].shape}
        model.load_state_dict(keep, strict=False)
        print(f"[resume] full-model warm-start: loaded {len(keep)}/{len(sd)} shape-matched tensors "
              f"from {cfg.resume_checkpoint}", flush=True)
    ctc_head = nn.Linear(cfg.d_model, len(gloss2idx) + 1).to(dev)   # +1 blank(0)
    params = list(model.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = (WarmupCosine(opt, cfg.lr, cfg.warmup_steps, steps) if cfg.lr_schedule == "cosine"
             else WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps))
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ctc_none = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")   # for per-sample freq weighting
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    nparams = sum(p.numel() for p in params) / 1e6
    print(f"[ctc] params={nparams:.2f}M train_batches={len(train_dl)} "
          f"pseudo_token_drop={a.pseudo_token_drop:.3f}", flush=True)

    def gloss_targets(texts, epoch):
        ids, lens, wts = [], [], []
        for t in texts:
            toks = to_gloss(t)
            # This is an auditable robustness ablation, not a claim that English-order
            # content words are Auslan glosses.  Stable per (seed, epoch, sentence)
            # sampling makes its pseudo-target corruption reproducible across workers.
            if a.pseudo_token_drop and len(toks) > 1:
                digest = hashlib.blake2b(
                    f"{cfg.seed}:{epoch}:{t}".encode("utf-8"), digest_size=16
                ).digest()
                rng = random.Random(int.from_bytes(digest, "big"))
                kept = [tok for tok in toks if rng.random() >= a.pseudo_token_drop]
                if not kept:
                    kept = [toks[int.from_bytes(digest[:8], "big") % len(toks)]]
                toks = kept
            g = [gloss2idx.get(x, gloss2idx["<g>"]) for x in toks]
            ids.extend(g); lens.append(len(g))
            if gwt is not None:
                wts.append(sum(gwt.get(x, 1.0) for x in toks) / max(1, len(toks)))
        W = torch.tensor(wts, dtype=torch.float32) if gwt is not None else None
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lens, dtype=torch.long), W

    best = {"bleu4": -1, "wer": 9, "epoch": 0}; history = []
    for ep in range(1, cfg.epochs + 1):
        model.train(); ctc_head.train(); tot = tce = tctc = n = 0
        train_dl.dataset.set_epoch(ep)   # re-draws paraphrase targets each epoch (no-op when off)
        for src, src_pad, tgt, texts in train_dl:
            src, src_pad, tgt = src.to(dev), src_pad.to(dev), tgt.to(dev)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_pad = tgt_in.eq(PAD)
            logits, memory = model(src, src_pad, tgt_in, tgt_pad, return_memory=True)
            loss_ce = ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss_rdrop = torch.zeros((), device=dev)
            if cfg.r_drop_alpha > 0:                       # R-Drop: 2nd dropout pass + symmetric KL
                logits2, _ = model(src, src_pad, tgt_in, tgt_pad, return_memory=True)
                loss_ce = 0.5 * (loss_ce + ce(logits2.reshape(-1, logits2.size(-1)), tgt_out.reshape(-1)))
                lp1, lp2 = F.log_softmax(logits, -1), F.log_softmax(logits2, -1)
                kl = 0.5 * (F.kl_div(lp1, lp2.exp(), reduction="none")
                            + F.kl_div(lp2, lp1.exp(), reduction="none")).sum(-1)
                m = (~tgt_out.eq(PAD)).float()
                loss_rdrop = (kl * m).sum() / m.sum().clamp_min(1.0)
            # CTC on CPU (MPS lacks a CTC kernel); grads flow back to MPS params
            logp = torch.log_softmax(ctc_head(memory), dim=-1).transpose(0, 1).float().cpu()
            in_len = (~src_pad).sum(1).cpu()
            tgt_ids, tgt_len, tgt_w = gloss_targets(texts, ep)
            if tgt_w is not None:                          # freq-aware: per-sample weighted mean
                per = ctc_none(logp, tgt_ids, in_len, tgt_len)      # (B,) on CPU
                w = tgt_w / tgt_w.mean().clamp_min(1e-6)            # normalise to mean 1 (keeps scale)
                loss_ctc = (per * w).mean()
            else:
                loss_ctc = ctc(logp, tgt_ids, in_len, tgt_len)
            loss = loss_ce + a.ctc_weight * loss_ctc.to(dev) + cfg.r_drop_alpha * loss_rdrop
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); sched.step()
            b = src.size(0); tot += loss.item() * b; tce += loss_ce.item() * b
            tctc += float(loss_ctc.detach()) * b; n += b
        bleu, wer, hyps, _ = evaluate(model, val_dl, vocab, dev, max_len=mx,
                                      bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        uniq = len(set(hyps))
        print(f"[ctc ep {ep:03d}] loss={tot/n:.3f} ce={tce/n:.3f} ctc={tctc/n:.3f} "
              f"val_bleu4={bleu:.3f} wer={wer:.3f} uniq={uniq}", flush=True)
        history.append({"epoch": ep, "bleu4": bleu, "wer": wer, "uniq": uniq})
        if bleu > best["bleu4"]:
            best = {"bleu4": bleu, "wer": wer, "epoch": ep}
            torch.save({"model": model.state_dict(), "ctc_head": ctc_head.state_dict(),
                        "vocab": vocab.itos, "cfg": cfg.to_dict(), "gloss2idx": gloss2idx},
                       os.path.join(cfg.out_dir, "best.pt"))
        if a.swa_last_k and ep > cfg.epochs - a.swa_last_k:    # SWA: save last-K epoch ckpts
            torch.save({"model": model.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
                       os.path.join(cfg.out_dir, f"swa_ep{ep}.pt"))
        json.dump({"status": "running", "epoch": ep, "epochs": cfg.epochs,
                   "best": best, "history": history,
                   "pseudo_token_drop": a.pseudo_token_drop,
                   "pseudo_target_order": "english_spoken_order"},
                  open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)

    # test on best
    ck = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    if cfg.test_manifest:
        test_dl = loader(cfg.test_manifest, False)
        bleu, wer, hyps, _ = evaluate(model, test_dl, vocab, dev, max_len=mx,
                                      bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        test = {"bleu4": bleu, "wer": wer, "uniq": len(set(hyps)), "from_epoch": best["epoch"]}
        print(f"[ctc TEST] bleu4={bleu:.3f} wer={wer:.3f} distinct_outputs={len(set(hyps))}", flush=True)
        prog = json.load(open(os.path.join(cfg.out_dir, "progress.json")))
        prog["test"] = test
    else:
        prog = json.load(open(os.path.join(cfg.out_dir, "progress.json")))
    prog["status"] = "done"
    prog["test_set_used"] = bool(cfg.test_manifest)
    json.dump(prog, open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
