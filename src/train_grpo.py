"""GRPO RL fine-tuning for SLT (the 2025 open-domain SOTA mechanism, SLT-GRPO-style).

Fine-tunes a trained supervised translator (PoseSLT / reference model, loaded from a train_ctc
checkpoint) with Group-Relative Policy Optimisation: for each clip, sample G hypotheses, score
each with a BLEU+ROUGE-L reward, compute group-relative advantages ((r-mean)/std), and do a
policy-gradient update with a per-token KL penalty to the frozen reference policy. No value
network (that's GRPO's point). Reuses the model's encode/decode_step, so it works on any of our
pose checkpoints (VLP 10.27, RGB-fusion base, etc.).

  python train_grpo.py --init_ckpt <best.pt> --train_manifest ... --val_manifest ... \
      --test_manifest ... --num_keypoints 133 --keypoint_subset all --coord_norm frame \
      --grpo_group 8 --grpo_beta 0.04 --lr 1e-6 --epochs 3 --device cuda --out_dir runs/grpo
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, PAD, BOS, EOS
from slt.reference_model import build_model
from slt.engine import evaluate, resolve_device, WarmupCosine
import sacrebleu


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F-measure via LCS (dependency-free)."""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if h[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(h)][len(r)]
    prec, rec = lcs / len(h), lcs / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def reward(hyp: str, ref: str) -> float:
    b = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0     # 0..1
    return 0.5 * b + 0.5 * rouge_l(hyp, ref)


@torch.no_grad()
def _ref_token_logp(ref_model, ys, mem, src_pad):
    logits = ref_model.decode_step(ys[:, :-1], mem, src_pad, None)
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)   # (N, L-1)


def sample_group(policy, ref_model, src, src_pad, G, max_len, vocab, temp=1.0):
    """Sample G hypotheses per clip; return decoded text, policy seq-logp, KL-to-ref, valid mask."""
    dev = src.device
    B = src.size(0)
    src_r = src.repeat_interleave(G, 0)
    pad_r = src_pad.repeat_interleave(G, 0)
    mem = policy.encode(src_r, pad_r)                          # (B*G, T, d) grad
    with torch.no_grad():
        mem_ref = ref_model.encode(src_r, pad_r)
    N = B * G
    ys = torch.full((N, 1), BOS, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    step_lp, step_reflp, step_valid = [], [], []
    for _ in range(max_len - 1):
        logits = policy.decode_step(ys, mem, pad_r, None)[:, -1]        # (N,V) grad
        logp = F.log_softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(logp.exp(), 1).squeeze(1)              # (N,)
        tok_lp = logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            rlogits = ref_model.decode_step(ys, mem_ref, pad_r, None)[:, -1]
            rlp = F.log_softmax(rlogits, dim=-1).gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        valid = (~done).float()
        step_lp.append(tok_lp * valid); step_reflp.append(rlp * valid); step_valid.append(valid)
        nxt = nxt.masked_fill(done, PAD)
        ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
        done = done | nxt.eq(EOS)
        if bool(done.all()):
            break
    lp = torch.stack(step_lp, 1)                                        # (N, L)
    reflp = torch.stack(step_reflp, 1)
    valid = torch.stack(step_valid, 1)
    seq_logp = lp.sum(1)                                                # (N,)
    kl = ((lp - reflp) * valid).sum(1) / valid.sum(1).clamp(min=1)      # per-token mean KL est
    texts = [vocab.decode(ys[i].tolist()) for i in range(N)]
    return texts, seq_logp, kl


def main():
    ap = argparse.ArgumentParser()
    cfg0 = Config()
    for name, val in cfg0.to_dict().items():
        if isinstance(val, bool):
            ap.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            ap.add_argument(f"--{name}", type=str, default=None)
        else:
            ap.add_argument(f"--{name}", type=type(val), default=val)
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--grpo_group", type=int, default=8)
    ap.add_argument("--grpo_beta", type=float, default=0.04)
    ap.add_argument("--grpo_temp", type=float, default=1.0)
    ap.add_argument("--limit_samples", type=int, default=0)
    a = ap.parse_args()
    extra = {"init_ckpt", "grpo_group", "grpo_beta", "grpo_temp", "limit_samples"}
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in extra})
    dev = resolve_device(cfg.device); os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
    vocab = Vocab(ck["vocab"])
    policy = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
    policy.load_state_dict(ck["model"])
    ref_model = copy.deepcopy(policy).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print(f"[grpo] loaded {a.init_ckpt} | vocab={len(vocab)} G={a.grpo_group} beta={a.grpo_beta} "
          f"lr={cfg.lr} feat={cfg.feature_dim()}", flush=True)

    def loader(m, shuf, limit=0):
        ds = AuslanDailyDataset(m, vocab, cfg, "")
        if limit:
            ds.samples = ds.samples[:limit]
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuf,
                          num_workers=cfg.num_workers, collate_fn=collate)
    train_dl = loader(cfg.train_manifest, True, a.limit_samples)
    val_dl = loader(cfg.val_manifest, False)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, steps)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    G = a.grpo_group

    # baseline before RL
    b0, w0, _, _ = evaluate(policy, val_dl, vocab, dev, max_len=mx,
                            bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
    best = {"bleu4": b0, "epoch": 0}
    torch.save({"model": policy.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
               os.path.join(cfg.out_dir, "best.pt"))
    print(f"[grpo] pre-RL val_bleu4={b0:.3f}", flush=True)

    for ep in range(1, cfg.epochs + 1):
        policy.train(); tot_r = tot_kl = n = 0
        for src, src_pad, tgt, texts in train_dl:
            src, src_pad = src.to(dev), src_pad.to(dev)
            hyps, seq_logp, kl = sample_group(policy, ref_model, src, src_pad, G, mx, vocab, a.grpo_temp)
            refs = [t for t in texts for _ in range(G)]
            r = torch.tensor([reward(h, rf) for h, rf in zip(hyps, refs)], device=dev)  # (B*G,)
            rg = r.view(-1, G)
            adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
            loss = -(adv.detach() * seq_logp).mean() + a.grpo_beta * kl.mean()
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip); sched.step()
            bs = src.size(0); tot_r += r.mean().item() * bs; tot_kl += kl.mean().item() * bs; n += bs
        bleu, wer, _, _ = evaluate(policy, val_dl, vocab, dev, max_len=mx,
                                   bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        print(f"[grpo ep {ep:02d}] mean_reward={tot_r/n:.3f} mean_kl={tot_kl/n:.3f} "
              f"val_bleu4={bleu:.3f} (pre-RL {b0:.3f})", flush=True)
        if bleu > best["bleu4"]:
            best = {"bleu4": bleu, "epoch": ep}
            torch.save({"model": policy.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
                       os.path.join(cfg.out_dir, "best.pt"))

    ck2 = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev, weights_only=False)
    policy.load_state_dict(ck2["model"])
    if cfg.test_manifest:
        test_dl = loader(cfg.test_manifest, False)
        bleu, wer, hyps, _ = evaluate(policy, test_dl, vocab, dev, max_len=mx,
                                      bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        print(f"[grpo TEST] greedy_bleu4={bleu:.3f} wer={wer:.3f} (pre-RL baseline {b0:.3f}) "
              f"best_ep={best['epoch']}", flush=True)
        json.dump({"pre_rl_val": b0, "best": best, "test_greedy_bleu4": bleu, "test_wer": wer},
                  open(os.path.join(cfg.out_dir, "grpo_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
