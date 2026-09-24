"""Per-clip visual grounding: does the model's USE of the video predict whether it gets that clip right?

Two results now sit in tension.
  * The pose-reliance probe (2026-09-05) showed the models DO use their visual input: replacing a clip's
    pose with another clip's raises teacher-forced CE by 0.79-0.93 nats, corpus-wide.
  * The input-side predictor analysis (2026-09-09) found NO measurable property of the video predicts
    per-clip translation quality -- 11 kinematic/estimator descriptors all inside the null band, and a
    video-grouped multivariate model at AUC 0.518 against its own shuffled control's 0.519.

They reconcile only if the model extracts something aggregate kinematics do not capture. That is
testable: measure reliance PER CLIP rather than corpus-wide, and ask whether it predicts success.

  reliance(i) = mean_j CE(target_i | pose_j) - CE(target_i | pose_i),  j != i

A positive correlation would mean visual grounding is real, VARIABLE across clips, and predictive --
a confidence signal derived from the model rather than from descriptors, which is exactly the thing the
descriptor analysis failed to find. A null would mean the model's use of pose is uniform: present on
average, but carrying no per-clip information about whether the output is right.

Controls: a shuffled-outcome null band, and CE(true) itself as a positive control -- per-clip CE MUST
predict per-clip quality (both measure whether the model likes the reference), so if it does not, the
harness is broken and any null here is uninterpretable.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config                                            # noqa: E402
from slt.data import PAD, AuslanDailyDataset, Vocab, collate         # noqa: E402
from slt.reference_model import build_model                          # noqa: E402
from slt.metrics import detokenize                                   # noqa: E402
from tools.stitch_lexicon import content_words                       # noqa: E402


@torch.no_grad()
def per_clip_ce(model, ds, device, idx_pairs, batch_size=16):
    """CE (summed over target tokens, and token count) for a list of (target_idx, pose_idx) pairs."""
    out = np.zeros(len(idx_pairs)), np.zeros(len(idx_pairs))
    ce_sum, ntok = out
    for b in range(0, len(idx_pairs), batch_size):
        chunk = idx_pairs[b:b + batch_size]
        # target side from ti, pose side from pi -- built by taking each sample and swapping its pose
        items = []
        for ti, pi in chunk:
            src_t, tgt_t, txt_t = ds[ti][0], ds[ti][1], ds[ti][2]
            src_p = ds[pi][0]
            items.append((src_p, tgt_t, txt_t))
        src, src_pad, tgt, _ = collate(items)
        src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        mem = model.encode(src, src_pad)
        logits = model.decode_step(tgt_in, mem, src_pad, tgt_in == PAD)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                             ignore_index=PAD, reduction="none").reshape(tgt_out.shape)
        ce_sum[b:b + len(chunk)] = ce.sum(1).float().cpu().numpy()
        ntok[b:b + len(chunk)] = (tgt_out != PAD).sum(1).float().cpu().numpy()
    return ce_sum, ntok


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--manifest", default="comm_run/manifests/test.tsv")
    ap.add_argument("--clip_dir", default="results/test_clips_dwpose")
    ap.add_argument("--k", type=int, default=8, help="random donor poses per clip")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    # rewrite the manifest to point at the local clip copies
    rows = [l.rstrip("\n").split("\t") for l in open(root / a.manifest)]
    # Name the rewritten manifest after the SPLIT IT CAME FROM. The old fixed name
    # "_local_test_manifest.tsv" silently repurposed itself when run on validation, which is one
    # locked-evaluation mistake away from reporting a validation score as an official test score.
    local = root / "results" / f"_local_{Path(a.manifest).stem}_manifest.tsv"
    local.write_text("".join(f"{root / a.clip_dir / Path(r[0]).name}\t{r[1]}\n" for r in rows))

    ck = torch.load(root / a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(str(local), vocab, cfg)
    dev = torch.device(a.device)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev).eval()
    missing = model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)
    print(f"loaded {a.ckpt}: {len(ds)} clips, vocab {len(vocab)}, missing={len(missing.missing_keys)}")

    n = len(ds)
    rng = np.random.default_rng(a.seed)
    ce_true, ntok = per_clip_ce(model, ds, dev, [(i, i) for i in range(n)])
    ce_sh = np.zeros((a.k, n))
    for r in range(a.k):
        donors = (np.arange(n) + rng.integers(1, n, size=n)) % n     # random j != i
        ce_sh[r], _ = per_clip_ce(model, ds, dev, list(zip(range(n), donors)))
        print(f"  donor round {r + 1}/{a.k} done", flush=True)
    ce_true_pt = ce_true / np.maximum(ntok, 1)
    ce_shuf_pt = (ce_sh / np.maximum(ntok, 1)).mean(0)
    reliance = ce_shuf_pt - ce_true_pt
    print(f"\ncorpus-level: CE(true) {ce_true.sum()/ntok.sum():.4f}  "
          f"CE(shuffled) {ce_sh.sum(1).mean()/ntok.sum():.4f}  "
          f"reliance {ce_sh.sum(1).mean()/ntok.sum() - ce_true.sum()/ntok.sum():.4f}")
    print(f"per-clip reliance: mean {reliance.mean():.3f}  sd {reliance.std():.3f}  "
          f"p10 {np.percentile(reliance,10):.3f}  p90 {np.percentile(reliance,90):.3f}  "
          f"negative on {100*(reliance<0).mean():.1f}% of clips")

    import sacrebleu
    recs = [json.loads(l) for l in open(root / a.hyps)]
    assert all(r["ref"].strip() == t[1].strip() for r, t in zip(recs, rows)), "ref mismatch"
    y = {}
    y["content_recall"] = np.array([
        np.mean([w in set(content_words(r["hyp_beam"])) for w in content_words(r["ref"])])
        if content_words(r["ref"]) else np.nan for r in recs])
    y["any_content"] = np.array([
        float(any(w in set(content_words(r["hyp_beam"])) for w in content_words(r["ref"])))
        if content_words(r["ref"]) else np.nan for r in recs])
    y["chrf"] = np.array([sacrebleu.sentence_chrf(detokenize(r["hyp_beam"]),
                                                  [detokenize(r["ref"])]).score for r in recs])
    y["exact"] = np.array([float(detokenize(r["hyp_beam"]).strip()
                                 == detokenize(r["ref"]).strip()) for r in recs])

    preds = {"reliance (shuf - true)": reliance,
             "[+ctrl] CE(true) per token": -ce_true_pt,     # negated: lower CE should mean better output
             "[+ctrl] CE(shuffled) per tok": -ce_shuf_pt,
             "[ctrl] n_target_tokens": ntok.astype(float)}
    print(f"\n{'predictor':32}" + "".join(f"{k:>16}" for k in y))
    rep = {}
    bands = {}
    for k, yy in y.items():
        mx = []
        for _ in range(a.perm):
            yp = yy.copy(); rng.shuffle(yp)
            mx.append(max(abs(corr(v, yp)) for v in preds.values()))
        bands[k] = float(np.percentile(mx, 95))
    for pn, pv in preds.items():
        line = f"{pn:32}"
        for k, yy in y.items():
            r = corr(pv, yy); line += f"{r:>15.3f}{'*' if abs(r) > bands[k] else ' '}"
            rep.setdefault(pn, {})[k] = r
        print(line)
    print(f"{'NULL BAND (p95 max|r|)':32}" + "".join(f"{bands[k]:>16.3f}" for k in y))
    print("  * = exceeds the family-wise shuffled-outcome null band")

    if a.out:
        (root / a.out).write_text(json.dumps(
            {"ckpt": a.ckpt, "n": n, "k": a.k, "corr": rep, "null_band": bands,
             "reliance_mean": float(reliance.mean()), "reliance_sd": float(reliance.std()),
             "reliance_neg_frac": float((reliance < 0).mean()),
             "reliance": reliance.tolist(), "ce_true_pt": ce_true_pt.tolist()}, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
