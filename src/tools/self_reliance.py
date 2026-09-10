"""Two follow-ups to per_clip_reliance.py.

(1) PARTIAL CORRELATION. reliance = CE(shuffled) - CE(true) literally contains -CE(true), and CE(true)
    alone correlates ~0.46 with quality (a model that assigns the reference low CE tends to produce it).
    So the raw 0.474 could be entirely CE(true) wearing a different hat. Partial correlation settles it.

(2) SELF-RELIANCE, which unlike reliance IS available at inference time. Reliance needs the reference.
    Substituting the model's OWN beam output for the reference gives
        self_reliance(i) = CE(hyp_i | pose_j) - CE(hyp_i | pose_i)
    which needs nothing but the clip and the model. If that predicts quality, the confidence signal the
    descriptor analysis failed to find exists after all -- just derived from the model, not the pixels.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config                                            # noqa: E402
from slt.data import PAD, AuslanDailyDataset, Vocab, collate         # noqa: E402
from slt.reference_model import build_model                          # noqa: E402
from slt.metrics import detokenize                                   # noqa: E402
from tools.stitch_lexicon import content_words                       # noqa: E402
from tools.per_clip_reliance import per_clip_ce, corr                # noqa: E402


def partial(x, y, z):
    """corr(x, y) with z linearly removed from both."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--reliance_json", required=True)
    ap.add_argument("--manifest", default="comm_run/manifests/test.tsv")
    ap.add_argument("--clip_dir", default="results/test_clips_dwpose")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    rows = [l.rstrip("\n").split("\t") for l in open(root / a.manifest)]
    recs = [json.loads(l) for l in open(root / a.hyps)]
    assert all(r["ref"].strip() == t[1].strip() for r, t in zip(recs, rows)), "ref mismatch"

    # manifest whose TEXT is the model's own beam output -> the dataset encodes hyp instead of ref
    hyp_man = root / "results" / "_self_manifest.tsv"
    hyp_man.write_text("".join(
        f"{root / a.clip_dir / Path(r[0]).name}\t{(rec['hyp_beam'].strip() or 'the')}\n"
        for r, rec in zip(rows, recs)))

    ck = torch.load(root / a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(str(hyp_man), vocab, cfg)
    dev = torch.device(a.device)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev).eval()
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)

    n = len(ds)
    rng = np.random.default_rng(a.seed)
    ce_t, ntok = per_clip_ce(model, ds, dev, [(i, i) for i in range(n)])
    ce_s = np.zeros((a.k, n))
    for r in range(a.k):
        donors = (np.arange(n) + rng.integers(1, n, size=n)) % n
        ce_s[r], _ = per_clip_ce(model, ds, dev, list(zip(range(n), donors)))
        print(f"  self donor round {r+1}/{a.k}", flush=True)
    self_ce = ce_t / np.maximum(ntok, 1)
    self_rel = (ce_s / np.maximum(ntok, 1)).mean(0) - self_ce
    print(f"\nself-reliance: mean {self_rel.mean():.3f} sd {self_rel.std():.3f} "
          f"negative on {100*(self_rel<0).mean():.1f}% of clips")

    prev = json.loads((root / a.reliance_json).read_text())
    reliance = np.array(prev["reliance"]); ce_true = np.array(prev["ce_true_pt"])

    import sacrebleu
    y = {"content_recall": np.array([
            np.mean([w in set(content_words(r["hyp_beam"])) for w in content_words(r["ref"])])
            if content_words(r["ref"]) else np.nan for r in recs]),
         "any_content": np.array([
            float(any(w in set(content_words(r["hyp_beam"])) for w in content_words(r["ref"])))
            if content_words(r["ref"]) else np.nan for r in recs]),
         "chrf": np.array([sacrebleu.sentence_chrf(detokenize(r["hyp_beam"]),
                                                   [detokenize(r["ref"])]).score for r in recs]),
         "exact": np.array([float(detokenize(r["hyp_beam"]).strip()
                                  == detokenize(r["ref"]).strip()) for r in recs])}

    preds = {"reliance (needs ref)": reliance,
             "reliance | CE(true) removed": None,                    # partial, handled below
             "SELF-reliance (inference-time)": self_rel,
             "SELF-CE of own output (neg)": -self_ce,
             "[ctrl] CE(true) (neg)": -ce_true}
    bands = {}
    real = {k: v for k, v in preds.items() if v is not None}
    for k, yy in y.items():
        mx = []
        for _ in range(a.perm):
            yp = yy.copy(); rng.shuffle(yp)
            mx.append(max(abs(corr(v, yp)) for v in real.values()))
        bands[k] = float(np.percentile(mx, 95))

    print(f"\n{'predictor':34}" + "".join(f"{k:>16}" for k in y))
    rep = {}
    for pn in preds:
        line = f"{pn:34}"
        for k, yy in y.items():
            r = partial(reliance, yy, ce_true) if preds[pn] is None else corr(preds[pn], yy)
            line += f"{r:>15.3f}{'*' if abs(r) > bands[k] else ' '}"
            rep.setdefault(pn, {})[k] = r
        print(line)
    print(f"{'NULL BAND (p95 max|r|)':34}" + "".join(f"{bands[k]:>16.3f}" for k in y))
    print("  * = exceeds the family-wise shuffled-outcome null band")
    print(f"\ncorr(reliance, CE(true)) = {corr(reliance, ce_true):+.3f}   "
          f"corr(self-reliance, reliance) = {corr(self_rel, reliance):+.3f}")

    if a.out:
        (root / a.out).write_text(json.dumps(
            {"ckpt": a.ckpt, "corr": rep, "null_band": bands,
             "self_reliance": self_rel.tolist(), "self_ce": self_ce.tolist()}, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
