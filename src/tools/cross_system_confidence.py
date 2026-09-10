"""Does model-derived confidence work as a CROSS-SYSTEM selector, or only within a system?

The ensemble headroom analysis (2026-09-09t/u) closed the selection line: plurality vote, MBR-chrF,
MBR-sentBLEU and a learned selector all captured <=7% of the oracle gap, while logit averaging captured
30%. Every one of those selectors read SURFACE features of finished hypotheses.

Self-CE is different in kind -- it is the model's own likelihood, and it predicts per-clip quality at
r = 0.42-0.47 WITHIN a system. Whether that transfers ACROSS systems is a separate question, because
cross-system comparison needs confidence to be CALIBRATED between two independently trained models,
which nothing guarantees.

Each candidate is scored under BOTH models, giving four quantities per clip. Selectors tested:
  own-CE       pick the hypothesis its own producer scores best  (needs calibration across models)
  cross-CE     pick the hypothesis the OTHER model scores best   (a judge with no stake in it)
  sum-CE       pick by total CE under both                        (a 2-model committee)
Baselines: each single system, random choice, and the per-sentence oracle over the two.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config                                            # noqa: E402
from slt.data import AuslanDailyDataset, Vocab                       # noqa: E402
from slt.reference_model import build_model                          # noqa: E402
from slt.metrics import corpus_bleu4, detokenize                     # noqa: E402
from tools.per_clip_reliance import per_clip_ce                      # noqa: E402


def score_under(ckpt_path, texts, clip_names, clip_dir, root, device):
    """Per-token CE of each given text under this checkpoint, paired with its own clip."""
    man = root / "results" / "_xsys_manifest.tsv"
    man.write_text("".join(f"{root / clip_dir / n}\t{(t.strip() or 'the')}\n"
                           for n, t in zip(clip_names, texts)))
    ck = torch.load(root / ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(str(man), vocab, cfg)
    dev = torch.device(device)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev).eval()
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)
    ce, ntok = per_clip_ce(model, ds, dev, [(i, i) for i in range(len(ds))])
    return ce / np.maximum(ntok, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="comm_run/manifests/test.tsv")
    ap.add_argument("--clip_dir", default="results/test_clips_dwpose")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    SYS = [("ctc", "results/ckpts/comm_dwpose_ctc.pt", "results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl"),
           ("vlp", "results/ckpts/comm_dwpose_vlp_ctc.pt", "results/semantic_adequacy_grading/hyps_vlp.jsonl")]
    rows = [l.rstrip("\n").split("\t") for l in open(root / a.manifest)]
    names = [Path(r[0]).name for r in rows]
    refs = [detokenize(r[1]) for r in rows]
    hyps = {}
    for tag, _, hp in SYS:
        recs = [json.loads(l) for l in open(root / hp)]
        assert all(r["ref"].strip() == t[1].strip() for r, t in zip(recs, rows)), f"{hp} ref mismatch"
        hyps[tag] = [detokenize(r["hyp_beam"]) for r in recs]

    # CE[scorer][candidate] : per-token CE of candidate's text under scorer's model
    CE = {}
    for stag, ck, _ in SYS:
        for ctag, _, _ in SYS:
            CE[(stag, ctag)] = score_under(ck, hyps[ctag], names, a.clip_dir, root, a.device)
            print(f"  scored {ctag} under {stag}: mean CE {CE[(stag, ctag)].mean():.4f}", flush=True)

    import sacrebleu
    q = {t: np.array([sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps[t], refs)])
         for t, _, _ in SYS}
    n = len(refs)
    tags = [t for t, _, _ in SYS]

    def bleu(pick):
        return corpus_bleu4([hyps[tags[p]][i] for i, p in enumerate(pick)], refs, tokenize="13a")

    rng = np.random.default_rng(a.seed)
    own = np.array([CE[(t, t)] for t in tags])                       # each scored by its producer
    cross = np.array([CE[(tags[1 - k], tags[k])] for k in range(2)])  # scored by the OTHER model
    tot = own + cross

    sel = {
        f"single: {tags[0]}": np.zeros(n, int),
        f"single: {tags[1]}": np.ones(n, int),
        "random choice (mean of 50)": None,
        "OWN-CE (each system's own confidence)": own.argmin(0),
        "CROSS-CE (scored by the other model)": cross.argmin(0),
        "SUM-CE (2-model committee)": tot.argmin(0),
        "[ORACLE] per-sentence best by chrF": np.array([q[tags[0]], q[tags[1]]]).argmax(0),
    }
    print(f"\nCROSS-SYSTEM SELECTION over {n} clips, corpus BLEU-4\n")
    for name, pick in sel.items():
        if pick is None:
            v = np.mean([bleu(rng.integers(0, 2, n)) for _ in range(50)])
            print(f"{name:42} {v:7.2f}")
            rand = v
            continue
        agree = 100 * (pick == sel["[ORACLE] per-sentence best by chrF"]).mean()
        print(f"{name:42} {bleu(pick):7.2f}   picks oracle-best {agree:5.1f}% of the time")
    print(f"\nchance agreement with oracle = 50.0%; systems produce identical output on "
          f"{100*np.mean([h1 == h2 for h1, h2 in zip(hyps[tags[0]], hyps[tags[1]])]):.1f}% of clips")
    print(f"corr(own-CE of ctc, own-CE of vlp) = {np.corrcoef(CE[('ctc','ctc')], CE[('vlp','vlp')])[0,1]:+.3f}")
    print(f"mean own-CE: ctc {CE[('ctc','ctc')].mean():.4f}  vlp {CE[('vlp','vlp')].mean():.4f}  "
          f"-> a shared threshold is only meaningful if these are comparable")


if __name__ == "__main__":
    main()
