"""Does the PoseStitch gain come from VISUAL grounding or from DECODER-SIDE language modelling?

This is the zero-training version of the pose-scrambled control, and it is decisive enough to
run before any fidelity work on the stitcher.

WHY THE QUESTION IS OPEN. Joint training on real+synthetic beats its own same-seed control in
3/3 paired seeds (mean +0.56 beam-5). But four observations all point away from visual transfer:
  * the gain is BEAM-ONLY (+0.71 beam-5 vs +0.04 greedy) -- if the encoder had learned
    sign->meaning, greedy would move too;
  * output diversity rises 39.6% -> 46.8% unique, a regularisation signature;
  * the gain is FLAT across synthetic:real ratio, unique-sentence count AND realisations-per-
    sentence simultaneously -- the signature of a benefit that is not about the visual data;
  * a model-free retrieval probe puts continuous->citation matching at 3.5% top-1 (chance 2.2%)
    against 43-64% within-domain, so the pose-level correspondence needed for lexical grounding
    is essentially absent.
Independently: one MM-WLAuslan citation clip averages 79.4 frames while a WHOLE real
Auslan-Daily sentence averages 91.2, and the two corpora are recorded at 30 fps vs 25 fps.

THE TEST. Teacher-forced decoder cross-entropy on the REAL validation targets under two
conditions, for both the joint model and its control:
  (a) TRUE pose      -- the clip that actually goes with the sentence
  (b) SHUFFLED pose  -- another clip's pose, same batch, sentence unchanged
`reliance` = CE(shuffled) - CE(true) measures how much a model actually USES its visual input.

READING IT:
  * if the joint model keeps most of its CE advantage under SHUFFLED pose, the advantage is
    language modelling / regularisation -- and stitcher fidelity work is dead on arrival;
  * if the joint model's advantage disappears under SHUFFLED pose, or its `reliance` is
    materially higher than the control's, synthetic data really did buy visual grounding.
"""
from __future__ import annotations
import argparse, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, PAD
from slt.reference_model import build_model


@torch.no_grad()
def measure(ckpt_path, manifest, device, batch_size=32, seed=0):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], tokenization=cfg.text_tokenization,
                  sentencepiece_model=cfg.sentencepiece_model)
    ds = AuslanDailyDataset(manifest, vocab, cfg)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(device).eval()
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)

    g = torch.Generator().manual_seed(seed)
    tot = {"true": 0.0, "shuf": 0.0}
    ntok = 0
    order = torch.randperm(len(ds), generator=g).tolist()
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        if len(idx) < 2:
            continue
        src, src_pad, tgt, _ = collate([ds[j] for j in idx])
        src, src_pad, tgt = src.to(device), src_pad.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_pad = tgt_in == PAD
        for tag, roll in (("true", 0), ("shuf", 1)):
            # roll src AND its mask together so the mismatched pose stays internally consistent
            s = torch.roll(src, roll, 0)
            sp = torch.roll(src_pad, roll, 0)
            mem = model.encode(s, sp)
            logits = model.decode_step(tgt_in, mem, sp, tgt_pad)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 tgt_out.reshape(-1), ignore_index=PAD, reduction="sum")
            tot[tag] += float(ce)
        ntok += int((tgt_out != PAD).sum())
    return tot["true"] / ntok, tot["shuf"] / ntok, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--manifest", required=True, help="REAL validation manifest")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    print(f"{'model':30} {'CE(true)':>9} {'CE(shuffled)':>13} {'reliance':>9}")
    res = {}
    for c in a.ckpts:
        name = os.path.basename(os.path.dirname(c))
        try:
            t, s, n = measure(c, a.manifest, a.device)
            res[name] = (t, s)
            print(f"{name:30} {t:9.4f} {s:13.4f} {s - t:9.4f}")
        except Exception as e:
            print(f"{name:30} FAILED {type(e).__name__}: {e}")
    print("\nreliance = CE(shuffled) - CE(true): how much the model actually uses its visual input.")
    ks = list(res)
    if len(ks) >= 2:
        base = ks[0]
        # NOTE: an earlier version reported "% of the CE advantage surviving pose shuffling". That
        # was built on an assumption the data falsified -- the joint models have WORSE CE(true) than
        # the control, so the "advantage" is negative and the ratio of two negatives produced
        # meaningless figures (378%, 555%, 2705%). RELIANCE is well defined regardless of sign, so
        # the comparison is made on reliance, and the CE dissociation is reported as its own fact.
        rb = res[base][1] - res[base][0]
        for k in ks[1:]:
            rk = res[k][1] - res[k][0]
            dce = res[k][0] - res[base][0]
            print(f"\n{k} vs {base}:")
            print(f"  reliance {rk:.4f} vs {rb:.4f}  ->  {100 * (rk - rb) / rb:+.1f}% "
                  f"{'MORE' if rk > rb else 'LESS'} use of the visual input")
            print(f"  CE(true) {dce:+.4f} ({'worse' if dce > 0 else 'better'} token-level likelihood)")
        print("\nHigher reliance in the joint arms => synthetic data increased visual grounding.")
        print("Worse CE(true) alongside better beam BLEU => a regularisation/calibration effect:")
        print("the model is less peaked on the training distribution, which beam search exploits.")


if __name__ == "__main__":
    main()
