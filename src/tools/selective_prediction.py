"""Risk-coverage: if the system declines the clips it is least sure about, does quality actually rise?

The descriptor analysis found no confidence signal in the VIDEO. The reliance analysis then found two in
the MODEL, both computable at inference time: the CE of the model's own output, and self-reliance
(how much that output's likelihood depends on this clip's pose rather than another's).

A correlation is not an abstention system. This measures the thing a deployment would care about:
corpus BLEU-4 on the retained set as a function of coverage, against two controls -- random abstention
(what any ranking beats by chance) and an ORACLE that abstains by true per-clip quality (the ceiling).
Output length is included as a ranking too, because a "confidence" signal that is secretly length would
be exploiting the same brevity channel that makes WER useless on this corpus.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slt.metrics import corpus_bleu4, detokenize                     # noqa: E402


def curve(rank: np.ndarray, hyps, refs, covs) -> list[float]:
    """rank: higher = keep. corpus BLEU on the top-coverage fraction."""
    order = np.argsort(-rank, kind="stable")
    return [corpus_bleu4([hyps[i] for i in order[:int(round(c * len(hyps)))]],
                         [refs[i] for i in order[:int(round(c * len(hyps)))]], tokenize="13a")
            for c in covs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--self_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    recs = [json.loads(l) for l in open(root / a.hyps)]
    hyps = [detokenize(r["hyp_beam"]) for r in recs]
    refs = [detokenize(r["ref"]) for r in recs]
    d = json.loads((root / a.self_json).read_text())
    self_ce = np.array(d["self_ce"]); self_rel = np.array(d["self_reliance"])
    hyp_len = np.array([len(h.split()) for h in hyps], float)

    import sacrebleu
    per_clip = np.array([sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)])

    rng = np.random.default_rng(a.seed)
    covs = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
    rankings = {"self-CE of own output (keep low CE)": -self_ce,
                "self-reliance (keep high)": self_rel,
                "self-CE + self-reliance (z-sum)": ((-self_ce - (-self_ce).mean()) / self_ce.std()
                                                    + (self_rel - self_rel.mean()) / self_rel.std()),
                "[ctrl] output length (keep long)": hyp_len,
                "[ORACLE] true per-clip chrF": per_clip}
    rand = np.array([curve(rng.random(len(hyps)), hyps, refs, covs) for _ in range(50)])

    print(f"corpus BLEU-4 on the RETAINED set, by coverage   (n={len(hyps)}, full-set BLEU-4 = "
          f"{corpus_bleu4(hyps, refs, tokenize='13a'):.2f})\n")
    print(f"{'ranking':38}" + "".join(f"{int(c*100):>9}%" for c in covs))
    print(f"{'[ctrl] random abstention (mean of 50)':38}"
          + "".join(f"{v:>10.2f}" for v in rand.mean(0)))
    for name, r in rankings.items():
        vals = curve(r, hyps, refs, covs)
        line = f"{name:38}" + "".join(f"{v:>10.2f}" for v in vals)
        print(line)
    print(f"\n{'random-abstention sd across 50 draws':38}" + "".join(f"{v:>10.2f}" for v in rand.std(0)))
    print(f"\ncorr(self-CE, output length) = "
          f"{np.corrcoef(self_ce, hyp_len)[0,1]:+.3f}   "
          f"corr(self-reliance, output length) = {np.corrcoef(self_rel, hyp_len)[0,1]:+.3f}")
    print("A ranking that merely keeps long outputs would track the output-length row.")


if __name__ == "__main__":
    main()
