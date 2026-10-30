#!/usr/bin/env python3
"""Intervals and a matched-coverage contrast for S6 selective prediction (fix owed since 2026-09-23d).

Reuses tools/ensemble_selective.py's load/bleu and its deployable rule: confidence = -self-CE of the selected
hypothesis; threshold tau_c = quantile of the VALIDATION confidence at 1 - c; applied unchanged to test. Adds:
  * a 95% paired sentence bootstrap of the retained-set BLEU (tau fixed; each draw resamples test sentences and keeps
    those with conf >= tau);
  * the matched-coverage contrast: in each draw, retained-set BLEU minus the BLEU of a RANDOM subset of the same size
    drawn from the same resample (the random-abstention baseline at identical coverage);
  * the composition of S6's own retained set (verbatim-training hypothesis %, seen-reference %, reference and
    hypothesis length), replacing the single-model composition borrowed from 09-10b.
Gate: re-derives the recorded val-fixed test figures (all 12.90 / 14.61 / 19.07 at 89.4 / 75.6 / 51.9 %).

  python tools/selective_ci.py --val_dump results/final_ensemble_validation_decodes --test_dump results/final_ensemble_test_decodes --lp 0.3 \
      --train_manifest results/_kaya_train_manifest.tsv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ensemble_selective import load, bleu          # noqa: E402

COVS = (0.9, 0.75, 0.5)
RECORD = {0.9: (12.90, 0.894), 0.75: (14.61, 0.756), 0.5: (19.07, 0.519)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dump", required=True); ap.add_argument("--test_dump", required=True)
    ap.add_argument("--lp", type=float, default=0.3); ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train = {l.rstrip("\n").split("\t", 1)[1].strip() for l in open(a.train_manifest) if "\t" in l}
    vr, _, vc = load(a.val_dump, a.lp)
    r, h, c = load(a.test_dump, a.lp)
    vseen = np.array([x.strip() in train for x in vr]); seen = np.array([x.strip() in train for x in r])
    verb = np.array([x.strip() in train for x in h])
    N = len(r); rng = np.random.default_rng(a.seed)
    full = bleu(h, r)
    print(f"S6 test full coverage BLEU {full:.2f} | hyp verbatim-train {verb.mean():.1%} | ref seen {seen.mean():.1%} | "
          f"ref len {np.mean([len(x.split()) for x in r]):.2f} hyp len {np.mean([len(x.split()) for x in h]):.2f}")
    for subset in ("all", "unseen"):
        keep = np.ones(N, bool) if subset == "all" else ~seen
        vcs = vc if subset == "all" else vc[~vseen]
        idx0 = np.flatnonzero(keep)
        print(f"\n== {subset} (n={len(idx0)}) ==")
        for cov in COVS:
            tau = np.quantile(vcs, 1 - cov)
            sel = idx0[c[idx0] >= tau]
            b = bleu([h[i] for i in sel], [r[i] for i in sel]); got_cov = len(sel) / len(idx0)
            if subset == "all":
                rb, rcov = RECORD[cov]
                assert abs(b - rb) < 0.006 and abs(got_cov - rcov) < 0.0006, f"GATE: {cov} -> {b:.3f} @ {got_cov:.3f}"
            bs, ds = [], []
            for _ in range(a.n):
                ii = rng.choice(idx0, size=len(idx0), replace=True)
                s_ = ii[c[ii] >= tau]
                rnd = rng.choice(ii, size=len(s_), replace=False)
                bsel = bleu([h[i] for i in s_], [r[i] for i in s_])
                bs.append(bsel); ds.append(bsel - bleu([h[i] for i in rnd], [r[i] for i in rnd]))
            lo, hi = np.percentile(bs, [2.5, 97.5]); dlo, dhi = np.percentile(ds, [2.5, 97.5])
            comp = (f"hyp verbatim-train {verb[sel].mean():.1%} | ref seen {seen[sel].mean():.1%} | ref len "
                    f"{np.mean([len(r[i].split()) for i in sel]):.2f} | hyp len {np.mean([len(h[i].split()) for i in sel]):.2f}")
            print(f"cov {cov:.2f} (val-fixed tau) -> test {b:6.2f} @ {got_cov:5.1%}  95% [{lo:5.2f}, {hi:5.2f}] | "
                  f"selected - random at matched coverage {np.mean(ds):+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
                  + (f"\n      retained-set composition: {comp}" if subset == "all" else ""))
    print("\n(test sampling only: thresholds fixed on validation, so validation sampling is not in these intervals)")


if __name__ == "__main__":
    main()
