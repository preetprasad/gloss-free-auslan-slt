#!/usr/bin/env python3
"""Paired bootstrap CIs + seen/unseen BLEU-4 for saved decode dumps (official Communication test).

Why a committed tool: every earlier confidence interval in this project was computed ad hoc. The method
is the one in results/TEST_SPLIT_RESOLUTION_2026-09-10.md -- resample the test sentences with replacement,
recompute BOTH systems' corpus BLEU-4 on the SAME resample, repeat 2,000 times -- scored through the
project's own slt.metrics (detokenise, sacreBLEU 13a), never a hand-rolled tokeniser (09-09l: a
hand-rolled tokeniser once moved WER by ~0.16).

Before any CI is printed, --expect NAME=VALUE asserts that the dump re-scores to a number already on
record, so a wrong file, a wrong split or a changed metric fails loudly instead of producing a plausible CI.

Usage:
  python tools/paired_bootstrap.py \
      --system S6=results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl --system F=results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl \
      --pairs F:S6 --train_manifest results/_kaya_train_manifest.tsv --expect S6=11.802 --expect F=12.114
Dumps: jsonl rows {"idx", "ref", "hyp_beam"} as written by ensemble_mbr_wer.py --dump_dir.
"seen" = the reference occurs verbatim (lowercased, whitespace-collapsed) among training targets.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402


def norm(text):
    return " ".join(text.lower().split())


def load_dump(spec):
    name, path = spec.split("=", 1)
    rows = sorted((json.loads(line) for line in open(path, encoding="utf-8")), key=lambda r: r["idx"])
    return name, [r["hyp_beam"] for r in rows], [r["ref"] for r in rows]


def bleu(hyps, refs, keep=None):
    if keep is not None:
        hyps = [h for h, k in zip(hyps, keep) if k]
        refs = [r for r, k in zip(refs, keep) if k]
    return corpus_bleu4(hyps, refs, "13a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, help="NAME=path/to/hyps.jsonl (repeatable)")
    ap.add_argument("--pairs", default="", help="comma-separated A:B pairs to bootstrap")
    ap.add_argument("--train_manifest", default="", help="train TSV for the seen/unseen split")
    ap.add_argument("--expect", action="append", default=[], help="NAME=BLEU on record; abort on mismatch")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    systems, raw_refs = {}, None
    for spec in a.system:
        name, hyps, refs = load_dump(spec)
        if raw_refs is None:
            raw_refs = refs
        if refs != raw_refs:
            sys.exit(f"FATAL: {name} references differ in content or order from the first system")
        systems[name] = [detokenize(h) for h in hyps]
    R = [detokenize(r) for r in raw_refs]
    n = len(R)

    seen = None
    if a.train_manifest:
        train = {norm(line.split("\t", 1)[1]) for line in open(a.train_manifest, encoding="utf-8") if "\t" in line}
        seen = np.array([norm(r) in train for r in raw_refs])
        print(f"test n={n}  seen-in-train={int(seen.sum())}  unseen={int((~seen).sum())}")

    full = {}
    for name, H in systems.items():
        full[name] = bleu(H, R)
        extra = ""
        if seen is not None:
            extra = f"  seen {bleu(H, R, seen):6.2f}  unseen {bleu(H, R, ~seen):5.2f}"
        print(f"{name:12s} BLEU-4 {full[name]:7.3f}{extra}")

    for spec in a.expect:
        name, value = spec.split("=", 1)
        if abs(full[name] - float(value)) > 0.0015:
            sys.exit(f"FATAL: {name} re-scores to {full[name]:.3f}, record says {value} -- not bootstrapping")
        print(f"check: {name} reproduces the recorded {value}")

    rng = np.random.default_rng(a.seed)
    draws = rng.integers(0, n, size=(a.n, n))
    for pair in filter(None, a.pairs.split(",")):
        A, B = pair.split(":")
        HA, HB = systems[A], systems[B]
        deltas = np.empty(a.n)
        for k, ii in enumerate(draws):
            Rk = [R[i] for i in ii]
            deltas[k] = corpus_bleu4([HA[i] for i in ii], Rk, "13a") - corpus_bleu4([HB[i] for i in ii], Rk, "13a")
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        verdict = "significant" if (lo > 0 or hi < 0) else "not resolvable"
        print(f"{A} vs {B}: observed {full[A] - full[B]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P({A}>{B}) {100 * np.mean(deltas > 0):.1f}%  -> {verdict}")


if __name__ == "__main__":
    main()
