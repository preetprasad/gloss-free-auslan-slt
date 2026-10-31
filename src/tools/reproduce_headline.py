#!/usr/bin/env python3
"""Tier-0 reproduction: re-score every committed decode dump and assert it matches the record.

No GPU, no dataset, no checkpoints -- only the saved hypotheses in results/dumps_*/ and the project's
own metric code. This is the check that the reported numbers are what the repository actually contains.

    python tools/reproduce_headline.py            # re-score and assert
    python tools/reproduce_headline.py --verbose  # also print per-system detail

Exits non-zero on ANY mismatch, so it is usable as a CI gate.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.metrics import corpus_bleu4, corpus_wer, detokenize          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)

# (label, dump path relative to repo root, recorded BLEU-4, recorded WER or None, source of the record)
CASES = [
    ("S6 ensemble (reported operating point, beam 10 / LP 0.3)",
     "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl", 11.802, 0.9069,
     "the project log headline; re-decoded 2026-09-22 jobs 6806/7921 to +-0.002"),
    ("S6 ensemble, validation split",
     "results/final_ensemble_validation_decodes/hyps_LP=0.3.jsonl", 12.470, 0.9164,
     "the project log 2026-09-22k; the LP was selected on this split"),
    ("F' = 1.1:1 duration members + 2 RGB-fusion (NOT CLAIMED: CI vs S6 includes zero)",
     "results/ensemble_one_to_one_duration_members_decodes/hyps_LP=0.6.jsonl", 12.836, 0.9213,
     "results/PREREG_ENSEMBLE_R111_2026-09-15.md"),
    ("F = duration members + 2 RGB-fusion (RETRACTED 2026-09-10: not resolvable)",
     "results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl", 12.114, 0.9526,
     "results/ENSEMBLE_F_RESULT_2026-09-10.md; retraction in the project log 2026-09-10t"),
    ("G1 = 6 duration + 2 fusion (best on VALIDATION, worse on test -- the val/test divergence)",
     "results/ensemble_six_duration_two_fusion_decodes/hyps_LP=0.7.jsonl", 11.562, 0.9706,
     "the project log 2026-09-10t"),
    ("E = 3 duration members, pose-only",
     "results/ensemble_three_duration_members_decodes/hyps_LP=0.8.jsonl", 11.537, 0.9988,
     "the project log 2026-09-10h"),
]

# Reference subsets for the seen/unseen decomposition of the reported system.
TRAIN_MANIFEST = "comm_run/manifests/train.tsv"
# SHA-1 of every training target (tools/train_target_hashes.py): used when the subtitles are not on disk.
TRAIN_HASHES = "results/train_target_hashes.txt"


def load(path):
    rows = [json.loads(l) for l in open(os.path.join(REPO, path))]
    return ([detokenize(r["hyp_beam"]) for r in rows],
            [detokenize(r["ref"]) for r in rows])


def seen_unseen(hyps, refs):
    """Split by whether the reference appears verbatim among the training targets."""
    mp, hp = os.path.join(REPO, TRAIN_MANIFEST), os.path.join(REPO, TRAIN_HASHES)
    if os.path.exists(mp):
        train = {detokenize(l.rstrip("\n").split("\t", 1)[1]) for l in open(mp) if "\t" in l}
        is_seen = lambda r: r in train
    elif os.path.exists(hp):
        digests = set(open(hp).read().split())
        is_seen = lambda r: hashlib.sha1(r.encode("utf-8")).hexdigest() in digests
    else:
        return None
    s = [(h, r) for h, r in zip(hyps, refs) if is_seen(r)]
    u = [(h, r) for h, r in zip(hyps, refs) if not is_seen(r)]
    out = {}
    for name, pairs in (("seen", s), ("unseen", u)):
        out[name] = (len(pairs),
                     corpus_bleu4([h for h, _ in pairs], [r for _, r in pairs]) if pairs else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol_bleu", type=float, default=0.01,
                    help="absolute BLEU-4 tolerance (default 0.01; the dumps are exact, so this is slack only "
                         "for the rounding in the recorded values)")
    ap.add_argument("--tol_wer", type=float, default=0.001)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print(f"{'system':<62} {'BLEU-4':>8} {'rec':>8}  {'WER':>7} {'rec':>7}  n")
    print("-" * 104)
    failures, missing = [], []
    for label, path, rec_b, rec_w, src in CASES:
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            missing.append((label, path)); continue
        hyps, refs = load(path)
        b, w = corpus_bleu4(hyps, refs), corpus_wer(hyps, refs)
        ok_b = abs(b - rec_b) <= a.tol_bleu
        ok_w = rec_w is None or abs(w - rec_w) <= a.tol_wer
        flag = "" if (ok_b and ok_w) else "   <-- MISMATCH"
        print(f"{label[:62]:<62} {b:8.3f} {rec_b:8.3f}  {w:7.4f} "
              f"{(rec_w if rec_w is not None else float('nan')):7.4f}  {len(hyps)}{flag}")
        if a.verbose:
            print(f"{'':>4}source: {src}")
        if not (ok_b and ok_w):
            failures.append((label, b, rec_b, w, rec_w))

    # The honest generalisation number for the reported system.
    hyps, refs = load(CASES[0][1])
    su = seen_unseen(hyps, refs)
    if su:
        print("-" * 104)
        print(f"reported system decomposed by whether the reference occurs verbatim in training:")
        print(f"  seen   n={su['seen'][0]:4d}  BLEU-4 {su['seen'][1]:6.2f}   (recorded 35.00)")
        print(f"  unseen n={su['unseen'][0]:4d}  BLEU-4 {su['unseen'][1]:6.2f}   (recorded  6.60)  <- the honest number")
        for name, rec in (("seen", 35.00), ("unseen", 6.60)):
            if abs(su[name][1] - rec) > 0.05:
                failures.append((f"{name} subset", su[name][1], rec, float('nan'), None))

    if missing:
        print("\nNOT FOUND (these dumps are not in this checkout):")
        for label, path in missing:
            print(f"  {path}   [{label}]")
    if failures:
        print(f"\nFAILED: {len(failures)} mismatch(es) against the recorded values.")
        return 1
    print(f"\nOK: {len(CASES) - len(missing)} system(s) reproduce their recorded BLEU-4 and WER exactly.")
    if missing:
        print(f"    ({len(missing)} dump(s) absent from this checkout and therefore not checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
