"""Robustness follow-up to switch_lens.py section 2: is the S6 test association about the SWITCH, or about crowded scenes?

switch exposure (frac_chosen_nonzero) can only be > 0 when a second detection exists, and it rises with labelled
bystanders, so a crowded scene is a confound switch_lens.py did not control. Here the primary model gains
  (a) bystander count (0/1/2/3+ dummies) and Signer_ID group,
  (b) + frac_multi (share of frames with >= 2 detections: crowding itself),
  (c) the primary model restricted to clips where a switch was possible (max_ndet >= 2).
Same outcome definitions, same video-cluster bootstrap (2,000). Self-check: the generic regression reproduces
switch_lens.py's recorded PRIMARY test chrF slope exactly (same design columns, same point estimate).
"""
import csv, json
from pathlib import Path
import numpy as np
import switch_lens as sl

B = 2000
RES = json.load(open(Path(__file__).with_name("switch_lens_results.json")))


def fit(t, y, cols, rng):
    ok = ~np.isnan(y); t, y = t[ok].reset_index(drop=True), y[ok]
    X = np.column_stack([np.ones(len(t))] + [c(t) for c in cols.values()])
    b = sl.ols(y, X)[1]                                               # column 1 is always switch_frac
    bs = sl.cluster_boot(lambda i, c: sl.ols(y[i], X[i])[1], t.video.to_numpy(), rng, B)
    return float(b), sl.pct(bs), int(len(y))


BASE = {"switch_frac": lambda t: t.frac.to_numpy(), "ref_words": lambda t: t.ref_words.to_numpy(float),
        "seen": lambda t: t.seen.to_numpy(float), "log_frames": lambda t: np.log(t.frames.to_numpy())}
CROWD = {**BASE, "signer2": lambda t: (t.sgrp == "2").to_numpy(float), "signer_other": lambda t: (t.sgrp == "other").to_numpy(float),
         "byst1": lambda t: (t.byst == 1).to_numpy(float), "byst2": lambda t: (t.byst == 2).to_numpy(float),
         "byst3p": lambda t: (t.byst >= 3).to_numpy(float)}
MULTI = {**CROWD, "frac_multi": lambda t: t.frac_multi.to_numpy()}


def main():
    rng = np.random.default_rng(20260926)
    df = sl.load(); _, man = sl.gates(df)
    nd = {r["clip"]: r for r in csv.DictReader(open(sl.R / "ndet_per_clip.csv"))}
    seen_set = {t for _, t in man["train"]}
    out = {}
    for split, rec in (("test", 11.802), ("val", 12.470)):
        refs = [t for _, t in man[split]]
        hyps = sl.load_dump(sl.R / f"dumps_S6_{split}/hyps_LP=0.3.jsonl", "hyp_beam", refs)
        assert abs(sl.corpus_bleu4([sl.detokenize(h) for h in hyps], [sl.detokenize(r) for r in refs]) - rec) < 1e-3
        t = sl.clip_table(df, man[split], hyps, seen_set)
        t["frac_multi"] = [float(nd[c]["frac_multi"]) for c in t["clip"]]; t["max_ndet"] = [int(nd[c]["max_ndet"]) for c in t["clip"]]
        # SELF-CHECK: generic fit == switch_lens PRIMARY point estimate (bootstrap draws differ by rng, point must not)
        b, _, _ = fit(t, t.chrf.to_numpy(float), BASE, np.random.default_rng(0))
        want = RES["quality"][f"S6_{split}"]["chrf"]["primary"]["switch_frac"][0]
        assert abs(b - want) < 1e-9, (split, b, want)
        multi = t[t.max_ndet >= 2].reset_index(drop=True)
        print(f"--- S6 {split}  (clips with a possible switch, max_ndet>=2: {len(multi)}/{len(t)};"
              f" corr(switch_frac, frac_multi) {np.corrcoef(t.frac, t.frac_multi)[0, 1]:.2f})")
        out[split] = {}
        for yname in ("chrf", "recall"):
            y = t[yname].to_numpy(float) * (100 if yname == "recall" else 1)
            ym = multi[yname].to_numpy(float) * (100 if yname == "recall" else 1)
            for lab, tt, yy, cols in (("+ signer + bystanders", t, y, CROWD), ("+ signer + bystanders + frac_multi", t, y, MULTI),
                                      ("primary, max_ndet>=2 only", multi, ym, BASE)):
                b, ci, n = fit(tt, yy, cols, rng)
                out[split][f"{yname} {lab}"] = [b, ci, n]
                print(f"    {yname:6s} {lab:36s} n {n:4d} switch_frac {sl.fmt(b, ci)}")
    json.dump(out, open(Path(__file__).with_name("switch_lens_controls_results.json"), "w"), indent=1)
    print("SELF-CHECK OK (generic fit reproduces switch_lens PRIMARY slope on test and val)")


if __name__ == "__main__":
    main()
