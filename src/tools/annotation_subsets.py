"""Report the STRICT, leak-free Auslan-Daily weak-label subsets usable for training.

Why this exists
---------------
`report-source.md` sec 2.2 tabulated sign-spotting and fingerspelling subset sizes with no
script behind them, and an independent recount on 2026-09-09 disagreed by ~23%.  A number
that gates an experiment has to be reproducible, so this is the committed computation.

The load-bearing correctness point is the LEAK.  Both annotation files carry their own
`Split` column, and that column does NOT agree with the Communication translation split:
rows those files label `train` include Communication dev and test clips.  Training on
`Split == 'train'` would therefore leak evaluation clips into a weak-label objective.
The only safe subset is the intersection with the Communication TRAIN split.

Clip ids also differ in form: the annotation files use `<clip>_signer.mp4` while the
translation table uses `<clip>`, and Fingerspelling_Detection.xlsx additionally mixes in
News clips that are not in Communication at all.
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

SIGNER_SUFFIX = re.compile(r"_signer\.mp4$")


def _clip_id(value) -> str:
    return SIGNER_SUFFIX.sub("", str(value).strip())


def _split_sets(comm_xlsx: str) -> dict[str, set[str]]:
    comm = pd.read_excel(comm_xlsx)
    out = {}
    for name, key in (("train", "train"), ("dev", "dev"), ("test", "test")):
        rows = comm[comm["Split"].astype(str).str.lower() == key]
        out[name] = {_clip_id(v) for v in rows["Video_Clip_Name"]}
    return out


def _report(tag: str, path: str, clip_col: str, label_col: str, split_col: str,
            splits: dict[str, set[str]]) -> dict:
    d = pd.read_excel(path)
    d["_cid"] = d[clip_col].map(_clip_id)
    in_train = d[d["_cid"].isin(splits["train"])]

    own_train = d[d[split_col].astype(str).str.lower() == "train"]
    leak_dev = int(own_train["_cid"].isin(splits["dev"]).sum())
    leak_test = int(own_train["_cid"].isin(splits["test"]).sum())
    known = splits["train"] | splits["dev"] | splits["test"]
    outside = int((~d["_cid"].isin(known)).sum())

    print(f"=== {tag} ===")
    print(f"  file rows                     : {len(d)}")
    print(f"  SAFE SUBSET (Comm TRAIN only) : {len(in_train)} spans, "
          f"{in_train['_cid'].nunique()} clips, {in_train[label_col].nunique()} distinct {label_col}")
    print(f"  LEAK if you trust its own Split=='train': {leak_dev} Comm-DEV + {leak_test} Comm-TEST rows")
    print(f"  rows outside Communication entirely (News/other): {outside}")
    return {"tag": tag, "rows": len(d), "safe_spans": len(in_train),
            "safe_clips": int(in_train["_cid"].nunique()),
            "safe_classes": int(in_train[label_col].nunique()),
            "leak_dev": leak_dev, "leak_test": leak_test, "outside_communication": outside}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm_xlsx", default="data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
    ap.add_argument("--split_table_root", default="data/raw/Auslan-Daily/Dataset Split Table")
    a = ap.parse_args()

    splits = _split_sets(a.comm_xlsx)
    print(f"Communication splits: train={len(splits['train'])} dev={len(splits['dev'])} "
          f"test={len(splits['test'])}\n")

    _report("SIGN SPOTTING",
            os.path.join(a.split_table_root, "Sign Spotting", "Sign_Spotting.xlsx"),
            "Video_Name_Clip", "Query_Gloss", "Split", splits)
    print()
    _report("FINGERSPELLING DETECTION",
            os.path.join(a.split_table_root, "Fingerspelling Detection", "Fingerspelling_Detection.xlsx"),
            "Video_Clip_Name", "Finger_Spelling", "split", splits)
    print("\nNote: neither file supplies ordered continuous gloss sequences, so neither can "
          "support Auslan-order gloss CTC. They support LOCAL span-level auxiliary objectives only.")


if __name__ == "__main__":
    main()
