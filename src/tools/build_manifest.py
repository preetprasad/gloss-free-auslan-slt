"""Build train/val/test TSV manifests from Auslan-Daily translation .xlsx files.

The trainer consumes a simple `clip_path<TAB>text` manifest so it never has to
know the Drive layout. This tool bridges the real dataset to that format and is
designed to run **one-shot with auto-detected columns** (it prints what it chose),
while still accepting explicit overrides.

Examples
--------
Inspect only (print sheets/columns/sample, no output written):
    python tools/build_manifest.py --xlsx Auslan-Daily_Communication.xlsx --print_only

Build from both translation sheets, resolving clip ids to .pt files:
    python tools/build_manifest.py \
        --xlsx Auslan-Daily_Communication.xlsx Auslan-Daily_News.xlsx \
        --pose_root "/content/drive/MyDrive/Honours Datasets/Auslan-Daily" \
        --out_dir manifests --val_frac 0.1

Override auto-detection if the guesses are wrong:
    ... --id_col name --text_col translation --split_col split

Path resolution: for each row we search `<pose_root>/**/<id>.pt` (also .npz),
matching on the file stem. Rows whose pose file is missing are skipped and
counted, so a low "kept" count usually means the id<->filename mapping needs a
tweak (see --id_transform).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
from typing import Dict, List, Optional

import pandas as pd


# ── column auto-detection ────────────────────────────────────────────────────
def guess_text_col(df: pd.DataFrame) -> Optional[str]:
    best, best_score = None, 0.0
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        avg_words = s.str.split().apply(len).mean()
        alpha = s.str.contains(r"[A-Za-z]").mean()
        score = avg_words * float(alpha)
        if avg_words >= 3 and score > best_score:
            best, best_score = c, score
    return best


def guess_id_col(df: pd.DataFrame, text_col: Optional[str]) -> Optional[str]:
    # Skip pandas' unnamed index columns — "Unnamed: 0" contains the substring
    # "name", which previously hijacked the match and produced a row-index id.
    cols = [c for c in df.columns
            if c != text_col and not str(c).lower().startswith("unnamed")]
    # "clip" first: the per-clip id (e.g. Video_Clip_Name) is what pose files are
    # keyed by; the per-video name is a coarser fallback.
    for key in ("clip", "video", "name", "file", "sample", "id"):
        for c in cols:
            if key in str(c).lower():
                return c
    # fallback: a mostly-unique, mostly-single-token column
    for c in cols:
        s = df[c].dropna().astype(str)
        if not s.empty and s.nunique() / len(s) > 0.8 and s.str.split().apply(len).mean() < 2:
            return c
    return None


def guess_split_col(df: pd.DataFrame) -> Optional[str]:
    want = {"train", "val", "valid", "validation", "dev", "test"}
    for c in df.columns:
        s = set(df[c].dropna().astype(str).str.lower().unique()[:10])
        if s and s <= (want | {"testing"}):
            return c
        if "split" in str(c).lower() or "subset" in str(c).lower():
            return c
    return None


def index_pose_files(pose_root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for ext in ("pt", "npz"):
        for p in glob.glob(os.path.join(pose_root, "**", f"*.{ext}"), recursive=True):
            idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return idx


NORM_SPLIT = {"validation": "val", "valid": "val", "dev": "val", "testing": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", nargs="+", required=True, help="one or more .xlsx")
    ap.add_argument("--pose_root", default="", help="root to search for <id>.pt")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--id_transform", default="",
                    help="python str expr on `x` to map id->stem, e.g. \"x.split('.')[0]\"")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="val fraction when no split column is present")
    ap.add_argument("--out_dir", default="manifests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_only", action="store_true")
    a = ap.parse_args()

    pose_idx = index_pose_files(a.pose_root) if a.pose_root else {}
    if a.pose_root:
        print(f"[info] indexed {len(pose_idx)} pose files under {a.pose_root}")

    def to_stem(x: str) -> str:
        x = str(x).strip()
        if a.id_transform:
            try:
                x = str(eval(a.id_transform, {"__builtins__": {}}, {"x": x, "re": re}))
            except Exception:
                pass
        return x

    buckets: Dict[str, List[str]] = {}
    kept = skipped = 0
    for xlsx in a.xlsx:
        sheets = pd.read_excel(xlsx, sheet_name=None)
        df = sheets[a.sheet] if a.sheet else next(iter(sheets.values()))
        id_col = a.id_col or guess_id_col(df, a.text_col or guess_text_col(df))
        text_col = a.text_col or guess_text_col(df)
        split_col = a.split_col or guess_split_col(df)
        print(f"\n=== {os.path.basename(xlsx)} — sheets {list(sheets)} ===")
        print(f"shape {df.shape} | columns {list(df.columns)}")
        print(f"[chosen] id_col={id_col!r} text_col={text_col!r} split_col={split_col!r}")
        print(df[[c for c in (id_col, text_col, split_col) if c]].head(3).to_string(max_colwidth=50))
        if a.print_only:
            continue
        if not (id_col and text_col):
            print("[warn] could not determine id/text columns — pass --id_col/--text_col")
            continue
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text.lower() == "nan":
                skipped += 1; continue
            stem = to_stem(row[id_col])
            if a.pose_root:
                path = pose_idx.get(stem)
                if path is None:
                    skipped += 1; continue
            else:
                path = stem  # user resolves paths later
            if split_col and str(row[split_col]).strip():
                split = NORM_SPLIT.get(str(row[split_col]).strip().lower(),
                                       str(row[split_col]).strip().lower())
            else:
                split = None
            buckets.setdefault(split or "_pool", []).append(f"{path}\t{text}")
            kept += 1

    if a.print_only:
        return

    # deterministic train/val split for anything without an explicit split
    pool = buckets.pop("_pool", [])
    if pool:
        random.Random(a.seed).shuffle(pool)
        n_val = int(len(pool) * a.val_frac)
        buckets.setdefault("val", []).extend(pool[:n_val])
        buckets.setdefault("train", []).extend(pool[n_val:])

    # guarantee a val split (trainer needs one) — carve from train if absent
    if "val" not in buckets and buckets.get("train"):
        tr = list(buckets["train"])
        random.Random(a.seed).shuffle(tr)
        n_val = max(1, int(len(tr) * a.val_frac))
        buckets["val"] = tr[:n_val]
        buckets["train"] = tr[n_val:]

    os.makedirs(a.out_dir, exist_ok=True)
    for split, rows in buckets.items():
        out = os.path.join(a.out_dir, f"{split}.tsv")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[write] {out}: {len(rows)} rows")
    print(f"[done] kept={kept} skipped={skipped} "
          f"(low kept => check id<->filename mapping via --id_transform)")


if __name__ == "__main__":
    main()
