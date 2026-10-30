"""Swing step 3 — build News manifests and combine with Communication (data scaling).

Reads the News translation xlsx (Video_Clip_Name -> Subtitle, Split), pairs each clip with its
DWPose npz, and writes News train/dev/test manifests. Then builds JOINT manifests:
  joint_train = Communication train  +  News train
  joint_val   = Communication val    (unchanged — the in-domain selector)
Evaluation stays on the Communication official TEST split, so the headline number remains
comparable to the 11.80 pose result. News test is written too (for an optional out-of-domain read).

Manifest line = "<npz_path>\t<subtitle>", matching slt/data.py.
"""
from __future__ import annotations
import argparse, os, glob

def load_news_labels(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True); ws = wb.active
    it = ws.iter_rows(values_only=True); hdr = list(next(it))
    def col(name):
        for i, h in enumerate(hdr):
            if h and str(h).strip().lower() == name: return i
        raise KeyError(name)
    ci, ti, si = col("video_clip_name"), col("subtitle"), col("split")
    rows = {}
    for r in it:
        if r[ci] is None or r[ti] is None: continue
        rows[str(r[ci])] = (str(r[ti]).strip(), str(r[si]).strip().lower())
    return rows

def norm_split(s):
    return {"dev": "val", "validation": "val"}.get(s, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news_xlsx", required=True)
    ap.add_argument("--news_npz_dir", required=True)
    ap.add_argument("--comm_manifest_dir", required=True, help="dir with Communication train/val/test.tsv")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    labels = load_news_labels(a.news_xlsx)

    # News: pair clip-id -> npz that actually exists
    by_split = {"train": [], "val": [], "test": []}
    missing = 0
    for cid, (text, split) in labels.items():
        npz = os.path.join(a.news_npz_dir, cid + ".npz")
        if not os.path.exists(npz): missing += 1; continue
        s = norm_split(split)
        if s in by_split and text:
            by_split[s].append(f"{npz}\t{text}")
    for s, rows in by_split.items():
        open(os.path.join(a.out_dir, f"news_{s}.tsv"), "w").write("\n".join(rows) + "\n")
        print(f"news_{s}: {len(rows)} clips")
    print(f"news clips with no npz yet: {missing}")

    # Combined: Comm train + News train ; Comm val unchanged ; eval stays Comm test.
    def read(p): return [l.rstrip("\n") for l in open(p) if l.strip()]
    comm_tr = read(os.path.join(a.comm_manifest_dir, "train.tsv"))
    comm_val = read(os.path.join(a.comm_manifest_dir, "val.tsv"))
    joint_tr = comm_tr + by_split["train"]
    open(os.path.join(a.out_dir, "joint_train.tsv"), "w").write("\n".join(joint_tr) + "\n")
    open(os.path.join(a.out_dir, "joint_val.tsv"), "w").write("\n".join(comm_val) + "\n")
    print(f"joint_train: {len(joint_tr)} ( Comm {len(comm_tr)} + News {len(by_split['train'])} )")
    print(f"joint_val (Comm): {len(comm_val)}  | eval on Comm test.tsv (unchanged)")

if __name__ == "__main__":
    main()
