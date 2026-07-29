"""Stream per-clip pose .npz straight out of an Auslan-Daily pose zip.

Disk-safe counterpart to ``extract_clips.py``: instead of unzipping the whole
archive to disk (News_Annotation.zip expands to ~14 GB) and then converting, this
reads each per-video ``.pt`` member from the zip **in memory** (io.BytesIO),
extracts the (T,136,3) KPS / (T,4) BBOX arrays by shape, and writes one
compressed ``.npz`` per clip — so the 14 GB of ``.pt`` never touches disk.

Usage
-----
    python tools/extract_from_zip.py \
        --zip "<...>/News_Annotation.zip" --out_dir /path/to/clips

Skips clips whose .npz already exists (safe to re-run / resume).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import torch

# reuse the shape-based array finder from the sibling extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_clips import find_pose_arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to the pose .zip")
    ap.add_argument("--out_dir", required=True, help="where per-clip .npz go (fast local disk)")
    ap.add_argument("--num_kp", type=int, default=136)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    zf = zipfile.ZipFile(a.zip)
    members = [n for n in zf.namelist() if n.lower().endswith(".pt")]
    print(f"[info] {os.path.basename(a.zip)}: {len(members)} .pt members -> {a.out_dir}")

    n_clip = n_skip = n_badvid = 0
    for vi, name in enumerate(members, 1):
        try:
            raw = zf.read(name)
            obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        except Exception as e:  # corrupt / unreadable member
            print(f"  [warn] load failed {name}: {e}")
            n_badvid += 1
            continue
        del raw
        if not isinstance(obj, dict):
            n_badvid += 1
            continue
        for clip_name, clip in obj.items():
            out = os.path.join(a.out_dir, f"{clip_name}.npz")
            if os.path.exists(out) and not a.overwrite:
                n_clip += 1
                continue
            kps, bbox = find_pose_arrays(clip, a.num_kp)
            if kps is None:
                n_skip += 1
                continue
            if bbox is None:  # tolerate missing bbox -> identity normalisation
                bbox = np.tile(np.array([0, 0, 1, 1], np.float32), (kps.shape[0], 1))
            np.savez_compressed(out, KPS=kps, BBOX=bbox)
            n_clip += 1
        del obj
        if vi % 5 == 0 or vi == len(members):
            print(f"  [{vi}/{len(members)}] clips={n_clip} skipped={n_skip} bad_videos={n_badvid}")

    print(f"[done] clips={n_clip} skipped_no_kps={n_skip} bad_videos={n_badvid} -> {a.out_dir}")


if __name__ == "__main__":
    main()
