"""Extract per-clip pose arrays from Auslan-Daily per-VIDEO .pt files.

Confirmed real structure (verified against the data 2026-07-18):

    video_1.pt = {
        'video_1_0': {'Name': str, 'Annotation': { ... KPS/BBOX arrays ... }},
        'video_1_1': {...},           # ~94 clips per video file
        ...
    }
    157 video files  ->  ~14.7k clips total

The clip keys (`video_1_0`) match the translation sheet's `Video_Clip_Name`
exactly. Note `Video_Name` is capitalised (`Video_1`) while the file is
lowercase (`video_1.pt`).

Loading a 70 MB video file per training sample would be crippling, so this does a
one-time pass writing **one .npz per clip** (`<clip_name>.npz` with `KPS`/`BBOX`).
`slt/data.py` already reads .npz, and `build_manifest.py` then matches
`Video_Clip_Name` -> `<clip_name>.npz` directly.

Usage
-----
    # quick subset first (fast path to a first BLEU-4)
    python tools/extract_clips.py --pose_root "<BASE>/Auslan-Daily" \
        --out_dir /content/clips --limit_videos 20

    # full extraction
    python tools/extract_clips.py --pose_root "<BASE>/Auslan-Daily" --out_dir /content/clips

Write to fast local disk (/content on Colab, $MYSCRATCH on Kaya), not Drive —
tens of thousands of small files on Drive is painfully slow.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch


def _as_array(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return None


def find_pose_arrays(node, num_kp: int = 136):
    """Recursively locate the (T,K,>=2) keypoint array and (T,4) bbox in a nest.

    Key names aren't assumed — we match on shape, so this survives whatever the
    inner `Annotation` dict calls things.
    """
    found = {"kps": None, "bbox": None}

    def walk(n, depth=0):
        if depth > 6 or (found["kps"] is not None and found["bbox"] is not None):
            return
        if isinstance(n, dict):
            for v in n.values():
                walk(v, depth + 1)
            return
        if isinstance(n, (list, tuple)):
            for v in n:
                walk(v, depth + 1)
            return
        a = _as_array(n)
        if a is None:
            return
        if a.ndim == 3 and a.shape[1] == num_kp and a.shape[2] >= 2 and found["kps"] is None:
            found["kps"] = a.astype(np.float32)
        elif a.ndim == 2 and a.shape[1] == 4 and found["bbox"] is None:
            found["bbox"] = a.astype(np.float32)

    walk(node)
    return found["kps"], found["bbox"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_root", required=True, help="dir containing the video .pt files")
    ap.add_argument("--out_dir", required=True, help="where per-clip .npz go (use fast local disk)")
    ap.add_argument("--limit_videos", type=int, default=0, help="0 = all; else first N videos")
    ap.add_argument("--num_kp", type=int, default=136)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(a.pose_root, "**", "*.pt"), recursive=True))
    if a.limit_videos:
        vids = vids[: a.limit_videos]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[info] {len(vids)} video .pt files -> {a.out_dir}")

    n_clip = n_skip = 0
    for vi, vp in enumerate(vids, 1):
        try:
            obj = torch.load(vp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [warn] load failed {os.path.basename(vp)}: {e}")
            continue
        if not isinstance(obj, dict):
            print(f"  [warn] {os.path.basename(vp)} is {type(obj).__name__}, expected dict of clips")
            continue
        for clip_name, clip in obj.items():
            out = os.path.join(a.out_dir, f"{clip_name}.npz")
            if os.path.exists(out) and not a.overwrite:
                n_clip += 1
                continue
            # Each clip's Annotation holds every person in the scene ({1: ..., 2: ..., 'Signer': ...}); a shape-based
            # walk takes the first person, a non-signer in 88.4% of Communication clips (found 25 Oct 2026). Take 'Signer'.
            ann = clip.get("Annotation") if isinstance(clip, dict) else None
            if not (isinstance(ann, dict) and "Signer" in ann):
                raise SystemExit(f"{clip_name}: no Annotation['Signer'] entry (keys {list(ann) if isinstance(ann, dict) else type(clip)})")
            kps, bbox = find_pose_arrays(ann["Signer"], a.num_kp)
            if kps is None:
                n_skip += 1
                continue
            if bbox is None:  # tolerate missing bbox -> identity normalisation
                bbox = np.tile(np.array([0, 0, 1, 1], np.float32), (kps.shape[0], 1))
            np.savez_compressed(out, KPS=kps, BBOX=bbox)
            n_clip += 1
        if vi % 10 == 0 or vi == len(vids):
            print(f"  [{vi}/{len(vids)}] clips written={n_clip} skipped={n_skip}")

    print(f"[done] clips={n_clip} skipped_no_kps={n_skip} -> {a.out_dir}")
    if n_clip:
        ex = sorted(os.listdir(a.out_dir))[:3]
        z = np.load(os.path.join(a.out_dir, ex[0]))
        print(f"[check] {ex} | KPS{z['KPS'].shape} BBOX{z['BBOX'].shape}")


if __name__ == "__main__":
    main()
