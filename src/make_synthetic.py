"""Generate tiny synthetic HALPE-136 clips + manifests for the smoke test.

This mimics the real Auslan-Daily tensor layout — KPS (T,136,3) in raw pixel
coordinates with a confidence channel, BBOX (T,4), and an all-zero frame 0 to
exercise the zero-frame handling — WITHOUT any real data. Targets are drawn from
a tiny templated grammar so a small model can actually learn something and BLEU
moves off the floor, which is what makes the smoke test meaningful.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

W, H = 1920, 1080
SUBJECTS = ["the man", "the woman", "the boy", "the girl", "the teacher"]
VERBS = ["sees", "helps", "greets", "teaches", "calls"]
OBJECTS = ["the child", "the doctor", "a friend", "the family", "the class"]


def make_clip(seed: int):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(24, 80))
    # a moving bbox
    x1 = rng.uniform(200, 600); y1 = rng.uniform(100, 300)
    bw = rng.uniform(500, 800); bh = rng.uniform(600, 760)
    bbox = np.zeros((T, 4), np.float32)
    kps = np.zeros((T, 136, 3), np.float32)
    for t in range(T):
        cx = x1 + 5 * np.sin(t / 5.0)
        cy = y1 + 5 * np.cos(t / 5.0)
        # Match the released Auslan-Daily tensors: [x, y, width, height].
        bbox[t] = [cx, cy, bw, bh]
        if t == 0:
            continue  # all-zero frame 0 (real-data quirk)
        pts = rng.uniform(0, 1, size=(136, 2)) * [bw, bh] + [cx, cy]
        conf = rng.uniform(0.3, 1.0, size=(136, 1))
        kps[t, :, :2] = pts
        kps[t, :, 2:] = conf
    return kps, bbox


def sentence(seed: int) -> str:
    r = random.Random(seed)
    return f"{r.choice(SUBJECTS)} {r.choice(VERBS)} {r.choice(OBJECTS)} ."


def build(out_dir: str, n_train: int = 48, n_val: int = 16):
    clips = os.path.join(out_dir, "clips")
    man = os.path.join(out_dir, "manifests")
    os.makedirs(clips, exist_ok=True)
    os.makedirs(man, exist_ok=True)

    def write_split(name: str, n: int, base: int):
        rows = []
        for i in range(n):
            s = base + i
            kps, bbox = make_clip(s)
            path = os.path.join(clips, f"{name}_{i:04d}.pt")
            torch.save({"KPS": torch.from_numpy(kps), "BBOX": torch.from_numpy(bbox)}, path)
            rows.append(f"{path}\t{sentence(s)}")
        with open(os.path.join(man, f"{name}.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    write_split("train", n_train, 1000)
    write_split("val", n_val, 5000)
    return os.path.join(man, "train.tsv"), os.path.join(man, "val.tsv")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "_synthetic"
    tr, va = build(out)
    print("wrote", tr, va)
