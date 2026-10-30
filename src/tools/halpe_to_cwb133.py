"""Re-index AlphaPose HALPE-136 KPS onto the COCO-WholeBody-133 slots, so 133-layout tools run unchanged.

HALPE-136: 0-16 body (COCO order), 17-19 head/neck/hip, 20-25 feet, 26-93 face (68), 94-114 left hand, 115-135 right.
COCO-WB-133: 0-16 body, 17-22 feet, 23-90 face (68), 91-111 left hand, 112-132 right hand.
Every point group lands on the same keypoint SET; the 6 feet are permuted within their block, which leaves any
distance between descriptors unchanged. Head/neck/hip (17-19) have no WB slot and are dropped.

python tools/halpe_to_cwb133.py --clips a.npz b.npz --src_dir DIR --out_dir DIR   (self-check runs every call)
"""
import argparse, os

import numpy as np

IDX = np.r_[0:17, 20:136]
assert len(IDX) == 133 and list(IDX[91:112]) == list(range(94, 115)) and list(IDX[112:]) == list(range(115, 136))
assert list(IDX[23:91]) == list(range(26, 94))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="text file of <clip>.npz names")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for name in open(a.clips).read().split():
        z = np.load(os.path.join(a.src_dir, name))
        k = z["KPS"]
        assert k.shape[1] == 136, (name, k.shape)
        out = k[:, IDX]
        assert np.array_equal(out[:, 91:133], k[:, 94:136])
        np.savez(os.path.join(a.out_dir, name), KPS=out, BBOX=z["BBOX"])
        n += 1
    print(f"converted {n} clips -> {a.out_dir}")


if __name__ == "__main__":
    main()
