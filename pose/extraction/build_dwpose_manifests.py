"""Repoint the official Communication manifests to the DWPose npz (same text/split),
keeping only clips that were successfully re-extracted."""
import os
DE = os.path.expanduser("~/Honours/results/dwpose_experiment")
CLIPS = os.path.join(DE, "dwpose_clips")
SRC = os.path.expanduser("~/Honours/comm_run/manifests")
OUT = os.path.join(DE, "manifests"); os.makedirs(OUT, exist_ok=True)
for split in ["train", "val", "test"]:
    n = tot = 0
    with open(f"{OUT}/{split}.tsv", "w") as w:
        for ln in open(f"{SRC}/{split}.tsv"):
            ln = ln.rstrip("\n")
            if not ln:
                continue
            tot += 1
            p, t = ln.split("\t")
            cid = os.path.basename(p).replace(".npz", "")
            dp = os.path.join(CLIPS, cid + ".npz")
            if os.path.exists(dp):
                w.write(f"{dp}\t{t}\n"); n += 1
    print(f"{split}: {n}/{tot} clips have DWPose pose", flush=True)
