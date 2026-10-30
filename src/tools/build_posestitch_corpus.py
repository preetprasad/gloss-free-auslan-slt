"""PoseStitch corpus generator — synthesize continuous (pose, text) pairs from MM-WLAuslan isolated signs.

For each Auslan-Daily training sentence with enough content-word coverage in the MM-WLAuslan
3,215-sign dictionary, stitch the covered signs' DWPose sequences in sentence order into one
synthetic continuous clip, paired with the REAL sentence text. Multiple pose realizations per
sentence (different exemplar clips) give augmentation. Output is DWPose-format npz (KPS+BBOX) +
a manifest, so the existing VLP/CTC pipeline consumes it unchanged.

This turns the word-level dataset into synthetic *continuous* pre-training data (PoseStitch-style).
Stitched pre-training teaches lexical grounding; fine-tuning on real Auslan-Daily restores grammar
and co-articulation. v1 uses simple concatenation (co-articulation gap noted).
"""
from __future__ import annotations
import argparse, json, os, re, random, collections
import numpy as np

STOP = set("a an the is are am was were be been being to of in on at and or but i you he she it we "
           "they this that these those my your his her its our their do does did not have has had will "
           "would can could should may might must shall there here then than so as if for with from up "
           "out about into over after by".split())

def words(t): return re.findall(r"[a-z']+", t.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid_labels", required=True)     # MM-WLAuslan Valid.json {id: GLOSS}
    ap.add_argument("--valid_npz_dir", required=True)    # dir of <id>.npz (KPS,BBOX)
    ap.add_argument("--ad_manifest", required=True)      # Auslan-Daily train.tsv (npz\ttext)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_cov", type=float, default=0.6)
    ap.add_argument("--min_signs", type=int, default=2)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    npz_out = os.path.join(a.out_dir, "clips"); os.makedirs(npz_out, exist_ok=True)

    va = json.load(open(a.valid_labels))
    have = {os.path.basename(f)[:-4] for f in os.listdir(a.valid_npz_dir) if f.endswith(".npz")}
    gloss2clips = collections.defaultdict(list)
    for cid, g in va.items():
        cid = str(cid)
        if cid in have:
            gloss2clips[(g if isinstance(g, str) else str(g)).strip().lower()].append(cid)
    gset = set(gloss2clips)
    print(f"[stitch] pose dictionary: {len(gset)} glosses with >=1 valid clip", flush=True)

    cache = {}
    def load(cid):
        if cid not in cache:
            d = np.load(os.path.join(a.valid_npz_dir, cid + ".npz"))
            cache[cid] = (d["KPS"].astype(np.float32), d["BBOX"].astype(np.float32))
        return cache[cid]

    rows = []; n_written = 0
    for ln in open(a.ad_manifest):
        if "\t" not in ln: continue
        text = ln.rstrip("\n").split("\t", 1)[1]
        cw = [w for w in words(text) if w not in STOP]
        if not cw: continue
        seq = [w for w in cw if w in gset]                      # covered content words, in order
        if len(seq) < a.min_signs or len(seq) / len(cw) < a.min_cov:
            continue
        for r in range(a.realizations):
            kps_parts, bbox_parts = [], []
            for w in seq:
                cid = random.choice(gloss2clips[w])
                k, b = load(cid)
                kps_parts.append(k); bbox_parts.append(b)
            KPS = np.concatenate(kps_parts, 0); BBOX = np.concatenate(bbox_parts, 0)
            name = f"stitch_{n_written:06d}.npz"
            np.savez_compressed(os.path.join(npz_out, name), KPS=KPS, BBOX=BBOX)
            rows.append(f"{os.path.join(npz_out, name)}\t{text}")
            n_written += 1
    open(os.path.join(a.out_dir, "posestitch_train.tsv"), "w").write("\n".join(rows) + "\n")
    print(f"[stitch] wrote {n_written} synthetic clips from "
          f"{n_written // max(1,a.realizations)} covered sentences -> {a.out_dir}/posestitch_train.tsv", flush=True)
    if rows:
        ex = np.load(rows[0].split("\t")[0]); print(f"[stitch] example KPS{ex['KPS'].shape} BBOX{ex['BBOX'].shape}")

if __name__ == "__main__":
    main()
