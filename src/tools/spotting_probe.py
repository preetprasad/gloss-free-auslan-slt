"""Feasibility probe for word-level sign-spotting on CONTINUOUS clips.

The two prior word-level attempts (ISLR transfer 9.98, PoseStitch 9.66) used the dictionary as
*pre-training* and were neutral. The untried angle is using it as an explicit lexical signal
(spotting / feature injection). Both hinge on one question: can the MM-WLAuslan ISLR encoder
recognise its known signs inside continuous, co-articulated Auslan-Daily signing? (Isolated->continuous
is a domain gap — the likely failure mode.) This probe answers it CHEAPLY before any big build:

  1. Build a per-gloss PROTOTYPE = mean encoder embedding over MM-WLAuslan valid clips of that gloss.
  2. For Auslan-Daily clips whose sentence contains known glosses, slide a window, embed each,
     nearest-prototype (cosine) -> spotted glosses.
  3. Report spotting recall/precision against the sentence's known-gloss set.
Decision: good recall -> build spotting-CTC / feature injection; poor -> the gap is fundamental.
"""
from __future__ import annotations
import argparse, json, os, re, glob, collections
import numpy as np, torch

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.reference_model import build_model

STOP = set("a an the is are am was were be been being to of in on at and or but i you he she it we they "
           "this that these those my your his her its our their do does did not have has had will would can "
           "could should may might must shall there here then than so as if for with from up out about into "
           "over after by".split())

def per_frame_norm(kps):  # matches pretrain_islr.normalise_isolated (per-frame standardisation)
    xy = kps[..., :2].astype(np.float32); conf = kps[..., 2:3].astype(np.float32)
    T = xy.shape[0]; flat = xy.reshape(T, -1)
    mu = flat.mean(1, keepdims=True); sd = flat.std(1, keepdims=True) + 1e-6
    xy = ((flat - mu) / sd).reshape(T, -1, 2)
    return np.concatenate([xy, conf], -1).reshape(T, -1).astype(np.float32)

@torch.no_grad()
def embed(model, feats, dev):
    x = torch.from_numpy(feats).unsqueeze(0).to(dev)
    pad = torch.zeros(1, x.shape[1], dtype=torch.bool, device=dev)
    mem = model.encode(x, pad)                 # (1,T,d)
    return torch.nn.functional.normalize(mem.mean(1), dim=-1)[0].cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True); ap.add_argument("--mmwl_labels", required=True)
    ap.add_argument("--mmwl_npz_dir", required=True); ap.add_argument("--ad_manifest", required=True)
    ap.add_argument("--protos_per_gloss", type=int, default=3); ap.add_argument("--limit_ad", type=int, default=40)
    ap.add_argument("--win", type=int, default=16); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame")
    model = build_model(cfg.feature_dim(), 8, cfg).to(dev).eval()
    ck = torch.load(a.encoder, map_location="cpu", weights_only=False); sd = ck.get("model", ck)
    ref = model.state_dict(); enc = {k: v for k, v in sd.items() if k in ref and v.shape == ref[k].shape}
    model.load_state_dict(enc, strict=False)
    print(f"[probe] loaded {len(enc)} encoder tensors", flush=True)

    va = json.load(open(a.mmwl_labels))
    g2c = collections.defaultdict(list)
    have = {os.path.basename(f)[:-4] for f in glob.glob(a.mmwl_npz_dir + "/*.npz")}
    for cid, g in va.items():
        if str(cid) in have: g2c[(g if isinstance(g, str) else str(g)).strip().lower()].append(str(cid))
    protos = {}
    for g, cids in g2c.items():
        embs = []
        for cid in cids[:a.protos_per_gloss]:
            d = np.load(os.path.join(a.mmwl_npz_dir, cid + ".npz"))
            embs.append(embed(model, per_frame_norm(d["KPS"]), dev))
        protos[g] = np.mean(embs, 0)
    P = np.stack(list(protos.values())); gnames = list(protos)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    print(f"[probe] {len(gnames)} gloss prototypes built", flush=True)

    rows = [l.rstrip("\n").split("\t", 1) for l in open(a.ad_manifest) if "\t" in l]
    gset = set(gnames); n = rec_num = rec_den = prec_num = prec_den = 0
    for path, text in rows:
        known = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and w in gset]
        if len(set(known)) < 2 or not os.path.exists(path): continue
        d = np.load(path); feats = per_frame_norm(d["KPS"]); T = feats.shape[0]
        spotted = []
        for s in range(0, max(1, T - a.win + 1), a.stride):
            e = embed(model, feats[s:s + a.win], dev)
            en = e / (np.linalg.norm(e) + 1e-8)
            spotted.append(gnames[int((P @ en).argmax())])
        spot = set(spotted); kn = set(known)
        rec_num += len(spot & kn); rec_den += len(kn)
        prec_num += len(spot & kn); prec_den += len(spot)
        n += 1
        if n <= 6:
            print(f"  text-known={sorted(kn)}  spotted∩known={sorted(spot & kn)}  n_spot={len(spot)}", flush=True)
        if n >= a.limit_ad: break
    print(f"\n[probe] over {n} clips: spotting RECALL={rec_num/max(1,rec_den):.2f} "
          f"(known signs found), PRECISION={prec_num/max(1,prec_den):.2f} (spots that are correct)", flush=True)
    print("[probe] verdict:", "VIABLE (recall>=0.4) -> build spotting/injection" if rec_num/max(1,rec_den)>=0.4
          else "WEAK -> isolated->continuous gap is fundamental; spotting unlikely to help", flush=True)

if __name__ == "__main__":
    main()
