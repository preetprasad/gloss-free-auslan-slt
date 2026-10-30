"""Are the adopted-recipe runs from five different jobs the SAME configuration? Compares saved checkpoint cfgs
(ignoring paths) and the training manifest's content md5, per seed."""
import hashlib, os, sys
import torch
G = "/group/pmc099/pprasad"
RUNS = {"pa_aug": "pose_aug/runs/pa_aug_s{s}", "w1A": "wave1_r2/runs/w1A_s{s}", "w3C": "wave3_track/runs/w3C_s{s}",
        "r3_111": "wave3_ratioaug/runs/r3_111_s{s}", "a3ctl": "wave3_augdose/runs/a3ctl_s{s}"}
IGN = {"out_dir", "train_manifest", "val_manifest", "test_manifest"}
for s in (42, 101, 202):
    ref = None
    for name, pat in RUNS.items():
        p = f"{G}/{pat.format(s=s)}/best.pt"
        if not os.path.exists(p):
            print(s, name, "MISSING"); continue
        ck = torch.load(p, map_location="cpu", weights_only=False); c = ck["cfg"]
        tm = c["train_manifest"]
        body = [l.split("\t", 1)[1] for l in open(tm)] if os.path.exists(tm) else []
        h = hashlib.md5("".join(body).encode()).hexdigest()[:8]
        if ref is None:
            ref = c; diff = {}
        else:
            diff = {k: (ref.get(k), c.get(k)) for k in set(ref) | set(c) if k not in IGN and ref.get(k) != c.get(k)}
        print(f"s{s} {name:7s} train rows {len(body)} text-md5 {h} vocab {len(ck['vocab'])} diff {diff}")
