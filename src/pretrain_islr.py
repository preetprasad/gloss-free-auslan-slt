"""MM-WLAuslan ISLR pretraining (Stage 1 of the ISLR->CSLT transfer experiment).

Pretrains the *same* pose encoder used for Auslan-Daily translation on word-level
Auslan sign recognition (pose -> one of 3,215 glosses), then saves an encoder
checkpoint loadable by `train.py --init_encoder`. This is the supervised arm of
the 08-06 factorial (masked-MAE / MM-WLAuslan-supervised / combined -> Auslan-Daily).

Data
----
- Pose: gzip-compressed pickle mapping {sample_id: ndarray(T, K, C)} from AlphaPose
  (HALPE-136, same detector as Auslan-Daily). One camera view is sufficient.
- Labels: Labels+Split JSON mapping {sample_id: GLOSS}. Gloss set (3,215) is fixed
  from the train split so the classifier head is stable across runs.

Encoder transfer: the model is built by the shared `build_model` factory with the
identical `keypoint_subset` / feature dim as the downstream translator, so its
`input_proj` + positional encoding + `transformer.encoder` weights load directly.

Inspect the real pose format before a full run:
    python pretrain_islr.py --pose_pkl <cam1.pkl> --train_labels <Train.json> --inspect
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import Config, KP_SUBSETS, NUM_KEYPOINTS
from slt.engine import WarmupCosine, WarmupInverseSqrt, resolve_device
from slt.reference_model import build_model


# ── pose loading / normalisation ─────────────────────────────────────────────
def load_gzip_pickle(path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # old-numpy _frombuffer in the pkl
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def find_pose_array(node, num_kp=NUM_KEYPOINTS, depth=0):
    """Locate the (T, K, >=2) keypoint array by shape (survives container quirks)."""
    if isinstance(node, np.ndarray):
        if node.ndim == 3 and node.shape[2] >= 2 and node.shape[1] in (num_kp, 133, 134, 137):
            return node.astype(np.float32)
        return None
    if torch.is_tensor(node):
        return find_pose_array(node.detach().cpu().numpy(), num_kp, depth)
    if depth > 4:
        return None
    if isinstance(node, dict):
        for v in node.values():
            a = find_pose_array(v, num_kp, depth + 1)
            if a is not None:
                return a
    if isinstance(node, (list, tuple)):
        for v in node:
            a = find_pose_array(v, num_kp, depth + 1)
            if a is not None:
                return a
    return None


def normalise_isolated(kps: np.ndarray, cfg: Config) -> np.ndarray:
    """(T,K,C) AlphaPose -> (T, F) features matching the Auslan-Daily encoder.

    MM-WLAuslan pose has no per-frame signer box, so coordinates are standardised
    per frame (zero-mean/unit-std over valid joints) to a range comparable to the
    Auslan-Daily bbox+center_to_unit features. Keeps the confidence channel and the
    same keypoint subset so feature_dim matches the translator encoder.
    """
    kps = np.asarray(kps, dtype=np.float32)
    T, K, C = kps.shape
    xy = kps[..., :2].copy()
    conf = kps[..., 2:3].copy() if C >= 3 else np.ones((T, K, 1), np.float32)

    ok = np.abs(kps).reshape(T, -1).sum(1) > 1e-6           # non-zero frames
    # per-frame standardisation of xy over confident joints
    for t in range(T):
        m = conf[t, :, 0] >= max(cfg.conf_threshold, 1e-3)
        pts = xy[t, m] if m.any() else xy[t]
        mu = pts.mean(0, keepdims=True) if len(pts) else 0.0
        sd = pts.std() + 1e-6
        xy[t] = (xy[t] - mu) / sd
    xy[~ok] = 0.0
    conf[~ok] = 0.0

    if cfg.keypoint_subset != "all":
        if K != NUM_KEYPOINTS:
            raise ValueError(f"keypoint_subset={cfg.keypoint_subset!r} needs HALPE-{NUM_KEYPOINTS}; got K={K}")
        idx = KP_SUBSETS[cfg.keypoint_subset]
        xy, conf = xy[:, idx], conf[:, idx]

    feats = np.concatenate([xy, conf], -1) if cfg.use_confidence else xy
    feats = feats.reshape(feats.shape[0], -1).astype(np.float32)
    feats = feats[ok] if ok.any() else feats[:1]
    if feats.shape[0] > cfg.max_src_frames:
        sel = np.linspace(0, feats.shape[0] - 1, cfg.max_src_frames).round().astype(int)
        feats = feats[sel]
    return feats


class MMWLPoseGloss(Dataset):
    def __init__(self, pose_pkl, label_json, gloss2idx, cfg, limit=0):
        self.poses = load_gzip_pickle(pose_pkl)
        self.labels = json.load(open(label_json))
        self.gloss2idx = gloss2idx
        self.cfg = cfg
        self.ids = [s for s in self.poses
                    if s in self.labels and self.labels[s] in gloss2idx
                    and find_pose_array(self.poses[s]) is not None]
        if limit and limit < len(self.ids):
            self.ids = random.Random(0).sample(self.ids, limit)  # dev/smoke cap

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        arr = find_pose_array(self.poses[sid], self.cfg.num_keypoints)
        feats = normalise_isolated(arr, self.cfg)
        return torch.from_numpy(feats), self.gloss2idx[self.labels[sid]]


class MMWLPoseDirGloss(Dataset):
    """DWPose per-clip npz (keyed by bare <id>) + {id: gloss} label json.

    This is the representation-parity path: pose is re-extracted with DWPose (133-kpt
    COCO-WholeBody, same estimator as the Auslan-Daily translator) so the pretrained
    encoder transfers into `train_ctc.py --init_encoder`. Each npz has KPS (T,133,3).
    """

    def __init__(self, pose_dir, label_json, gloss2idx, cfg, limit=0, keep=None):
        import glob
        self.dir = pose_dir
        self.labels = json.load(open(label_json))
        self.gloss2idx = gloss2idx
        self.cfg = cfg
        # QC keep-list (ids passing DWPose-confidence quality gate) -> cleaner
        # pretraining signal. None = keep everything (backward compatible).
        files = glob.glob(os.path.join(pose_dir, "*.npz"))
        self.ids = []
        for f in files:
            sid = os.path.splitext(os.path.basename(f))[0]
            if keep is not None and sid not in keep:
                continue
            if sid in self.labels and self.labels[sid] in gloss2idx:
                self.ids.append((sid, f))
        if limit and limit < len(self.ids):
            self.ids = random.Random(0).sample(self.ids, limit)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid, f = self.ids[i]
        with np.load(f) as z:
            arr = np.asarray(z["KPS"], dtype=np.float32)
        feats = normalise_isolated(arr, self.cfg)
        return torch.from_numpy(feats), self.gloss2idx[self.labels[sid]]


def collate(batch):
    feats, ys = zip(*batch)
    B, Tmax, F = len(feats), max(f.shape[0] for f in feats), feats[0].shape[1]
    src = torch.zeros(B, Tmax, F)
    pad = torch.ones(B, Tmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        src[i, :f.shape[0]] = f
        pad[i, :f.shape[0]] = False
    return src, pad, torch.tensor(ys, dtype=torch.long)


# ── model: shared pose encoder + classification head ─────────────────────────
class ISLRModel(nn.Module):
    def __init__(self, feat_dim, num_glosses, cfg):
        super().__init__()
        self.slt = build_model(feat_dim, 8, cfg)          # reuse the real translator encoder
        self.head = nn.Linear(cfg.d_model, num_glosses)

    def forward(self, src, src_pad):
        mem = self.slt.encode(src, src_pad)               # (B,T,d)
        m = (~src_pad).unsqueeze(-1).to(mem.dtype)
        pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


def build_gloss_index(train_labels):
    glosses = sorted(set(json.load(open(train_labels)).values()))
    return {g: i for i, g in enumerate(glosses)}


def parse():
    cfg = Config()
    p = argparse.ArgumentParser(description="MM-WLAuslan ISLR pretraining (pose->gloss)")
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--pose_pkl", default="")           # gzip-pickle dict path (AlphaPose HALPE)
    p.add_argument("--pose_dir", default="")           # dir of DWPose per-clip npz (133-kpt parity path)
    p.add_argument("--val_pose_pkl", default="")
    p.add_argument("--val_pose_dir", default="")
    p.add_argument("--train_labels", required=True)
    p.add_argument("--val_labels", default="")
    p.add_argument("--limit_samples", type=int, default=0, help="cap dataset size (0=all)")
    p.add_argument("--keep_list", default="", help="QC keep-list txt (one clip id/line); ''=keep all")
    p.add_argument("--inspect", action="store_true", help="dump pose format and exit")
    a = p.parse_args()
    extra = {"pose_pkl", "pose_dir", "val_pose_pkl", "val_pose_dir",
             "train_labels", "val_labels", "limit_samples", "keep_list", "inspect"}
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in extra})
    return cfg, a


def main():
    cfg, a = parse()

    if a.inspect:
        obj = load_gzip_pickle(a.pose_pkl)
        print("top type:", type(obj).__name__, "| n:", len(obj) if hasattr(obj, "__len__") else "?")
        k0 = next(iter(obj)) if isinstance(obj, dict) else None
        print("sample id:", k0)
        arr = find_pose_array(obj[k0] if k0 is not None else obj)
        print("pose array shape:", None if arr is None else arr.shape, "(T, K, C)")
        return

    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    gloss2idx = build_gloss_index(a.train_labels)
    print(f"[islr-setup] glosses={len(gloss2idx)} device={device} feat_dim={cfg.feature_dim()}")

    keep = None
    if a.keep_list and os.path.exists(a.keep_list):
        keep = set(x.strip() for x in open(a.keep_list) if x.strip())
        print(f"[islr-setup] QC keep-list: {len(keep)} clips pass the DWPose-confidence gate", flush=True)
    if a.pose_dir:                                     # DWPose 133-kpt parity path
        train_ds = MMWLPoseDirGloss(a.pose_dir, a.train_labels, gloss2idx, cfg, limit=a.limit_samples, keep=keep)
    else:
        train_ds = MMWLPoseGloss(a.pose_pkl, a.train_labels, gloss2idx, cfg, limit=a.limit_samples)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)
    val_dl = None
    if a.val_pose_dir and a.val_labels and os.path.isdir(a.val_pose_dir):
        val_dl = DataLoader(MMWLPoseDirGloss(a.val_pose_dir, a.val_labels, gloss2idx, cfg, keep=keep),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)
    elif a.val_pose_pkl and a.val_labels and os.path.exists(a.val_pose_pkl):
        val_dl = DataLoader(MMWLPoseGloss(a.val_pose_pkl, a.val_labels, gloss2idx, cfg),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)

    model = ISLRModel(cfg.feature_dim(), len(gloss2idx), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98),
                           eps=1e-9, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = WarmupCosine(opt, cfg.lr, cfg.warmup_steps, steps) if cfg.lr_schedule == "cosine" \
        else WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    print(f"[islr-setup] train={len(train_ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    def evaluate():
        if val_dl is None:
            return None
        model.eval(); correct = total = 0
        with torch.no_grad():
            for src, pad, y in val_dl:
                src, pad, y = src.to(device), pad.to(device), y.to(device)
                correct += (model(src, pad).argmax(1) == y).sum().item(); total += y.numel()
        return correct / max(1, total)

    best = -1.0
    for ep in range(1, cfg.epochs + 1):
        model.train(); tl = ta = n = 0
        for src, pad, y in train_dl:
            src, pad, y = src.to(device), pad.to(device), y.to(device)
            logits = model(src, pad)
            loss = crit(logits, y)
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); sched.step()
            b = y.numel(); tl += loss.item() * b; ta += (logits.argmax(1) == y).sum().item(); n += b
        acc = evaluate()
        line = f"[islr ep {ep:03d}] loss={tl/max(1,n):.4f} train_acc={ta/max(1,n):.3f}"
        if acc is not None:
            line += f" val_acc={acc:.3f}"
        print(line, flush=True)
        score = acc if acc is not None else ta / max(1, n)
        if score > best:
            best = score
            torch.save({"model": model.slt.state_dict(), "cfg": cfg.to_dict(),
                        "gloss2idx": gloss2idx}, os.path.join(cfg.out_dir, "islr_encoder_best.pt"))
    print(f"[islr done] best={best:.3f} -> {cfg.out_dir}/islr_encoder_best.pt", flush=True)


if __name__ == "__main__":
    main()
