"""Data: Auslan-Daily HALPE-136 pose loading, normalisation, vocab, batching.

The real Auslan-Daily pose files are per-clip tensors with:
    KPS  : (T, 136, 3)  AlphaPose HALPE-136 whole-body keypoints,
                        raw pixel (x, y) + confidence.
    BBOX : (T, 4)       per-frame signer bounding box.

Known quirks handled here (from the dataset audit / the project log):
  * frame 0 (and occasionally others) can be all-zeros  -> masked out
  * coordinates are raw pixels                           -> bbox-normalised
  * only ~3k/25k clips have gloss                        -> gloss-free: we train
    directly on translation text, no gloss/CTC anywhere.

Everything is driven by a TSV manifest (`clip_path\ttarget_text`) so the code is
independent of the exact Google-Drive folder layout.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config, KP_SUBSETS

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_word_re = re.compile(r"[A-Za-z']+|[.,!?;:]")


def tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower().strip())


# ── vocabulary ──────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 1) -> "Vocab":
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        itos = list(SPECIALS)
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                itos.append(tok)
        return cls(itos)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)]
        ids = ids[: max_len - 2]
        return [BOS] + ids + [EOS]

    def decode(self, ids: Sequence[int]) -> str:
        toks = []
        for i in ids:
            if i in (BOS, PAD):
                continue
            if i == EOS:
                break
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return " ".join(toks)


# ── keypoint normalisation ──────────────────────────────────────────────────
def normalize_clip(kps: np.ndarray, bbox: np.ndarray, cfg: Config
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """(T,136,3) raw-pixel keypoints -> (T, F) features + (T,) valid-frame mask.

    Returns float32 features and a boolean mask (True = real frame, False =
    padded/invalid frame that the encoder should ignore).
    """
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 3 or kps.shape[1] != 136 or kps.shape[2] < 3:
        raise ValueError(f"expected KPS (T,136,3), got {kps.shape}")
    T = kps.shape[0]
    xy = kps[..., :2].copy()
    conf = kps[..., 2:3].copy()

    # invalid/zero frames: no keypoint has any signal
    frame_ok = np.ones(T, dtype=bool)
    if cfg.drop_zero_frames:
        nonzero = np.abs(kps).reshape(T, -1).sum(axis=1) > 1e-6
        frame_ok &= nonzero

    if cfg.coord_norm == "bbox":
        b = np.asarray(bbox, dtype=np.float32).reshape(T, -1)
        if cfg.bbox_format == "xyxy":
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            w, h = (x2 - x1), (y2 - y1)
        else:  # xywh
            x1, y1, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = np.where(np.abs(w) < 1e-3, 1.0, w)
        h = np.where(np.abs(h) < 1e-3, 1.0, h)
        frame_ok &= (np.abs(x2 - x1) > 1e-3) if cfg.bbox_format == "xyxy" else (np.abs(w) > 1e-3)
        xy[..., 0] = (xy[..., 0] - x1[:, None]) / w[:, None]
        xy[..., 1] = (xy[..., 1] - y1[:, None]) / h[:, None]
        if cfg.center_to_unit:
            xy = xy * 2.0 - 1.0
    elif cfg.coord_norm == "frame":
        m = xy.reshape(T, -1).mean(axis=1, keepdims=True)[:, :, None]
        s = xy.reshape(T, -1).std(axis=1, keepdims=True)[:, :, None] + 1e-6
        xy = (xy - m) / s
    # "none": leave raw

    if cfg.conf_threshold > 0:
        xy = xy * (conf >= cfg.conf_threshold)

    # zero-out coordinates on invalid frames so padding is clean
    xy[~frame_ok] = 0.0
    conf[~frame_ok] = 0.0

    # keypoint subset
    idx = KP_SUBSETS[cfg.keypoint_subset]
    xy = xy[:, idx, :]
    conf = conf[:, idx, :]

    feats = np.concatenate([xy, conf], axis=-1) if cfg.use_confidence else xy
    feats = feats.reshape(T, -1).astype(np.float32)

    # drop fully-invalid frames, then temporal cap by uniform subsample
    feats = feats[frame_ok]
    if feats.shape[0] == 0:                      # degenerate clip guard
        feats = np.zeros((1, feats.shape[1] if feats.ndim == 2 else cfg_feat_dim(cfg)), np.float32)
    if feats.shape[0] > cfg.max_src_frames:
        sel = np.linspace(0, feats.shape[0] - 1, cfg.max_src_frames).round().astype(int)
        feats = feats[sel]
    mask = np.ones(feats.shape[0], dtype=bool)
    return feats, mask


def cfg_feat_dim(cfg: Config) -> int:
    return cfg.feature_dim()


def load_pose(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one clip's KPS/BBOX from a .pt (or .npz) file, robust to layout."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["KPS"], z["BBOX"]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        kps = obj.get("KPS", obj.get("kps"))
        bbox = obj.get("BBOX", obj.get("bbox"))
    elif isinstance(obj, (tuple, list)):
        kps, bbox = obj[0], obj[1]
    else:
        kps, bbox = obj, None
    kps = kps.numpy() if torch.is_tensor(kps) else np.asarray(kps)
    if bbox is None:
        T = kps.shape[0]
        bbox = np.tile(np.array([0, 0, 1, 1], np.float32), (T, 1))
    bbox = bbox.numpy() if torch.is_tensor(bbox) else np.asarray(bbox)
    return kps, bbox


# ── dataset ─────────────────────────────────────────────────────────────────
def read_manifest(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            clip, text = line.split("\t", 1)
            rows.append((clip, text))
    return rows


class AuslanDailyDataset(Dataset):
    def __init__(self, manifest: str, vocab: Vocab, cfg: Config, root: str = ""):
        self.samples = read_manifest(manifest)
        self.vocab = vocab
        self.cfg = cfg
        self.root = root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        clip, text = self.samples[i]
        path = os.path.join(self.root, clip) if self.root else clip
        kps, bbox = load_pose(path)
        feats, _ = normalize_clip(kps, bbox, self.cfg)
        tgt = self.vocab.encode(text, self.cfg.max_tgt_tokens)
        return (
            torch.from_numpy(feats),
            torch.tensor(tgt, dtype=torch.long),
            text,
        )


def collate(batch, pad_id: int = PAD):
    feats, tgts, texts = zip(*batch)
    B = len(feats)
    Tmax = max(f.shape[0] for f in feats)
    Fdim = feats[0].shape[1]
    src = torch.zeros(B, Tmax, Fdim, dtype=torch.float32)
    src_pad = torch.ones(B, Tmax, dtype=torch.bool)          # True = PAD/ignore
    for i, f in enumerate(feats):
        t = f.shape[0]
        src[i, :t] = f
        src_pad[i, :t] = False
    Lmax = max(t.shape[0] for t in tgts)
    tgt = torch.full((B, Lmax), pad_id, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt[i, : t.shape[0]] = t
    return src, src_pad, tgt, list(texts)
