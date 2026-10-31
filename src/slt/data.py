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

import hashlib
import os
import random
import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from config import Config, KP_SUBSETS, NUM_KEYPOINTS

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_word_re = re.compile(r"[A-Za-z']+|[.,!?;:]")


def _load_sentencepiece(model_path: str):
    if not model_path:
        raise ValueError("sentencepiece tokenization requires --sentencepiece_model")
    import sentencepiece as sentencepiece_module
    return sentencepiece_module.SentencePieceProcessor(model_file=model_path)


def tokenize(text: str, mode: str = "regex", sentencepiece=None) -> List[str]:
    """Tokenize text while keeping legacy checkpoints reproducible."""
    text = text.lower().strip()
    if mode == "whitespace":
        return text.split()
    if mode == "sentencepiece":
        if sentencepiece is None:
            raise ValueError("sentencepiece processor is required")
        return list(sentencepiece.encode(text, out_type=str))
    if mode == "regex":
        return _word_re.findall(text)
    raise ValueError(f"unknown text_tokenization={mode!r}")


# ── vocabulary ──────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self, itos: List[str], tokenization: str = "regex",
                 sentencepiece_model: str = ""):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.tokenization = tokenization
        self.sentencepiece_model = sentencepiece_model
        self.sentencepiece = (_load_sentencepiece(sentencepiece_model)
                              if tokenization == "sentencepiece" else None)

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 1,
              tokenization: str = "regex",
              sentencepiece_model: str = "") -> "Vocab":
        sentencepiece = (_load_sentencepiece(sentencepiece_model)
                         if tokenization == "sentencepiece" else None)
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t, tokenization, sentencepiece))
        itos = list(SPECIALS)
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                itos.append(tok)
        return cls(itos, tokenization, sentencepiece_model)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenize(
            text, self.tokenization, self.sentencepiece)]
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
        if self.tokenization == "sentencepiece":
            return self.sentencepiece.decode(toks)
        return " ".join(toks)


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Torchtext-style shuffled pools, length sorting, then sentence batches.

    The public GASLT/SignJoey iterator shuffles training examples into pools of
    100 batches, sorts each pool by source length, creates fixed-size sentence
    batches, and shuffles those batches. This preserves stochastic training while
    preventing a single long clip from padding an otherwise short batch.
    """

    def __init__(self, dataset, batch_size: int, max_src_frames: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.lengths = []
        for clip, _ in dataset.samples:
            path = os.path.join(dataset.root, clip) if dataset.root else clip
            if dataset.cfg.input_representation == "precomputed":
                with np.load(path) as payload:
                    length = int(payload[dataset.cfg.precomputed_feature_key].shape[0])
            elif path.endswith(".npz"):
                with np.load(path) as payload:
                    length = int(payload["KPS"].shape[0])
            else:
                keypoints, _ = load_pose(path)
                length = int(keypoints.shape[0])
            self.lengths.append(min(length, int(max_src_frames)))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        generator.shuffle(indices)
        pool_size = self.batch_size * 100
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = indices[start:start + pool_size]
            pool.sort(key=self.lengths.__getitem__)
            batches.extend(pool[offset:offset + self.batch_size]
                           for offset in range(0, len(pool), self.batch_size))
        generator.shuffle(batches)
        yield from batches


# ── keypoint normalisation ──────────────────────────────────────────────────
def normalize_clip(kps: np.ndarray, bbox: np.ndarray, cfg: Config
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """(T,136,3) raw-pixel keypoints -> (T, F) features + (T,) valid-frame mask.

    Returns float32 features and a boolean mask (True = real frame, False =
    padded/invalid frame that the encoder should ignore).
    """
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 3 or kps.shape[2] < 3:
        raise ValueError(f"expected KPS (T,K,3+), got {kps.shape}")
    if kps.shape[1] != cfg.num_keypoints:
        raise ValueError(
            f"KPS has {kps.shape[1]} keypoints but cfg.num_keypoints={cfg.num_keypoints}. "
            "Set --num_keypoints to the extractor layout size."
        )
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
            raw_w, raw_h = (x2 - x1), (y2 - y1)
        elif cfg.bbox_format == "xywh":
            x1, y1, raw_w, raw_h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        else:
            raise ValueError(f"bbox_format must be 'xywh' or 'xyxy', got {cfg.bbox_format!r}")
        # A valid detection box needs positive width *and* height. Checkpoints
        # produced before the bbox audit can explicitly request their historical
        # abs() validity rule so evaluation remains bit-for-bit input compatible.
        if cfg.legacy_abs_bbox_validity:
            box_ok = (np.abs(raw_w) > 1e-3) & (np.abs(raw_h) > 1e-3)
        else:
            box_ok = (raw_w > 1e-3) & (raw_h > 1e-3)
        frame_ok &= box_ok
        safe_w = np.where(box_ok, raw_w, 1.0)
        safe_h = np.where(box_ok, raw_h, 1.0)
        xy[..., 0] = (xy[..., 0] - x1[:, None]) / safe_w[:, None]
        xy[..., 1] = (xy[..., 1] - y1[:, None]) / safe_h[:, None]
        if cfg.center_to_unit:
            xy = xy * 2.0 - 1.0
    elif cfg.coord_norm == "frame":
        m = xy.reshape(T, -1).mean(axis=1, keepdims=True)[:, :, None]
        s = xy.reshape(T, -1).std(axis=1, keepdims=True)[:, :, None] + 1e-6
        xy = (xy - m) / s
    # "none": leave raw

    if cfg.conf_threshold > 0:
        xy = xy * (conf >= cfg.conf_threshold)

    if cfg.hierarchical_pose_norm:
        if kps.shape[1] != NUM_KEYPOINTS:
            raise ValueError(
                f"hierarchical_pose_norm requires HALPE-{NUM_KEYPOINTS}; "
                f"got {kps.shape[1]} keypoints"
            )
        # Body is made signer-scale invariant around the shoulder midpoint.
        # Global wrist motion remains in body joints 9/10.  Each detailed hand
        # is then expressed locally around its wrist and scaled by palm length,
        # so subtle finger shape is not dwarfed by the full signer box.
        left_shoulder, right_shoulder = 5, 6
        shoulder_center = (xy[:, left_shoulder] + xy[:, right_shoulder]) * 0.5
        shoulder_scale = np.linalg.norm(
            xy[:, left_shoulder] - xy[:, right_shoulder], axis=-1)
        shoulder_ok = ((conf[:, left_shoulder, 0] >= max(cfg.conf_threshold, 0.1))
                       & (conf[:, right_shoulder, 0] >= max(cfg.conf_threshold, 0.1))
                       & (shoulder_scale > 0.25))
        # A conservative bbox-coordinate fallback avoids amplifying bad shoulder
        # detections while keeping every frame finite.
        safe_shoulder_scale = np.where(shoulder_ok, shoulder_scale, 1.0)
        body_normalised = np.clip(
            (xy[:, :26] - shoulder_center[:, None, :])
            / safe_shoulder_scale[:, None, None], -5.0, 5.0)
        xy[:, :26] = np.where(shoulder_ok[:, None, None],
                              body_normalised, xy[:, :26])

        for start in (94, 115):
            wrist = xy[:, start].copy()
            middle_mcp = xy[:, start + 9].copy()
            palm_scale = np.linalg.norm(middle_mcp - wrist, axis=-1)
            hand_ok = ((conf[:, start, 0] >= max(cfg.conf_threshold, 0.1))
                       & (conf[:, start + 9, 0] >= max(cfg.conf_threshold, 0.1))
                       & (palm_scale > 0.03))
            safe_palm_scale = np.where(hand_ok, palm_scale, 1.0)
            hand_normalised = np.clip(
                (xy[:, start:start + 21] - wrist[:, None, :])
                / safe_palm_scale[:, None, None], -5.0, 5.0)
            xy[:, start:start + 21] = np.where(
                hand_ok[:, None, None], hand_normalised,
                xy[:, start:start + 21]
            )

    if cfg.hand_local_norm:
        # COCO-WholeBody-133: 91 = left wrist + 20 pts, 112 = right wrist + 20 pts; within a hand the
        # order is wrist, thumb, index, MIDDLE(+9), ring, pinky, so wrist->middle-MCP is palm length.
        #
        # WHY: under coord_norm="frame" the whole pose is standardised together, so in conversational
        # signing -- where the signer moves and the hands travel the signing space -- finger detail is a
        # tiny and highly variable share of the frame's variance. Expressing each hand around its own
        # wrist at palm scale restores it. This is the hand half of `hierarchical_pose_norm`, which
        # raises on every current run because it hard-requires HALPE-136.
        #
        # Measured before being built (tools/continuous_id_probe.py, 45 glosses, clip-disjoint):
        # continuous-domain top-1 18.1% -> 24.9%, top-5 34.7% -> 46.9%, stable over 5 seeds
        # (24.6-25.8%). Isolated-domain top-1 is UNCHANGED/slightly lower (40.0% -> 39.2%), i.e. this
        # is specifically a continuous-signing fix, which is the domain we translate.
        if kps.shape[1] < 133:
            raise ValueError(f"hand_local_norm requires COCO-WholeBody-133; got {kps.shape[1]}")
        for start in (91, 112):
            wrist = xy[:, start].copy()
            palm_scale = np.linalg.norm(xy[:, start + 9] - wrist, axis=-1)
            hand_ok = ((conf[:, start, 0] >= max(cfg.conf_threshold, 0.1))
                       & (conf[:, start + 9, 0] >= max(cfg.conf_threshold, 0.1))
                       & (palm_scale > 1e-3))
            safe = np.where(hand_ok, palm_scale, 1.0)
            normalised = np.clip((xy[:, start:start + 21] - wrist[:, None, :])
                                 / safe[:, None, None], -5.0, 5.0)
            xy[:, start:start + 21] = np.where(hand_ok[:, None, None], normalised,
                                               xy[:, start:start + 21])

    # zero-out coordinates on invalid frames so padding is clean
    xy[~frame_ok] = 0.0
    conf[~frame_ok] = 0.0

    # keypoint subset. Named HALPE subsets are only valid for the canonical
    # 136-keypoint AlphaPose layout; non-HALPE extractor sweeps should use all.
    if cfg.keypoint_subset == "all":
        idx = list(range(kps.shape[1]))
    elif cfg.keypoint_subset.startswith("cwb_"):
        # COCO-WholeBody-133 (DWPose/RTMW) subset for the estimator-only ablation.
        if kps.shape[1] != 133:
            raise ValueError(
                f"keypoint_subset={cfg.keypoint_subset!r} requires COCO-WholeBody-133; "
                f"got {kps.shape[1]} keypoints."
            )
        idx = KP_SUBSETS[cfg.keypoint_subset]
    else:
        if kps.shape[1] != NUM_KEYPOINTS:
            raise ValueError(
                f"keypoint_subset={cfg.keypoint_subset!r} requires HALPE-{NUM_KEYPOINTS}; "
                f"got {kps.shape[1]} keypoints. Use --keypoint_subset all."
            )
        idx = KP_SUBSETS[cfg.keypoint_subset]
    xy = xy[:, idx, :]
    conf = conf[:, idx, :]

    feats = np.concatenate([xy, conf], axis=-1) if cfg.use_confidence else xy
    feats = feats.reshape(T, -1).astype(np.float32)

    # drop fully-invalid frames, then temporal cap by uniform subsample
    feats = feats[frame_ok]
    if feats.shape[0] == 0:
        # Every frame failed validity. This is never a recoverable per-clip condition: it means the
        # normalisation is wrong for this data. The usual cause is coord_norm="bbox" on DWPose npz,
        # whose BBOX is all zeros by construction, so box_ok is False everywhere. Returning a single
        # zero frame here (the behaviour before the 2026-09-24 audit) let a whole run train and decode
        # on no input at all and still print a plausible BLEU.
        raise ValueError(
            "normalize_clip: all frames invalid, so this clip carries no input. "
            f"coord_norm={cfg.coord_norm!r}, bbox_format={cfg.bbox_format!r}, frames={T}. "
            "If coord_norm='bbox' on DWPose keypoints, that is the cause: the DWPose extractor writes an "
            "all-zero BBOX and every frame fails the positive width/height test. Use --coord_norm frame."
        )
    if feats.shape[0] > cfg.max_src_frames:
        sel = np.linspace(0, feats.shape[0] - 1, cfg.max_src_frames).round().astype(int)
        feats = feats[sel]
    if getattr(cfg, "temporal_pool", 1) > 1 and feats.shape[0] >= cfg.temporal_pool:
        # Temporal compression: average consecutive frames toward sign rate (~2-4 signs/s vs ~25 fps).
        # Attacks the frames-per-sign redundancy behind posterior collapse. Interface-preserving:
        # only the input sequence gets shorter; CTC/decoder plumbing is unchanged.
        k = int(cfg.temporal_pool)
        n = (feats.shape[0] // k) * k
        feats = feats[:n].reshape(n // k, k, feats.shape[1]).mean(axis=1).astype(np.float32)
    if cfg.use_velocity:                          # motion cue: per-frame temporal delta
        vel = np.zeros_like(feats)
        vel[1:] = feats[1:] - feats[:-1]
        feats = np.concatenate([feats, vel], axis=1)
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


def load_precomputed_features(path: str, key: str = "FEATURES") -> np.ndarray:
    """Load a finite ``(time, feature)`` sequence from an NPZ artifact."""
    if not path.endswith(".npz"):
        raise ValueError(f"precomputed visual features must be NPZ files, got {path}")
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"{path} has keys {payload.files}, not requested {key!r}")
        features = np.asarray(payload[key], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError(f"expected {key} with shape (T,F), got {features.shape} in {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite values in {key} from {path}")
    return features


# ── training-only pose augmentation ────────────────────────────────────────
def augment_pose(kps: np.ndarray, bbox: np.ndarray, cfg: Config,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Perturb one TRAINING clip's raw keypoints before normalisation (off unless cfg.pose_aug).

    WHY (2026-09-15): the translation bottleneck is sign identification under conversational
    production variability, and the stitched synthetic signs are ~7x LESS variable than real
    signing (E0c). Nothing in this pipeline augmented pose. Three standard label-preserving
    perturbations, each drawn once per clip from a (seed, epoch, clip) generator:
      * speed    -- nearest-frame temporal resampling by a factor in [1-aug_speed, 1+aug_speed]
                    (signing rate varies across signers; nearest-frame keeps all-zero frames exact,
                    so drop_zero_frames still masks them);
      * rotation -- one in-plane rotation of every frame about the clip's keypoint centroid;
      * scale    -- independent x / y scaling (camera aspect / perspective).
    Under coord_norm="frame" each frame is standardised with ONE pooled mean/std over x and y, which
    cancels a uniform scale but not a rotation or an anisotropic scale, so both survive into the
    features. Intended for coord_norm="frame" (the BBOX is resampled in time but not rotated).
    Only non-zero keypoints move; all-zero frames and keypoints stay exactly zero.
    """
    kps = np.array(kps, dtype=np.float32, copy=True)
    bbox = np.asarray(bbox, dtype=np.float32)
    T = kps.shape[0]
    if cfg.aug_speed > 0 and T > 1:
        factor = rng.uniform(1.0 - cfg.aug_speed, 1.0 + cfg.aug_speed)
        new_T = max(2, int(round(T * factor)))
        sel = np.linspace(0, T - 1, new_T).round().astype(int)
        kps = kps[sel]
        bbox = bbox.reshape(T, -1)[sel]
    if cfg.aug_rotate_deg > 0 or cfg.aug_scale > 0:
        xy = kps[..., :2]
        live = np.abs(xy).sum(axis=-1) > 1e-6          # (T, K): non-zero keypoints only
        if live.any():
            centre = xy[live].mean(axis=0)
            theta = np.deg2rad(rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg))
            sx = rng.uniform(1.0 - cfg.aug_scale, 1.0 + cfg.aug_scale)
            sy = rng.uniform(1.0 - cfg.aug_scale, 1.0 + cfg.aug_scale)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype=np.float32)
            affine = (rot @ np.diag([sx, sy])).astype(np.float32)
            moved = (xy - centre) @ affine.T + centre
            kps[..., :2] = np.where(live[..., None], moved, xy)
    return kps, bbox


# COCO-WholeBody-133 hands: left 91-111, right 112-132. With keypoint_subset="all" and use_confidence each keypoint is
# (x, y, conf) in the feature row, so a hand occupies feature dims 3*start .. 3*(start + 21).
HAND_FEATURE_SLICES = (slice(91 * 3, 112 * 3), slice(112 * 3, 133 * 3))


def mask_hands(feats: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Zero hand features on random frames and in one random span of one TRAINING clip (off unless aug_hand_* > 0).

    WHY (2026-09-16, research round 2 "R2-HM"): of the label-safe perturbations probed on trained checkpoints, only
    hand loss moves validation CE (both hands dropped on 20% of frames: +0.017 to +0.043 nats); jitter, frame drop and
    time warp do not. Applied AFTER normalize_clip because zeroing raw pixels would shift every body coordinate through
    the pooled frame mean/std. Under coord_norm="frame" x = y = 0 is the frame's pooled mean and conf = 0 is how a
    missing detection already looks. Each hand is dropped on each frame with probability aug_hand_drop, then one
    contiguous both-hands span of floor(U(0, aug_hand_span) * T) frames is zeroed. Frames stay valid (no padding).
    """
    if not (cfg.keypoint_subset == "all" and cfg.num_keypoints == 133 and cfg.use_confidence
            and not cfg.use_velocity and int(getattr(cfg, "temporal_pool", 1)) == 1):
        raise ValueError("hand masking needs keypoint_subset='all', num_keypoints=133, use_confidence=True, "
                         "use_velocity=False and temporal_pool=1")
    if feats.ndim != 2 or feats.shape[1] != 399:
        raise ValueError(f"hand masking expects (T, 399) features, got {feats.shape}")
    if not (0.0 <= cfg.aug_hand_drop <= 0.5 and 0.0 <= cfg.aug_hand_span <= 0.5):
        raise ValueError("aug_hand_drop and aug_hand_span must lie in [0, 0.5]")
    feats = feats.copy()
    T = feats.shape[0]
    for dims in HAND_FEATURE_SLICES:
        feats[rng.random(T) < cfg.aug_hand_drop, dims] = 0.0
    span = int(np.floor(rng.uniform(0.0, cfg.aug_hand_span) * T))
    if span > 0:
        start = int(rng.integers(0, T - span + 1))
        feats[start:start + span, HAND_FEATURE_SLICES[0].start:HAND_FEATURE_SLICES[1].stop] = 0.0
    return feats


def segment_tempo(kps: np.ndarray, bbox: np.ndarray, segs, x: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Resample each SIGN of one stitched TRAINING clip by its own tempo factor (off unless aug_segment_speed > 0).

    WHY (wave 12c "ST", from research round 2 "R2-ST"): a stitched sign has one fixed duration in every epoch, and the
    clip-level aug_speed changes all signs of a clip together, so nothing varies tempo WITHIN a clip. `segs` holds the
    [start_frame, end_frame] (inclusive) of every sign, from the stitcher's sidecar. Each sign of n frames becomes
    max(min(n, 4), round(n * U[1-x, 1+x])) frames by nearest-frame selection that keeps its first and last frame, so
    the joins are exactly the stored ones; frames outside the signs (the blended transitions) are copied unchanged.
    A sidecar that does not describe this clip (last sign not ending on the last frame) is a hard error.
    """
    T = kps.shape[0]
    if not 0.0 < x <= 0.5:
        raise ValueError(f"aug_segment_speed must lie in (0, 0.5], got {x}")
    if not segs or segs[-1][1] != T - 1:
        raise ValueError(f"segment bounds do not describe this clip: {T} frames, last sign ends at "
                         f"{segs[-1][1] if segs else None}")
    idx, prev = [], 0
    for s, e in segs:
        if s < prev or e < s:
            raise ValueError(f"segment bounds are not increasing: {segs}")
        idx.extend(range(prev, s))                       # transition frames, unchanged
        n = e - s + 1
        m = max(min(n, 4), int(round(n * rng.uniform(1.0 - x, 1.0 + x))))
        idx.extend(np.linspace(s, e, m).round().astype(int).tolist())
        prev = e + 1
    sel = np.asarray(idx, dtype=int)
    return np.asarray(kps)[sel], np.asarray(bbox).reshape(T, -1)[sel]


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
        # Multi-reference target augmentation (off unless a caller attaches a map).
        self.paraphrases: dict = {}
        self.paraphrase_prob: float = 0.0
        # Training-only pose augmentation: train_ctc sets this on the TRAIN loader only, so val/test
        # and every eval tool construct datasets with it False and never augment.
        self.pose_augment: bool = False
        self.epoch: int = 0
        # Per-sign tempo (wave 12c): empty unless a caller attaches segment bounds to the TRAIN dataset.
        self.segments: dict = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _augment_text(self, clip: str, text: str) -> str:
        """Swap in an LLM paraphrase of the reference for this epoch.

        Sampling is a hash of (seed, epoch, clip) rather than RNG state so it is
        reproducible and worker-independent, matching train_ctc's pseudo_token_drop.
        """
        alts = self.paraphrases.get(" ".join(text.lower().split()))
        if not alts:
            return text
        digest = hashlib.blake2b(
            f"{self.cfg.seed}:{self.epoch}:{clip}".encode("utf-8"), digest_size=16
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if draw >= self.paraphrase_prob:
            return text
        return alts[int.from_bytes(digest[8:], "big") % len(alts)]

    def _pose_rng(self, clip: str) -> np.random.Generator:
        """Per-(seed, epoch, clip) generator: reproducible and identical across DataLoader workers."""
        digest = hashlib.blake2b(
            f"{self.cfg.seed}:{self.epoch}:{clip}:pose_aug".encode("utf-8"), digest_size=8
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _mask_rng(self, clip: str) -> np.random.Generator:
        """Separate generator for hand masking, so the augment_pose draws are identical with or without it."""
        digest = hashlib.blake2b(
            f"{self.cfg.seed}:{self.epoch}:{clip}:mask".encode("utf-8"), digest_size=8
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def attach_segments(self, path: str) -> int:
        """Load sign bounds for the stitched rows of THIS manifest; returns how many rows have them (0 is an error)."""
        import json
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
        clips = {clip for clip, _ in self.samples}
        self.segments = {c: [(int(s), int(e)) for s, e in v] for c, v in table.items() if c in clips}
        if not self.segments:
            raise ValueError(f"{path}: none of its {len(table)} clips is in this manifest (keys must be the manifest's "
                             "clip paths)")
        return sum(clip in self.segments for clip, _ in self.samples)

    def _tempo_rng(self, clip: str) -> np.random.Generator:
        """Separate generator for per-sign tempo, so the augment_pose and mask draws are identical with or without it."""
        digest = hashlib.blake2b(
            f"{self.cfg.seed}:{self.epoch}:{clip}:segspeed".encode("utf-8"), digest_size=8
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        clip, text = self.samples[i]
        if self.paraphrase_prob > 0 and self.paraphrases:
            text = self._augment_text(clip, text)
        path = os.path.join(self.root, clip) if self.root else clip
        if self.cfg.input_representation == "precomputed":
            feats = load_precomputed_features(
                path, self.cfg.precomputed_feature_key)
            if feats.shape[1] != self.cfg.feature_dim():
                raise ValueError(
                    f"precomputed feature width {feats.shape[1]} != configured "
                    f"{self.cfg.feature_dim()} in {path}"
                )
            if feats.shape[0] > self.cfg.max_src_frames:
                selected = np.linspace(
                    0, feats.shape[0] - 1, self.cfg.max_src_frames
                ).round().astype(int)
                feats = feats[selected]
        else:
            kps, bbox = load_pose(path)
            if self.pose_augment and self.segments and clip in self.segments:
                kps, bbox = segment_tempo(kps, bbox, self.segments[clip],
                                          float(getattr(self.cfg, "aug_segment_speed", 0.0)), self._tempo_rng(clip))
            if self.pose_augment and getattr(self.cfg, "pose_aug", False):
                kps, bbox = augment_pose(kps, bbox, self.cfg, self._pose_rng(clip))
            if getattr(self.cfg, "w13_input", ""):          # wave 13 clip-level transforms (off by default)
                from .w13_input import transform as w13_transform
                feats = w13_transform(kps, self.cfg)
            else:
                feats, _ = normalize_clip(kps, bbox, self.cfg)
            if (self.pose_augment and getattr(self.cfg, "pose_aug", False)
                    and (getattr(self.cfg, "aug_hand_drop", 0.0) > 0 or getattr(self.cfg, "aug_hand_span", 0.0) > 0)):
                feats = mask_hands(feats, self.cfg, self._mask_rng(clip))
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
