"""Configuration for the Phase 2 gloss-free Auslan-Daily SLT baseline.

Phase 2 framing (see the project log): this is the *first* gloss-free continuous
Sign Language Translation (SLT) baseline on Auslan-Daily, evaluated against a
BLEU-4 decision gate. Nothing here is a final architecture — it is the minimal,
correct pipeline needed to produce a first WER/BLEU number (research gap G3) and
decide whether to proceed.

All values are plain dataclass fields so this file is import-safe with no torch
dependency; train.py applies them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ── HALPE-136 keypoint layout (AlphaPose whole-body) ────────────────────────
# Standard AlphaPose HALPE-136 index groups:
#   body       0..25   (26)
#   face      26..93   (68)
#   left hand 94..114  (21)
#   right hand 115..135 (21)
# NOTE: the project memory noted hands "≈91..135"; the canonical layout above
# puts hands at 94..135. The baseline defaults to using ALL 136 keypoints, so it
# is correct regardless of the exact hand start index — but confirm the layout
# against the real tensors (run inspect_datasets.ipynb) before using any subset.
NUM_KEYPOINTS = 136
KP_GROUPS = {
    "body": list(range(0, 26)),
    "face": list(range(26, 94)),
    "left_hand": list(range(94, 115)),
    "right_hand": list(range(115, 136)),
}
KP_SUBSETS = {
    "all": list(range(0, 136)),
    "body": KP_GROUPS["body"],
    "hands": KP_GROUPS["left_hand"] + KP_GROUPS["right_hand"],
    "body_hands": KP_GROUPS["body"] + KP_GROUPS["left_hand"] + KP_GROUPS["right_hand"],
}


@dataclass
class Config:
    # ── data ────────────────────────────────────────────────────────────────
    # A manifest is a TSV with two columns: <clip_path>\t<target_text>.
    # Decoupling from Drive layout keeps this runnable anywhere (Colab/Kaya).
    train_manifest: str = "manifests/train.tsv"
    val_manifest: str = "manifests/val.tsv"
    test_manifest: Optional[str] = None

    keypoint_subset: str = "all"        # one of KP_SUBSETS
    use_confidence: bool = True          # keep the AlphaPose confidence channel
    coord_norm: str = "bbox"             # "bbox" | "frame" | "none"
    bbox_format: str = "xyxy"            # "xyxy" | "xywh"
    center_to_unit: bool = True          # map normalised coords to [-1, 1]
    conf_threshold: float = 0.0          # zero-out coords below this confidence
    drop_zero_frames: bool = True        # mask all-zero frames (frame 0 issue)
    max_src_frames: int = 256            # temporal cap (subsample if longer)
    max_tgt_tokens: int = 64

    # ── model ───────────────────────────────────────────────────────────────
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # ── optimisation ─────────────────────────────────────────────────────────
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    min_freq: int = 1                    # vocab: min word frequency to keep

    # ── Phase 2 decision gate (research gap G3) ──────────────────────────────
    # The gate: does the gloss-free pose baseline clear this BLEU-4 on the
    # Auslan-Daily test split? Placeholder threshold — set with the supervisor
    # once the first numbers are in. Recorded here so the gate is explicit.
    bleu4_gate: float = 2.0

    # ── runtime ──────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 2
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    out_dir: str = "runs/baseline"

    def feature_dim(self) -> int:
        k = len(KP_SUBSETS[self.keypoint_subset])
        return k * (3 if self.use_confidence else 2)

    def to_dict(self) -> dict:
        return asdict(self)
