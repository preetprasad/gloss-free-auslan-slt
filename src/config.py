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
# COCO-WholeBody-133 (DWPose / RTMW) layout — additive, for the estimator-only ablation.
# Lets us reduce DWPose to the same articulator set as the AlphaPose baseline (body+hands,
# no face) so the ONLY difference is the estimator. Does not touch the HALPE paths above.
CWB_GROUPS = {
    "body": list(range(0, 17)),        # 17 COCO body keypoints
    "feet": list(range(17, 23)),       # 6
    "face": list(range(23, 91)),       # 68
    "left_hand": list(range(91, 112)),  # 21
    "right_hand": list(range(112, 133)),  # 21
}
KP_SUBSETS = {
    "all": list(range(0, 136)),
    "body": KP_GROUPS["body"],
    "hands": KP_GROUPS["left_hand"] + KP_GROUPS["right_hand"],
    "body_hands": KP_GROUPS["body"] + KP_GROUPS["left_hand"] + KP_GROUPS["right_hand"],
    # COCO-WholeBody body+hands (no face) = 17 + 21 + 21 = 59 keypoints
    "cwb_body_hands": CWB_GROUPS["body"] + CWB_GROUPS["left_hand"] + CWB_GROUPS["right_hand"],
}


@dataclass
class Config:
    # ── data ────────────────────────────────────────────────────────────────
    # A manifest is a TSV with two columns: <clip_path>\t<target_text>.
    # Decoupling from Drive layout keeps this runnable anywhere (Colab/Kaya).
    train_manifest: str = "manifests/train.tsv"
    val_manifest: str = "manifests/val.tsv"
    test_manifest: Optional[str] = None

    # DEFAULTS DESCRIBE THE ADOPTED RECIPE (audit 2026-09-24). Until then these fields still described the
    # pre-August AlphaPose/bbox configuration, which has been dead since the DWPose switch: a reader opening
    # this file learned the wrong recipe, and any tool that forgot a flag silently ran a dead configuration.
    num_keypoints: int = 133               # DWPose COCO-WholeBody-133 (was 136 HALPE); override for other extractors
    keypoint_subset: str = "all"        # one of KP_SUBSETS
    use_confidence: bool = True          # keep the AlphaPose confidence channel
    use_velocity: bool = False           # append per-frame temporal deltas (motion) -> doubles feat dim
    temporal_pool: int = 1               # avg-pool consecutive frames by this factor (1 = off); compress toward sign rate
    ctc_freq_weight: float = 0.0         # inverse-frequency exponent for rare-gloss CTC upweighting (0 = off); targets long-tail vocab
    r_drop_alpha: float = 0.0            # R-Drop consistency regularisation weight (0 = off); attacks overfitting/exposure-bias on small data
    pose_graph: bool = False             # learned-adjacency graph conv front-end over keypoints (skeleton structure); off = flat-vector
    pose_graph_layers: int = 2           # number of graph-conv layers when pose_graph is on
    paraphrase_map: str = ""             # TSV (original-text<TAB>paraphrase, repeatable key) of LLM paraphrases; multi-reference target augmentation
    paraphrase_prob: float = 0.0         # per-sample chance of training on a paraphrase instead of the reference (0 = off)
    coord_norm: str = "frame"            # "bbox" | "frame" | "none" -- ADOPTED: per-frame standardisation.
    # NB "bbox" is unusable on DWPose npz: the extractor writes an all-zero BBOX, so every frame fails the
    # positive-width/height test and the clip collapses to a single zero frame (now a hard error, see data.py).
    # Audited against 200 real Communication clips on 2026-08-05: the released
    # tensors store [x, y, width, height].  Treating columns 3/4 as x2/y2 put
    # only 14% of confident joints inside the alleged box and produced coordinate
    # outliers up to 1.3e4.  xywh contains 89% and gives a stable ~[-2.1,2.6] range.
    bbox_format: str = "xywh"            # Auslan-Daily release: "xywh"; "xyxy" only for other extractors
    # Compatibility only for checkpoints trained before the xywh audit. Their
    # loader accepted negative xyxy spans by absolute magnitude, so evaluating
    # them with the corrected positive-span rule changes the input sequence.
    legacy_abs_bbox_validity: bool = False
    center_to_unit: bool = True          # map normalised coords to [-1, 1]
    conf_threshold: float = 0.0          # zero-out coords below this confidence
    hierarchical_pose_norm: bool = False # torso-normalise body; wrist/palm-normalise hands (improvement ablation)
    hand_local_norm: bool = False       # COCO-WholeBody-133 port of the hand half of the above.
                                        # hierarchical_pose_norm hard-requires HALPE-136 and so has
                                        # NEVER run since the DWPose switch. Motivated by measurement:
                                        # the continuous-domain identification probe (2026-09-09) finds
                                        # hands carry the most sign identity and lose the most of it
                                        # (40.0%->18.1% top-1 isolated->continuous), and hand-local
                                        # normalisation recovers +6.8 points of that, top-5 34.7->46.9.
    pose_aug: bool = False               # TRAINING-ONLY pose augmentation (speed/rotation/x-y scale); eval never augments
    aug_speed: float = 0.2               # temporal resample factor from [1-x, 1+x] (nearest-frame; 0 = off)
    aug_rotate_deg: float = 10.0         # global in-plane rotation from [-x, +x] degrees (0 = off)
    aug_scale: float = 0.1               # independent x / y scale from [1-x, 1+x] (0 = off)
    aug_hand_drop: float = 0.0           # needs pose_aug: per-frame, per-hand chance of zeroing that hand's features (0 = off)
    aug_hand_span: float = 0.0           # needs pose_aug: one both-hands zero span of up to this fraction of frames (0 = off)
    aug_segment_speed: float = 0.0       # wave 12c "ST": per-SIGN tempo on stitched TRAIN rows, each sign resampled by its own
                                         # factor from [1-x, 1+x] (nearest-frame; transitions untouched; 0 = off)
    aug_segment_file: str = ""           # JSON {clip path as in the manifest: [[start_frame, end_frame], ...]} written by
                                         # tools/w12c_stitch_sidecar.py from the stitcher's --audit_jsonl sidecar
    w13_input: str = ""                  # wave 13: "" = the adopted per-frame standardisation; "b133" = all 133 points with clip-level body
                                         # scaling; "u69" = the 69-point set with clip-level part scaling (slt/w13_input.py)
    w13_frontend: str = ""               # wave 13: "" = flat encoder input; "tc" / "sg" = part-wise temporal-conv / ST-GCN front end
                                         # (slt/w13_frontend.py); needs w13_input="u69"
    drop_zero_frames: bool = True        # mask all-zero frames (frame 0 issue)
    max_src_frames: int = 256            # temporal cap (subsample if longer)
    max_tgt_tokens: int = 64
    decode_max_len: int = 0              # 0 => max_tgt_tokens; public SLT uses 30
    # ``regex`` preserves checkpoints produced before 2026-08-05. The public
    # SignJoey word loader uses lowercase whitespace splitting, so faithful
    # reference experiments explicitly select ``whitespace``.
    text_tokenization: str = "regex"      # "regex" | "whitespace" | "sentencepiece" (GASLT)
    sentencepiece_model: str = ""          # GASLT/BPEmb .model path
    pretrained_target_embeddings: str = "" # word2vec binary aligned by BPE token string

    # Source representation. ``pose`` preserves every existing run and loads
    # HALPE KPS/BBOX arrays. ``precomputed`` loads a 2-D feature sequence from
    # an NPZ key instead; this is used to evaluate published visual front ends
    # (for example GloFE's WLASL-pretrained CTR-GCN) without changing the
    # downstream SL-Transformer/GASLT implementation.
    input_representation: str = "pose"      # "pose" | "precomputed"
    precomputed_feature_key: str = "FEATURES"
    precomputed_feature_dim: int = 0         # required when input_representation=precomputed

    # ── RGB fusion (Stage B; two-stream pose+RGB) ────────────────────────────
    # When rgb_feat_dir is set, train_ctc.py dispatches to slt.rgb_fusion, adding
    # a precomputed-RGB appearance stream (rgb_<clip>.npz key FEATURES, width
    # rgb_feature_dim) fused into the decoder. Empty => unchanged pose-only path.
    rgb_feat_dir: str = ""                    # dir of rgb_<clip>.npz DINOv2 crop features
    rgb_feature_dim: int = 1152               # 3 crops x DINOv2-small 384
    face_feat_dir: str = ""                   # optional face_<clip>.npz DINOv2 features; train/val only in P5
    face_feature_dim: int = 0                 # 384 when the controlled face visual channel is enabled
    rgb_encoder_layers: int = 2               # RGB-stream Transformer depth
    rgb_vlp_init: str = ""                    # RGB-VLP checkpoint: warm-start the fusion RGB stream
    gloss_map: str = ""                        # PGG-SLT: TSV (normalised-text<TAB>gloss string) of LLM pseudo-glosses; overrides rule-based CTC glosses

    # ── BPE + pretrained mBART decoder (GFSLT-VLP-style; the strong-single-model lever) ──
    mbart_name: str = "facebook/mbart-large-50"   # pretrained multilingual decoder + BPE tokenizer
    mbart_lora_r: int = 32                          # LoRA rank on mBART attention+FFN (higher than the failed r=16)
    mbart_train_lora: bool = True                   # Stage 2b: False = ALIGN phase (mBART frozen, train adapter+encoder only); True = fine-tune

    # ── model ───────────────────────────────────────────────────────────────
    # ``generic`` preserves every existing checkpoint. ``signjoey_slt`` and
    # ``gaslt`` reproduce the public reference stacks (pre-LN layers, masked
    # BatchNorm+Softsign embeddings; GASLT adds deformable gloss attention).
    model_arch: str = "generic"          # "generic" | "signjoey_slt" | "gaslt"
    # Optional learned per-frame pose adapter. A positive value maps the raw
    # flattened pose vector to this width before the Transformer projection.
    # This lets pose experiments reproduce the 1,024-D visual-feature interface
    # used by SL-Transformer without pretending the coordinates are CNN features.
    # Zero preserves the original direct raw-pose -> d_model baseline.
    visual_feature_dim: int = 0
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    embedding_dropout: float = 0.1
    source_embedding_activation: str = "softsign"
    target_embedding_dim: int = 512       # GASLT paper/repo uses 300; SLT uses 512
    target_embedding_activation: str = "softsign"
    # 4096 preserves legacy checkpoint buffer shapes; the pinned SignJoey
    # source uses 5000 and exact-reference commands set it explicitly.
    position_max_len: int = 4096
    gaslt_num_keys: int = 7               # best setting in GASLT (two layers: [7,7])
    gaslt_similarity_weight: float = 0.0  # 0 for SLT; GASLT paper says 1, repo YAML says 10
    semantic_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    semantic_cache: str = ""              # optional .pt text->embedding cache

    # ── optimisation ─────────────────────────────────────────────────────────
    batch_size: int = 64                  # adopted recipe (32 for RGB fusion)
    length_bucket_batches: bool = False  # public SignJoey/GASLT training iterator
    grad_accum_steps: int = 1             # optimiser update every N mini-batches; effective batch = batch_size * N
    lr: float = 5e-4                      # adopted recipe
    weight_decay: float = 1e-4
    epochs: int = 40
    warmup_steps: int = 800               # adopted recipe
    lr_schedule: str = "cosine"      # "noam" | "cosine" | "plateau"; adopted = cosine (noam was the SignJoey setting)
    plateau_patience: int = 8
    plateau_factor: float = 0.7
    lr_min: float = 1e-7
    label_smoothing: float = 0.1
    loss_normalization: str = "tokens"    # "tokens" legacy | "batch" SignJoey
    grad_clip: float = 1.0
    min_freq: int = 1                    # vocab: min word frequency to keep
    # Transfer-learning control: keep a pretrained visual encoder stable while
    # the randomly initialised translation decoder learns. 1.0 is legacy/full LR.
    encoder_lr_scale: float = 1.0
    vlp_aux_checkpoint: str = ""          # full vlp_latest.pt; keeps pose/text alignment during SLT
    vlp_aux_weight: float = 0.0           # small joint contrastive weight; 0 disables

    # SignCL contrastive regularisation (Ye et al. 2024) — fights the representation
    # -density collapse (adjacent frames pulled together, distant frames pushed apart
    # on the encoder features). Added to the CE loss: L = CE + signcl_weight * L_SignCL.
    signcl_weight: float = 0.0           # lambda; 0 = off
    signcl_margin: float = 1.0           # push margin m on L2-normalised feats (dist in [0,2])
    signcl_zipf: float = 2.3             # speech-to-gesture Zipf factor for the frame margin

    # ── Phase 2 decision gate (research gap G3) ──────────────────────────────
    # Supervisor decision (Atif, 2026-07-21): the gate is NOT a fixed number but
    # a PERCENTAGE of a reference/SOTA model's BLEU-4 on the same split.
    #   reference_bleu4 : the reference model's BLEU-4 (e.g. Auslan-Daily paper's
    #                     GASLT-Pose = 9.95 on Communication; None => use bleu4_gate)
    #   gate_fraction   : fraction of that to clear ("90% or so")
    # Effective threshold = gate_fraction * reference_bleu4 when reference set.
    reference_bleu4: Optional[float] = None
    gate_fraction: float = 0.9
    bleu4_gate: float = 2.0          # fallback fixed threshold if no reference set

    # BLEU tokenisation policy — Auslan-Daily subtitles are already tokenised
    # ("hello ."). Supervisor decision (Atif): FOLLOW THE LITERATURE so numbers are
    # comparable/publishable — detokenise both sides and use sacreBLEU's standard
    # tokeniser (13a). (Previously "none"; that was internally consistent but not
    # comparable to published BLEU, and the gate is now reference-relative.)
    bleu_tokenize: str = "13a"
    detokenize: bool = True          # rejoin "hello ." -> "hello." before scoring

    # ── runtime ──────────────────────────────────────────────────────────────
    seed: int = 42
    deterministic: bool = False   # pin cuDNN/cuBLAS kernels so a seed reproduces a run bit-for-bit
    num_workers: int = 2
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    out_dir: str = "runs/baseline"
    resume_checkpoint: str = ""            # latest.pt from an interrupted compatible run
    init_encoder: str = ""               # path to a VLP-pretrained checkpoint; loads pose-encoder weights before fine-tuning
    vocab_file: str = ""                 # JSON itos list: use this fixed vocabulary instead of building one from the train
                                         # manifest (pre-train -> fine-tune must share index order; see train_ctc.py)
    save_last: bool = False              # also save the FINAL epoch's model as last.pt (pre-training carries this forward)
    eval_every: int = 1                  # evaluate validation every N epochs and always at the last epoch
    resume_ctc_head: bool = False        # with --resume_checkpoint: also restore the CTC head (a continuation, not a fine-tune)
    cf_pref_weight: float = 0.0          # wave 12c "CPO": weight of the counterfactual-pose preference term (0 = off); needs
                                         # --resume_checkpoint, whose model is also the frozen reference
    cf_beta: float = 0.1                 # CPO: sharpness of the preference sigmoid
    cf_stop_ce: float = 0.0              # CPO: stop if validation CE has risen by more than this (nats) after epoch 3 (0 = off)
    freeze_encoder: bool = False          # keep the loaded pose encoder fixed; train only decoder/embeddings/generator

    # ── LLM decoder (mT5 + LoRA) ─────────────────────────────────────────────
    llm_name: str = "google/mt5-base"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q,v,o,wi,wo"
    llm_lr: float = 1e-4
    pose_encoder_lr_scale: float = 1.0
    max_gen_len: int = 64
    num_beams: int = 5
    length_penalty: float = 0.6

    # ── VQ tokeniser + LLM-VQ decoder ──────────────────────────────────────────
    # Stage A: VQ tokeniser over pose/RGB features
    vq_stage: str = "tokenise"        # "tokenise" | "decode" (train tokeniser vs train LLM over tokens)
    vq_modality: str = "pose"         # "pose" | "rgb" | "pose_rgb" (which features to tokenise)
    vq_encoder_ckpt: str = ""         # pretrained encoder to use as VQ front-end (VLP/RGB-VLP)
    vq_codebook_size: int = 1024
    vq_code_dim: int = 256
    vq_num_codebooks: int = 1         # residual VQ: number of codebooks (1 = standard VQ-VAE)
    vq_commitment_cost: float = 0.25
    vq_ema_decay: float = 0.99        # EMA codebook update decay (0 = no EMA, use gradient)
    vq_downsample: int = 1            # temporal downsample factor for codes (1 = per-frame)
    vq_reconstruction_weight: float = 1.0
    vq_ctc_weight: float = 0.0        # optional CTC on codes for linguistic alignment
    # Stage B: LLM decoder over VQ tokens
    vq_llm_name: str = "google/mt5-base"
    vq_llm_lr: float = 1e-4
    vq_lora_r: int = 16
    vq_lora_alpha: int = 32
    vq_lora_dropout: float = 0.05
    vq_lora_target_modules: str = "q,v,o,wi,wo"
    vq_max_gen_len: int = 64
    vq_num_beams: int = 5
    vq_length_penalty: float = 0.6
    vq_token_prefix: str = "vq_"      # prefix for VQ tokens added to tokenizer

    def feature_dim(self) -> int:
        if self.input_representation == "precomputed":
            if self.precomputed_feature_dim <= 0:
                raise ValueError(
                    "precomputed_feature_dim must be positive when "
                    "input_representation='precomputed'"
                )
            return int(self.precomputed_feature_dim)
        if self.input_representation != "pose":
            raise ValueError(
                f"input_representation must be 'pose' or 'precomputed', "
                f"got {self.input_representation!r}"
            )
        if self.w13_input in ("b133", "u69"):                 # wave 13: fixed layouts (x, y, confidence per point)
            return 3 * (133 if self.w13_input == "b133" else 69)
        if self.keypoint_subset == "all":
            k = int(self.num_keypoints)
        else:
            k = len(KP_SUBSETS[self.keypoint_subset])
        d = k * (3 if self.use_confidence else 2)
        return d * 2 if self.use_velocity else d

    def to_dict(self) -> dict:
        return asdict(self)
