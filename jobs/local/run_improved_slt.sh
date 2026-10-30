#!/usr/bin/env bash
# Controlled improvement run: retain the source-identical 3+3 SL-Transformer,
# correct Auslan xywh pose semantics, suppress unreliable coordinates using
# AlphaPose confidence, and replace the unstable batch-normalised objective with
# the locally stable smoothed token objective plus warmup/cosine scheduling.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/comm_signjoey_improved_confidence_full}"
auslan_epochs="${AUSLAN_EPOCHS:-60}"
auslan_resume_checkpoint="${AUSLAN_RESUME_CHECKPOINT:-}"
auslan_init_encoder="${AUSLAN_INIT_ENCODER:-}"
auslan_encoder_lr_scale="${AUSLAN_ENCODER_LR_SCALE:-1.0}"
auslan_vlp_aux_checkpoint="${AUSLAN_VLP_AUX_CHECKPOINT:-}"
auslan_vlp_aux_weight="${AUSLAN_VLP_AUX_WEIGHT:-0.0}"

mkdir -p "${run_dir}/out"
resume_args=()
if [[ -n "${auslan_resume_checkpoint}" ]]; then
  resume_args=(--resume_checkpoint "${auslan_resume_checkpoint}")
fi
init_args=()
if [[ -n "${auslan_init_encoder}" ]]; then
  init_args=(--init_encoder "${auslan_init_encoder}")
fi
vlp_aux_args=()
if [[ -n "${auslan_vlp_aux_checkpoint}" ]]; then
  vlp_aux_args=(--vlp_aux_checkpoint "${auslan_vlp_aux_checkpoint}")
fi

python -u "${script_dir}/train.py" \
  --model_arch signjoey_slt \
  --train_manifest "${project_dir}/comm_run/manifests/train.tsv" \
  --val_manifest "${project_dir}/comm_run/manifests/val.tsv" \
  --test_manifest "${project_dir}/comm_run/manifests/test.tsv" \
  --keypoint_subset body_hands \
  --use_confidence true \
  --coord_norm bbox \
  --bbox_format xywh \
  --conf_threshold 0.1 \
  --drop_zero_frames true \
  --max_src_frames 400 \
  --text_tokenization whitespace \
  --visual_feature_dim 0 \
  --d_model 512 \
  --nhead 8 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --dim_feedforward 2048 \
  --dropout 0.1 \
  --embedding_dropout 0.1 \
  --source_embedding_activation softsign \
  --target_embedding_dim 512 \
  --target_embedding_activation softsign \
  --position_max_len 5000 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --lr 0.0003 \
  --weight_decay 0.0001 \
  --epochs "${auslan_epochs}" \
  --lr_schedule cosine \
  --warmup_steps 500 \
  --label_smoothing 0.1 \
  --loss_normalization tokens \
  --grad_clip 1.0 \
  --encoder_lr_scale "${auslan_encoder_lr_scale}" \
  --vlp_aux_weight "${auslan_vlp_aux_weight}" \
  --decode_max_len 30 \
  --num_workers 0 \
  --device auto \
  --reference_bleu4 9.20 \
  --gate_fraction 0.9 \
  --out_dir "${run_dir}/out" \
  ${init_args[@]+"${init_args[@]}"} \
  ${vlp_aux_args[@]+"${vlp_aux_args[@]}"} \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/train.log"
