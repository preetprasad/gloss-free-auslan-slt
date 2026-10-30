#!/usr/bin/env bash
# Same-language, same-detector masked pose pretraining.  This stage uses only
# official Auslan-Daily train rows (Communication + News), never val/test poses.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/auslan_pose_mae_pretrain}"
mae_epochs="${AUSLAN_MAE_EPOCHS:-20}"
mae_batch="${AUSLAN_MAE_BATCH_SIZE:-16}"
mae_accum="${AUSLAN_MAE_GRAD_ACCUM_STEPS:-2}"
mae_device="${AUSLAN_MAE_DEVICE:-auto}"
mae_resume="${AUSLAN_MAE_RESUME_CHECKPOINT:-}"

mkdir -p "${run_dir}"
resume_args=()
if [[ -n "${mae_resume}" ]]; then
  resume_args=(--resume_checkpoint "${mae_resume}")
fi

python -u "${script_dir}/pretrain_pose_mae.py" \
  --train_manifest "${project_dir}/full_run/manifests/train.tsv" \
  --val_manifest "${project_dir}/full_run/manifests/val.tsv" \
  --keypoint_subset body_hands \
  --use_confidence true \
  --coord_norm bbox \
  --bbox_format xywh \
  --center_to_unit true \
  --conf_threshold 0.1 \
  --drop_zero_frames true \
  --max_src_frames 128 \
  --batch_size "${mae_batch}" \
  --length_bucket_batches true \
  --grad_accum_steps "${mae_accum}" \
  --lr 0.0003 \
  --weight_decay 0.0001 \
  --epochs "${mae_epochs}" \
  --warmup_steps 500 \
  --dropout 0.1 \
  --grad_clip 1.0 \
  --num_workers 0 \
  --device "${mae_device}" \
  --mae_hidden 256 \
  --mae_encoder_layers 4 \
  --mae_heads 8 \
  --mae_ff 1024 \
  --mae_decoder_hidden 256 \
  --mae_decoder_layers 2 \
  --mae_patch_size 4 \
  --mae_mask_ratio 0.5 \
  --mae_output_dim 1024 \
  --mae_min_confidence 0.1 \
  --out_dir "${run_dir}/out" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/train.log"
