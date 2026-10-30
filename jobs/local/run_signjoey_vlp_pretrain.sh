#!/usr/bin/env bash
# Stage 1 of the evidence-backed improvement path: contrastively pretrain the
# exact SignJoey pose encoder against Communication subtitles before translation.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/comm_signjoey_vlp_pretrain}"
auslan_epochs="${AUSLAN_EPOCHS:-20}"
auslan_resume_checkpoint="${AUSLAN_RESUME_CHECKPOINT:-}"

mkdir -p "${run_dir}/out"
resume_args=()
if [[ -n "${auslan_resume_checkpoint}" ]]; then
  resume_args=(--resume_checkpoint "${auslan_resume_checkpoint}")
fi

python -u "${script_dir}/pretrain_vlp.py" \
  --model_arch signjoey_slt \
  --train_manifest "${project_dir}/comm_run/manifests/train.tsv" \
  --val_manifest "${project_dir}/comm_run/manifests/val.tsv" \
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
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --epochs "${auslan_epochs}" \
  --lr_schedule cosine \
  --warmup_steps 500 \
  --grad_clip 1.0 \
  --vlp_proj 256 \
  --vlp_temp 0.07 \
  --num_workers 0 \
  --device auto \
  --out_dir "${run_dir}/out" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/pretrain.log"
