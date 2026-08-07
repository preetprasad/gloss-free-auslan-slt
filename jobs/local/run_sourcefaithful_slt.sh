#!/usr/bin/env bash
# Closest reproducible Auslan pose adaptation of the pinned public SignJoey SLT.
# The Auslan-Daily authors did not release their pose adapter. This command uses
# the least assumptive interpretation: flattened raw AlphaPose body+hand points
# feed the public configurable SpatialEmbedding directly.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/comm_signjoey_sourcefaithful_raw_full}"
auslan_epochs="${AUSLAN_EPOCHS:-60}"
auslan_test_manifest="${AUSLAN_TEST_MANIFEST-${project_dir}/comm_run/manifests/test.tsv}"
auslan_resume_checkpoint="${AUSLAN_RESUME_CHECKPOINT:-}"

mkdir -p "${run_dir}/out"
resume_args=()
if [[ -n "${auslan_resume_checkpoint}" ]]; then
  resume_args=(--resume_checkpoint "${auslan_resume_checkpoint}")
fi

python -u "${script_dir}/train.py" \
  --model_arch signjoey_slt \
  --train_manifest "${project_dir}/comm_run/manifests/train.tsv" \
  --val_manifest "${project_dir}/comm_run/manifests/val.tsv" \
  --test_manifest "${auslan_test_manifest}" \
  --keypoint_subset body_hands \
  --use_confidence false \
  --coord_norm none \
  --drop_zero_frames false \
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
  --lr 0.001 \
  --weight_decay 0.001 \
  --epochs "${auslan_epochs}" \
  --lr_schedule plateau \
  --plateau_patience 8 \
  --plateau_factor 0.7 \
  --lr_min 1e-7 \
  --label_smoothing 0.0 \
  --loss_normalization batch \
  --decode_max_len 30 \
  --num_workers 0 \
  --device auto \
  --reference_bleu4 9.20 \
  --gate_fraction 0.9 \
  --out_dir "${run_dir}/out" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/train.log"
