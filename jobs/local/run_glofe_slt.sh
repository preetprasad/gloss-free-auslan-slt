#!/usr/bin/env bash
# Controlled input-pipeline experiment: public GloFE WLASL-pretrained
# CTR-GCN/MS-TCN features -> source-identical 3+3 SL-Transformer.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/comm_glofe_slt_full}"
train_manifest="${GLOFE_TRAIN_MANIFEST:-${project_dir}/glofe_full/manifests/train.tsv}"
val_manifest="${GLOFE_VAL_MANIFEST:-${project_dir}/glofe_full/manifests/val.tsv}"
test_manifest="${GLOFE_TEST_MANIFEST:-${project_dir}/glofe_full/manifests/test.tsv}"
auslan_epochs="${AUSLAN_EPOCHS:-60}"
auslan_batch="${AUSLAN_BATCH_SIZE:-16}"
auslan_accum="${AUSLAN_GRAD_ACCUM_STEPS:-2}"
auslan_device="${AUSLAN_DEVICE:-auto}"
auslan_resume_checkpoint="${AUSLAN_RESUME_CHECKPOINT:-}"

for manifest in "${train_manifest}" "${val_manifest}" "${test_manifest}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing GloFE feature manifest: ${manifest}" >&2
    exit 1
  fi
done

mkdir -p "${run_dir}/out"
resume_args=()
if [[ -n "${auslan_resume_checkpoint}" ]]; then
  resume_args=(--resume_checkpoint "${auslan_resume_checkpoint}")
fi

python -u "${script_dir}/train.py" \
  --model_arch signjoey_slt \
  --input_representation precomputed \
  --precomputed_feature_key FEATURES \
  --precomputed_feature_dim 1024 \
  --train_manifest "${train_manifest}" \
  --val_manifest "${val_manifest}" \
  --test_manifest "${test_manifest}" \
  --max_src_frames 128 \
  --max_tgt_tokens 80 \
  --decode_max_len 30 \
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
  --batch_size "${auslan_batch}" \
  --length_bucket_batches true \
  --grad_accum_steps "${auslan_accum}" \
  --lr 0.001 \
  --weight_decay 0.001 \
  --epochs "${auslan_epochs}" \
  --lr_schedule plateau \
  --plateau_patience 8 \
  --plateau_factor 0.7 \
  --lr_min 1e-7 \
  --label_smoothing 0.0 \
  --loss_normalization batch \
  --num_workers 0 \
  --device "${auslan_device}" \
  --reference_bleu4 9.20 \
  --gate_fraction 0.9 \
  --out_dir "${run_dir}/out" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/train.log"
