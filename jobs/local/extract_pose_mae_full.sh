#!/usr/bin/env bash
# Export train/validation/test temporal features after Auslan pose-MAE pretraining.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
checkpoint="${AUSLAN_MAE_CHECKPOINT:-${project_dir}/auslan_pose_mae_pretrain_v2/out/best.pt}"
destination_root="${AUSLAN_MAE_FEATURE_ROOT:-${project_dir}/pose_mae_full}"
extract_device="${AUSLAN_MAE_EXTRACT_DEVICE:-auto}"
extract_batch="${AUSLAN_MAE_EXTRACT_BATCH_SIZE:-32}"

mkdir -p "${destination_root}/features" "${destination_root}/manifests"
for split in train val test; do
  python -u "${script_dir}/tools/extract_pose_mae_features.py" \
    --manifest "${project_dir}/full_run/manifests/${split}.tsv" \
    --checkpoint "${checkpoint}" \
    --output-dir "${destination_root}/features/${split}" \
    --output-manifest "${destination_root}/manifests/${split}.tsv" \
    --batch-size "${extract_batch}" \
    --device "${extract_device}"
done
