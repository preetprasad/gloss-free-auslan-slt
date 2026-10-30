#!/usr/bin/env bash
# Export the official Auslan-Daily Communication train/validation/test splits
# from the completed same-language pose-MAE checkpoint.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
checkpoint="${AUSLAN_MAE_CHECKPOINT:-${project_dir}/auslan_pose_mae_pretrain_v2/out/best.pt}"
source_manifests="${AUSLAN_MAE_SOURCE_MANIFEST_ROOT:-${project_dir}/comm_run/manifests}"
destination_root="${AUSLAN_MAE_FEATURE_ROOT:-${project_dir}/pose_mae_comm}"
extract_device="${AUSLAN_MAE_EXTRACT_DEVICE:-auto}"
extract_batch="${AUSLAN_MAE_EXTRACT_BATCH_SIZE:-32}"

mkdir -p "${destination_root}/features" "${destination_root}/manifests"
for split in train val test; do
  python -u "${script_dir}/tools/extract_pose_mae_features.py" \
    --manifest "${source_manifests}/${split}.tsv" \
    --checkpoint "${checkpoint}" \
    --output-dir "${destination_root}/features/${split}" \
    --output-manifest "${destination_root}/manifests/${split}.tsv" \
    --batch-size "${extract_batch}" \
    --device "${extract_device}"
done
