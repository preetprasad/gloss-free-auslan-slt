#!/usr/bin/env bash
# Resumable extraction of public GloFE features for the fixed Communication
# split.  Keep this on CPU so an active MPS training run retains the GPU.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
source_root="${project_dir}/comm_run/manifests"
destination_root="${project_dir}/glofe_full"
glofe_repo="${project_dir}/tmp/reference_repos/GloFE"

mkdir -p "${destination_root}/manifests"

for split in train val test; do
  source_manifest="${source_root}/${split}.tsv"
  if [[ ! -f "${source_manifest}" ]]; then
    echo "Missing fixed Communication manifest: ${source_manifest}" >&2
    exit 1
  fi
  nice -n 10 python -u "${script_dir}/tools/extract_glofe_features.py" \
    --input_manifest "${source_manifest}" \
    --output_manifest "${destination_root}/manifests/${split}.tsv" \
    --output_dir "${destination_root}/features/${split}" \
    --glofe_repo "${glofe_repo}" \
    --device cpu
done

echo "[done] full fixed-split GloFE feature extraction"
