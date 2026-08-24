#!/bin/bash -l
# P5b: extract a face-only DINO feature bank for train+validation clips.
# It deliberately excludes test.tsv: this is preparation for validation-only
# architecture selection, not an official-test evaluation.
#SBATCH --job-name=auslan-face-features
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-15%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate

DE=/scratch/pmc099/pprasad/dwpose
export RGB_DIR="$DE/rgb_clips"
export DWPOSE_DIR="$DE/dwpose_clips"
export OUT_DIR="$DE/face_feats"
cd /group/pmc099/pprasad/code/phase2_baseline

python tools/extract_face_features.py \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 16 --frames 24 \
  --manifest "$DE/manifests/train.tsv" --manifest "$DE/manifests/val.tsv"
echo "FACE_FEATURE_SHARD_DONE shard=${SLURM_ARRAY_TASK_ID} test_set_used=false"
