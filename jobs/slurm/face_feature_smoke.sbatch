#!/bin/bash -l
# P5a: validate DWPose-localised face crops before full extraction or model work.
# This job writes no model score and never reads a test manifest.
#SBATCH --job-name=auslan-face-feature-smoke
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
export RGB_DIR=/scratch/pmc099/pprasad/dwpose/rgb_clips
export DWPOSE_DIR=/scratch/pmc099/pprasad/dwpose/dwpose_clips
export OUT_DIR=/scratch/pmc099/pprasad/dwpose/face_feats_smoke
cd /group/pmc099/pprasad/code/phase2_baseline
python tools/extract_face_features.py --frames 24 --limit 128
echo "FACE_FEATURE_SMOKE_DONE test_set_used=false"
