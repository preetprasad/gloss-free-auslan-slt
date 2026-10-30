#!/bin/bash -l
# Validation-only P1: order-INVARIANT pseudo-content supervision on DWPose+RGB.
# It never reads the official test manifest.  This directly tests whether the
# English-order pseudo-label problem is responsible for CTC's limited benefit.
# Submit: cd /group/pmc099/pprasad/code/phase2_baseline && sbatch kaya/presence_set_rgb.sbatch
#
# Comparator: original RGB+single-CTC training recipe (same VLP warm start,
# seed, batch size, schedule).  Selection is validation BLEU only.
#
# Do not alter into a test-scoring script: test is intentionally held back until
# the method is chosen and a clean protocol is agreed.
#SBATCH --job-name=auslan-presence-p1
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate

DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"

INIT="$G/comm_dwpose_vlp_ctc/best.pt"
echo "=== P1 order-invariant pseudo-label presence supervision ==="
echo "Validation only; no official test manifest is supplied."
# First touch real DWPose and RGB files with the exact production environment.
# A short GPU smoke gate prevents a failed full allocation because of a cache,
# checkpoint, or shape mismatch.
python train_presence.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$INIT" --seed 42 \
  --batch_size 32 --epochs 1 --limit_samples 64 --presence_weight 0.30 --presence_smooth_weight 0.01 \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 8 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgb_presence_p1_smoke_${SLURM_JOB_ID}"
echo "P1_GPU_SMOKE_PASSED"

python train_presence.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$INIT" --seed 42 \
  --batch_size 32 --epochs 40 --presence_weight 0.30 --presence_smooth_weight 0.01 \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgb_presence_p1"

cp -r "$DE/runs/comm_dwpose_rgb_presence_p1" "$G/"
echo "P1_DONE validation_only=true"
