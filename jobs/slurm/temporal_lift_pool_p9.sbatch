#!/bin/bash -l
# P9: P3-matched adaptive 2x decoder-memory pooling, validation only.
# CTC remains on the unpooled pose memory. Never add test.tsv here.
#SBATCH --job-name=auslan-liftpool-p9
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

echo "=== P9 TLP-inspired adaptive decoder-memory pooling ==="
echo "Validation only: train.tsv + val.tsv, no official test manifest."

python train_temporal_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$INIT" --seed 42 --batch_size 32 --epochs 1 --limit_samples 128 \
  --ctc_weight 0.30 --temporal_adapter pyramid_pool --lr 5e-4 --lr_schedule cosine --warmup_steps 8 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_liftpool_ctc_p9_smoke_${SLURM_JOB_ID}"
echo "P9_GPU_SMOKE_PASSED metrics_selection_valid=false"

python train_temporal_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$INIT" --seed 42 --batch_size 32 --epochs 40 \
  --ctc_weight 0.30 --temporal_adapter pyramid_pool --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_liftpool_ctc_p9"

cp -r "$DE/runs/comm_dwpose_liftpool_ctc_p9" "$G/"
echo "P9_DONE validation_only=true test_set_used=false ctc_memory=unpooled_pose"
