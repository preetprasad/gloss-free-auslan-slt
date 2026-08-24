#!/bin/bash -l
# P6 interface smoke only: never use this partial-validation metric for selection.
#SBATCH --job-name=auslan-gated-fusion-smoke
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
python train_gated_temporal_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" --seed 42 \
  --batch_size 32 --epochs 1 --limit_samples 128 --val_limit 32 \
  --ctc_weight 0.30 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 6 \
  --out_dir "$DE/runs/comm_dwpose_gated_temporal_ctc_p6_smoke"
echo "P6_GATED_FUSION_SMOKE_DONE test_set_used=false metrics_selection_valid=false"
