#!/bin/bash -l
# P3: does P2's lightweight temporal adapter stack with the established single
# legacy pseudo-CTC head? Validation-only; no official test manifest.
#SBATCH --job-name=auslan-temporal-ctc-p3
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

python train_temporal_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" --seed 42 \
  --batch_size 32 --epochs 40 --ctc_weight 0.30 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgb_temporal_ctc_p3"

cp -r "$DE/runs/comm_dwpose_rgb_temporal_ctc_p3" "$G/"
echo "P3_DONE validation_only=true"
