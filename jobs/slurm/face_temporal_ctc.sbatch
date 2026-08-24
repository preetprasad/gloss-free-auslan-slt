#!/bin/bash -l
# P5c: controlled face-channel test against P3.
# The sole representational change is appending a frozen 384-D face ROI vector
# to each existing 1,152-D RGB vector. Train/validation only; no test manifest.
#SBATCH --job-name=auslan-face-temporal-ctc
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"

python train_face_temporal_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 \
  --face_feat_dir "$DE/face_feats" --face_feature_dim 384 --rgb_encoder_layers 2 \
  --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" --seed 42 \
  --batch_size 32 --epochs 40 --ctc_weight 0.30 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_face_temporal_ctc_p5c"

cp -r "$DE/runs/comm_dwpose_face_temporal_ctc_p5c" "$G/"
echo "P5C_FACE_TEMPORAL_DONE validation_only=true test_set_used=false"
