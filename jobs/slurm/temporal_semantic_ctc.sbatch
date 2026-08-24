#!/bin/bash -l
# P4 semantic auxiliary: paired with P3's exact data/configuration, validation only.
#SBATCH --job-name=auslan-temporal-semantic-p4
# The model consumes cached DWPose/DINO features rather than an image backbone;
# it fits a standard V100.  Use the general GPU pool rather than waiting for a
# scarce 32-GB on-demand card, while preserving the exact P3-paired batch size.
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
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
python train_temporal_semantic_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" --seed 42 \
  --batch_size 32 --epochs 40 --ctc_weight 0.30 --semantic_weight 0.10 \
  --semantic_model sentence-transformers/all-MiniLM-L12-v2 \
  --semantic_cache "$DE/semantic_targets/all_minilm_l12_train.pt" \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgb_temporal_semantic_ctc_p4"
cp -r "$DE/runs/comm_dwpose_rgb_temporal_semantic_ctc_p4" "$G/"
echo "P4_DONE validation_only=true"
