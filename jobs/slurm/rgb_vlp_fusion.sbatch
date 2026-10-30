#!/bin/bash -l
# Full RGB-VLP stack: (1) RGB<->text contrastive pretraining on the DINOv2 features,
# (2) RGB fusion warm-started from BOTH the pose-VLP base (10.27) and the RGB-VLP appearance
# encoder. The literature's RGB-VLP route toward 15+; current best to beat = 10.335.
#SBATCH --job-name=rgb-vlp-fusion
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
POSE_INIT=/group/pmc099/pprasad/comm_dwpose_vlp_ctc/best.pt   # pose-VLP base (10.27)

echo "=== 1) RGB-VLP contrastive pretraining ==="
python pretrain_rgb_vlp.py \
  --train_manifest "$DE/manifests/train.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --batch_size 128 --epochs 40 --lr 5e-4 --lr_schedule cosine \
  --device cuda --num_workers 8 --out_dir "$DE/runs/rgb_vlp"

echo "=== 2) RGB fusion warm-started from pose-VLP base + RGB-VLP appearance encoder ==="
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$POSE_INIT" --rgb_vlp_init "$DE/runs/rgb_vlp/rgb_vlp_best.pt" \
  --batch_size 32 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --num_beams 5 --length_penalty 0.6 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgbvlp_fusion"
cp -r "$DE/runs/comm_dwpose_rgbvlp_fusion" /group/pmc099/pprasad/
echo "RGB_VLP_FUSION_DONE"
