#!/bin/bash -l
# RGB fusion Stage B: two-stream pose+RGB translator, warm-started from the VLP 10.27 best.
# Tests whether DINOv2 appearance features (handshape the pose can't capture) push past 10.27.
#SBATCH --job-name=rgb-fusion
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

INIT=/group/pmc099/pprasad/comm_dwpose_vlp_ctc/best.pt   # warm-start from the 10.27 VLP best
[ -f "$INIT" ] || INIT=/group/pmc099/pprasad/comm_dwpose_ctc/best.pt

echo "=== RGB fusion training (warm start: $INIT) ==="
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
  --init_encoder "$INIT" \
  --batch_size 32 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --num_beams 5 --length_penalty 0.6 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_rgb_fusion"
cp -r "$DE/runs/comm_dwpose_rgb_fusion" /group/pmc099/pprasad/
echo "RGB_FUSION_DONE"
