#!/bin/bash -l
# P2: paired translation-only temporal-motion ablation (validation only).
# Arm A is the existing RGB architecture with CTC removed. Arm B adds only the
# lightweight multi-scale temporal adapter. Same data, seed and VLP warm start.
# No pseudo labels and no official test manifest are supplied.
# Submit: cd /group/pmc099/pprasad/code/phase2_baseline && sbatch kaya/temporal_pyramid_ablation.sbatch
#SBATCH --job-name=auslan-temporal-p2
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
COMMON=(
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv"
  --num_keypoints 133 --keypoint_subset all --coord_norm frame
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2
  --init_encoder "$INIT" --seed 42 --batch_size 32 --epochs 40
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8
)

echo "=== P2A control: RGB translation-only (no CTC, no pseudo labels) ==="
python train_temporal_pyramid.py "${COMMON[@]}" --temporal_adapter none \
  --out_dir "$DE/runs/comm_dwpose_rgb_translation_control_p2"

echo "=== P2B temporal pyramid: RGB translation-only (no CTC, no pseudo labels) ==="
python train_temporal_pyramid.py "${COMMON[@]}" --temporal_adapter pyramid \
  --out_dir "$DE/runs/comm_dwpose_rgb_temporal_pyramid_p2"

cp -r "$DE/runs/comm_dwpose_rgb_translation_control_p2" "$G/"
cp -r "$DE/runs/comm_dwpose_rgb_temporal_pyramid_p2" "$G/"
echo "P2_DONE validation_only=true"
