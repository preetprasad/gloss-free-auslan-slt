#!/bin/bash -l
# Capacity re-test only after DWPose made the visual stream informative.
# Keeps the DWPose manifests and target protocol fixed; changes model scale.
#SBATCH --job-name=dwpose-scale
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate

CODE=/group/pmc099/pprasad/code/phase2_baseline
MANIFESTS=/scratch/pmc099/pprasad/dwpose/manifests
ROOT=/scratch/pmc099/pprasad/dwpose/runs
OUT="$ROOT/comm_dwpose_scale_d512_6x6"
mkdir -p "$ROOT"
cd "$CODE"

python train.py \
  --train_manifest "$MANIFESTS/train.tsv" --val_manifest "$MANIFESTS/val.tsv" \
  --test_manifest "$MANIFESTS/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --d_model 512 --nhead 8 --num_encoder_layers 6 --num_decoder_layers 6 \
  --dim_feedforward 2048 --batch_size 32 --grad_accum_steps 2 \
  --epochs 60 --lr 3e-4 --lr_schedule cosine --warmup_steps 1000 \
  --bleu_tokenize 13a --detokenize True --device cuda \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" --out_dir "$OUT"
python tools/eval_beam.py --ckpt "$OUT/best.pt" --manifest "$MANIFESTS/test.tsv" \
  --beam 5 --length_penalty 0.6 --device cuda --output_json "$OUT/beam5_test.json"
cp -r "$OUT" /group/pmc099/pprasad/
echo "DWPOSE_SCALE_DONE"
