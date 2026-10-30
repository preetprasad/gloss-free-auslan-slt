#!/bin/bash -l
# P8: validation-only stochastic filtered pseudo-content CTC.
# Never add a test manifest here: selection is strictly train + validation.
# The token target remains English-order pseudo content; this is a robustness
# ablation, not pseudo-gloss reordering or true Auslan gloss supervision.
#SBATCH --job-name=auslan-filtered-ctc-p8
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
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
INIT="$G/comm_dwpose_vlp_ctc/best.pt"

echo "=== P8 stochastic filtered English-order pseudo-CTC ==="
echo "Validation only: no official test manifest supplied or read."

# A real-file CUDA smoke gate; its full-val score is deliberately not used for selection.
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --init_encoder "$INIT" --seed 42 --batch_size 64 --epochs 1 --limit_samples 128 \
  --ctc_weight 0.30 --pseudo_token_drop 0.20 --lr 5e-4 --lr_schedule cosine --warmup_steps 8 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_filtered_ctc_p8_smoke_${SLURM_JOB_ID}"
echo "P8_GPU_SMOKE_PASSED metrics_selection_valid=false"

python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --init_encoder "$INIT" --seed 42 --batch_size 64 --epochs 40 \
  --ctc_weight 0.30 --pseudo_token_drop 0.20 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_filtered_ctc_p8"

cp -r "$DE/runs/comm_dwpose_filtered_ctc_p8" "$G/"
echo "P8_DONE validation_only=true test_set_used=false pseudo_target_order=english_spoken_order"
