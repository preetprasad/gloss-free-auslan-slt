#!/bin/bash -l
# =============================================================================
# Kaya: pseudo-gloss CTC anti-collapse experiment on the shipped-AlphaPose pose.
# Adds a CTC head on the encoder (content-word pseudo-glosses) to force temporal
# grounding vs the posterior collapse. Compare test BLEU + distinct-output count
# to the control (1.76 greedy / 2.87 beam, 53 unique). Independent of DWPose; can
# run at the same time as train_dwpose.sbatch (separate GPU).
#
# Needs the shipped HALPE pose clips on scratch (rsync ~/Honours/full_run/clips or
# re-extract) and the comm_run manifests (already copied in runbook step 2).
#
# Submit:   cd phase2_baseline && sbatch kaya/train_ctc.sbatch
# =============================================================================
#SBATCH --job-name=ctc-pseudogloss
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs

module purge
module load cuda            # <-- CONFIRM: module avail cuda
module load Anaconda3       # <-- CONFIRM: module avail
source /group/pmc099/pprasad/envs/auslan/bin/activate

# HALPE pose clips + manifests on scratch (see runbook); manifests point at them.
export MANI=/group/pmc099/pprasad/code/comm_run/manifests
export OUT=/scratch/pmc099/pprasad/dwpose/runs/comm_ctc_pseudogloss
cd "$SLURM_SUBMIT_DIR"      # = phase2_baseline

python train_ctc.py \
  --train_manifest "$MANI/train.tsv" --val_manifest "$MANI/val.tsv" \
  --test_manifest  "$MANI/test.tsv" \
  --keypoint_subset body_hands --batch_size 64 --epochs 40 --ctc_weight 0.3 \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True \
  --device cuda --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out_dir "$OUT"
# NOTE: CTC loss runs on CPU inside train_ctc.py (fine on CUDA too); grads flow back.
