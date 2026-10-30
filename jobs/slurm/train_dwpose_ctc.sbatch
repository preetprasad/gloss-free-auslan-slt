#!/bin/bash -l
# Joint translation + rule-based pseudo-gloss CTC, controlled on DWPose inputs.
# CTC weight is supplied as CTC_WEIGHT (default 0.3); validation selects the
# checkpoint and test is evaluated once by the trainer.
#SBATCH --job-name=dwpose-ctc
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
CTC_WEIGHT="${CTC_WEIGHT:-0.3}"
SAFE_WEIGHT="${CTC_WEIGHT/./p}"
OUT="$ROOT/comm_dwpose_ctc_w${SAFE_WEIGHT}"
mkdir -p "$ROOT"
cd "$CODE"

python train_ctc.py \
  --train_manifest "$MANIFESTS/train.tsv" --val_manifest "$MANIFESTS/val.tsv" \
  --test_manifest "$MANIFESTS/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --epochs 40 --ctc_weight "$CTC_WEIGHT" \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" --out_dir "$OUT"
python tools/eval_beam.py --ckpt "$OUT/best.pt" --manifest "$MANIFESTS/test.tsv" \
  --beam 5 --length_penalty 0.6 --device cuda --output_json "$OUT/beam5_test.json"
cp -r "$OUT" /group/pmc099/pprasad/
echo "DWPOSE_CTC_DONE weight=$CTC_WEIGHT"
