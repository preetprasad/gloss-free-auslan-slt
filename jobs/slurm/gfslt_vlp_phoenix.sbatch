#!/bin/bash -l
# GFSLT-VLP reproduction on PHOENIX-2014T — the multi-day 4-GPU training (Phase A).
# Uses the FULL `gpu` partition (3-day limit), NOT ondemand-gpu (12h cap) — a full VLP+SLT run
# will not fit in 12h. DO NOT launch until: (1) phoenix-stage done, (2) gfslt-env built + MBart
# weights present, (3) PHOENIX gzip-pickle labels built, (4) config paths pointed at the extracted
# frames. This is a large allocation — confirm with Atif before submitting (per his interim-compute note).
#SBATCH --job-name=gfslt-phoenix
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06 cuda/12.6.3
source activate /group/pmc099/pprasad/envs/gfslt
REPO=/group/pmc099/pprasad/code/GFSLT-VLP
cd "$REPO"
# configs/config_gloss_free.yaml must point img_path -> extracted PHOENIX fullFrame dir,
# and *_label_path -> the gzip-pickle labels (see PHOENIX_REPRO_STATUS.md step 4).

echo "=== Stage 1: VLP pretrain (visual encoder + text decoder), 80 ep, 4 GPU ==="
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_vlp_v2.py \
  --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp_v2 \
  --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False

echo "=== Stage 2: SLT finetune from the VLP checkpoint, 200 ep, 4 GPU ==="
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_slt.py \
  --batch-size 2 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free \
  --finetune ./out/vlp_v2/best_checkpoint.pth

echo "=== Eval (single GPU; needs nlg-eval) — confirm published PHOENIX-2014T BLEU-4 ==="
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_slt.py \
  --batch-size 2 --output_dir out/Gloss-Free --resume out/Gloss-Free/best_checkpoint.pth --eval
cp -r out /group/pmc099/pprasad/gfslt_phoenix_out 2>/dev/null || true
echo "GFSLT_PHOENIX_DONE"
