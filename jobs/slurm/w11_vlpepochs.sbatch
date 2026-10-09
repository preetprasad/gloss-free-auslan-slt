#!/bin/bash -l
# W11-PRE: number of pose-text pretraining epochs, chosen on VALIDATION only (the 80 epochs of the reported
# pre-trained model were taken from test scores of single runs). One arm per array task:
#   P0 = no pretraining (control) | P40 / P80 / P128 / P256 = pretraining epochs, then the same 40-epoch training.
# Seeds 42/101/202, all in this one job. Pretraining is now seeded; training uses deterministic kernels.
# Pre-registration: results/PREREG_WAVE11_VALSELECT_2026-10-09.md
#SBATCH --job-name=w11-pre
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-14
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh
E=(0 40 80 128 256); S=(42 101 202)
e=${E[$((SLURM_ARRAY_TASK_ID % 5))]}; s=${S[$((SLURM_ARRAY_TASK_ID / 5))]}
RUNS=$W11/runs_pre; mkdir -p "$RUNS"
pin_pose; assert_group "$M/train.tsv" "$M/val.tsv" "$M/test.tsv"
[ "$(wc -l < "$M/train.tsv")" = 12439 ] && [ "$(wc -l < "$M/val.tsv")" = 800 ] && [ "$(wc -l < "$M/test.tsv")" = 800 ] \
  || { echo "FAIL: manifest sizes"; exit 1; }
echo "=== task $SLURM_ARRAY_TASK_ID: P${e}, seed $s ==="
if [ "$e" = 0 ]; then
  w4run P0 "$RUNS" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" "$s" 40 --
else
  V="$RUNS/vlp_e${e}_s${s}"
  python pretrain_vlp.py --train_manifest "$M/train.tsv" --val_manifest "$M/val.tsv" $POSE --seed "$s" \
    --batch_size 64 --epochs "$e" --lr 5e-4 --lr_schedule cosine --device cuda --num_workers 8 --out_dir "$V"
  [ -s "$V/vlp_encoder_best.pt" ] || { echo "FAIL: no pretrained encoder"; exit 1; }
  w4run "P${e}" "$RUNS" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- --init_encoder "$V/vlp_encoder_best.pt"
fi
[ -s "$RUNS/P${e}_s${s}/beam5_test_best.json" ] && touch "$RUNS/DONE_P${e}_s${s}"
echo "W11_PRE_DONE P${e} s${s}"
