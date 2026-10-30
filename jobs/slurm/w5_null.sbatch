#!/bin/bash -l
# W5-NULL: the run-level null of the paired delta, in the mode the ladder was measured in.
# Six replicates, ALL seed 42, plain DWPose+CTC 40 ep, determinism OFF (DET="" -- six identical runs would
# measure nothing). Publish sd_hat with its chi-square CI and a 1.96*sd band; never an empirical percentile
# of n<=6; scope it to the SINGLE-RUN rows of the negatives table only. Pre-registration: results/PREREG_WAVE4_2026-09-22.md
#SBATCH --job-name=w5-null
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-5
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
DET=""; source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w4lib.sh
RUNS=$W4/runs_null; mkdir -p "$RUNS"
pin_pose; assert_group "$M"/train.tsv "$M"/val.tsv "$M"/test.tsv
w4run "null$SLURM_ARRAY_TASK_ID" "$RUNS" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" 42 40 --
echo "W5_NULL_DONE"
