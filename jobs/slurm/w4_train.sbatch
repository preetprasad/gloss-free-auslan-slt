#!/bin/bash -l
# WAVE 4 training array, afterok on w4-prep. Adopted recipe (1.1:1 duration-stitch + pose-aug), 40 ep,
# --deterministic True in EVERY arm. Arms: A = adopted corpus (control for w4-trim AND arm P of w5-div),
# T = untrimmed signs (w4-trim), Q = half the sentences x 2 realisations (w5-div).
#   tasks 0-2 seeds 42/101/202: A T Q     tasks 3-4 seeds 303/404: A Q
# Pre-registration: results/PREREG_WAVE4_2026-09-22.md (Stage-1 futility on 3-seed / 5-seed mean beam-5 VAL).
#SBATCH --job-name=w4-train
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-4
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w4lib.sh
S=(42 101 202 303 404); s=${S[$SLURM_ARRAY_TASK_ID]}
if [ "$SLURM_ARRAY_TASK_ID" -ge 3 ]; then ARMS=(A Q); else ARMS=(A T Q); fi
MF=$W4/manifests; RUNS=$W4/runs_tq; mkdir -p "$RUNS"
pin_pose; assert_group "$MF"/joint_A.tsv "$MF"/joint_T.tsv "$MF"/joint_Q.tsv "$M/val.tsv" "$M/test.tsv"
for arm in "${ARMS[@]}"; do
  w4run "w4$arm" "$RUNS" "$MF/joint_$arm.tsv" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF || true
done
w4summary "$RUNS" "$s" $(printf "w4%s " "${ARMS[@]}")
echo "W4_TRAIN_DONE"
