#!/bin/bash -l
# 25 Oct 2026 RE-RUN: the AlphaPose arm now reads Annotation["Signer"] (the first run read a non-signer in 88.4% of clips,
# the project log 2026-10-25g). Identical otherwise; Edw re-run in the same job as the in-batch control.
# W5-ESTREP: does the +4.45 estimator effect (DWPose 10.03 vs AlphaPose 5.58, ONE run per cell, 2026-08-22)
# replicate across seeds? Matched body+hands keypoints, frame norm, plain DWPose+CTC recipe, no aug, 40 ep.
# Seed 42 is run FRESH; the August pair is reported separately as a historical observation, never pooled.
# Claimed replicated only if all three seeds are positive. Spends 6 official-test decodes of a fixed,
# pre-registered contrast (a replication, not a selection look). Pre-registration: results/PREREG_WAVE4_2026-09-22.md
#SBATCH --job-name=w5-estsig
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-2
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w4lib.sh
S=(42 101 202); s=${S[$SLURM_ARRAY_TASK_ID]}
AD=$G/scratch_backup/alphapose_signer/manifests; RUNS=$W4/runs_estrep_signer; mkdir -p "$RUNS"
pin_pose; assert_group "$M"/train.tsv "$M"/val.tsv "$M"/test.tsv "$AD"/ap_train.tsv "$AD"/ap_val.tsv "$AD"/ap_test.tsv
POSE="--num_keypoints 133 --keypoint_subset cwb_body_hands --coord_norm frame"
w4run Edw "$RUNS" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- || true
POSE="--num_keypoints 136 --keypoint_subset body_hands --coord_norm frame"
w4run Eap "$RUNS" "$AD/ap_train.tsv" "$AD/ap_val.tsv" "$AD/ap_test.tsv" "$s" 40 -- || true
w4summary "$RUNS" "$s" Eap Edw
echo "W5_ESTSIG_DONE"
