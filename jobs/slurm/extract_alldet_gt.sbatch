#!/bin/bash -l
# Tracker v2 ground truth: the 60 clips of results/tracker_v2/gt_clips.json, extracted FIRST with the all-detections
# worker into the same output directory as job 31508 (which skips them when it reaches them), so the ground-truth
# panels and the training data come from one extraction.
#SBATCH --job-name=ext-alldet-gt
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-06:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
B=/group/pmc099/pprasad/scratch_backup/dwpose
LIST=$B/tracker_v2_gt_ids.txt
[ "$(grep -c . "$LIST")" = 60 ] || { echo "FAIL: GT list is not 60 clips"; exit 1; }
export RGB_DIR=$B/rgb_signer OUT_DIR=$B/dwpose_alldet CLIP_SUFFIX=_signer.mp4 CLIP_LIST=$LIST ORT_THREADS=$SLURM_CPUS_PER_TASK
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker_alldet.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 6 --device cpu --mode balanced
