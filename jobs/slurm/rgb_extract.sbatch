#!/bin/bash -l
# RGB fusion Stage A: extract DINOv2 crop features for all 14k clips on the work CPU
# nodes (idle) in parallel with the GPU VLP/LLM jobs. Resumable.
#SBATCH --job-name=rgb-extract
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
export RGB_DIR=/scratch/pmc099/pprasad/dwpose/rgb_clips
export DWPOSE_DIR=/scratch/pmc099/pprasad/dwpose/dwpose_clips
export OUT_DIR=/scratch/pmc099/pprasad/dwpose/rgb_feats
python /group/pmc099/pprasad/code/results/dwpose_experiment/extract_rgb_feats.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 32 --frames 24
echo "shard $SLURM_ARRAY_TASK_ID done"
