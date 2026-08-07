#!/bin/bash -l
# Classical-CV layer-0 analysis of all 14,041 raw signer clips (CPU, no GPU).
# Runs on the work partition in parallel with the DWPose GPU extraction.
#SBATCH --job-name=classical-cv
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-63
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge
module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
export RGB_DIR=/scratch/pmc099/pprasad/dwpose/rgb_clips
export OUT_DIR=/scratch/pmc099/pprasad/classical_cv
mkdir -p "$OUT_DIR"
python /group/pmc099/pprasad/code/results/dwpose_experiment/classical_cv_worker.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 64 --sample 24 --scale 0   # full resolution
echo "shard $SLURM_ARRAY_TASK_ID done"
