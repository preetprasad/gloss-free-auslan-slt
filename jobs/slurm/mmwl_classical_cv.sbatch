#!/bin/bash -l
# MM-WLAuslan classical-CV sweep (full pixel-level battery, NO neural nets) — the "give it
# a go" completeness pass, on idle work CPU in parallel with the GPU jobs. Reuses the
# Auslan-Daily classical_cv_worker.py. Produces per-clip difficulty stats for paper symmetry
# and cross-checks the DWPose-confidence QC. Waits for the Valid unzip marker first.
#SBATCH --job-name=mmwl-cv
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
M=/scratch/pmc099/pprasad/mmwl
RGB=$M/rgb_clips/valid            # start with Valid (unzipped by the extraction job); Train after
for i in $(seq 1 180); do [ -f "$RGB/UNZIP_DONE" ] && break; sleep 20; done
export RGB_DIR="$RGB" OUT_DIR="$M/classical_cv/valid"
python /group/pmc099/pprasad/code/results/dwpose_experiment/classical_cv_worker.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 16 --sample 24 --scale 0
echo "shard $SLURM_ARRAY_TASK_ID CV DONE"
