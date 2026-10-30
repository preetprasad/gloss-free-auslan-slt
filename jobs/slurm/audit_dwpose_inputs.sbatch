#!/bin/bash -l
# CPU-only DWPose quality audit; produces evidence for confidence/subset ablations.
#SBATCH --job-name=dwpose-input-audit
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
python /group/pmc099/pprasad/code/phase2_baseline/tools/audit_dwpose_inputs.py \
  --clip_dir /scratch/pmc099/pprasad/dwpose/dwpose_clips \
  --output_json /group/pmc099/pprasad/comm_dwpose_full/input_audit.json
