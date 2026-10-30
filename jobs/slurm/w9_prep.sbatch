#!/bin/bash -l
# WAVE 9 prep (CPU, afterok on the all-detections extraction): build pose arms C0 / V1 / V2 and their manifests from ONE
# extraction. Pre-registration: results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md (§5 and the wave-9 addendum).
#SBATCH --job-name=w9-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; B=$G/scratch_backup/dwpose; W9=$G/wave9; TL=$W9/tools
echo "f81defc699396b920ac69e16a125af92  $G/wave6/joint_A_s42.tsv" | md5sum -c -
echo "3b254fdd9c4f772b376adc57279291ab  $TL/tracker_v2.py" | md5sum -c -
echo "e53a4080d771824833a5cf8c9cb3b6e1  $TL/materialise_arms.py" | md5sum -c -
python "$TL/materialise_arms.py" --selftest
python "$TL/materialise_arms.py" "$B/dwpose_alldet" "$B/manifests" "$G/wave6/joint_A_s42.tsv" "$B/pose_w9" "$W9/manifests" \
    --workers "$SLURM_CPUS_PER_TASK"
touch "$W9/PREP_OK_$SLURM_JOB_ID"; echo "W9_PREP_DONE"
