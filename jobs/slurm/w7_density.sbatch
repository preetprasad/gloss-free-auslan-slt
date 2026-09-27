#!/bin/bash -l
# WAVE 7 -- sign density: stop rendering the signs Auslan drops. Task = (arm, seed):  0-2 A | 3-5 D  (42/101/202)
#   A = adopted corpus rebuilt in batch          D = the SAME sentences with --drop_function_signs
# Real Auslan carries 0.467 signs per English word; the stitched corpus carries 0.78 (09-05 audit, intervention (iv),
# never run until now). The English target, the vocabulary and the real half are identical between arms.
# The corpora are built and gated by kaya/w7_prep.sbatch (CPU); this array runs afterok on it. Job 20075 built them in
# task 0 and FAILED: $B was never defined, and the frozen wave-5 builder lacks --drop_function_signs (fixed in prep).
# Pre-registration: results/PREREG_WAVE7_DENSITY_2026-09-23.md
#SBATCH --job-name=w7-density
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-5
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202); ARMS=(A D)
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
W7=$G/wave7
RUNS=$W7/runs; mkdir -p "$W7" "$RUNS"
echo "task $T -> arm $arm seed $s"

# ---- the corpora are built and gated by the CPU prep job (w7_prep.sbatch); this array runs afterok on it ----
[ -n "${PREP:-}" ] || { echo "FAIL: submit with --export=ALL,PREP=<prep job id> --dependency=afterok:<id>"; exit 1; }
[ -f "$W7/BUILD_OK_$PREP" ] || { echo "FAIL: gate $W7/BUILD_OK_$PREP missing"; exit 1; }

JOINT=$W7/joint_${arm}.tsv
[ "$(grep -c . "$JOINT")" = 26199 ] || { echo "FAIL: $JOINT is not 26,199 rows"; exit 1; }
ALLOW_PREFIX="/scratch/pmc099/pprasad/wave7/corpora/" assert_group "$JOINT" "$M/val.tsv" "$M/test.tsv"
w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF \
  || { echo "[FAIL] train ${arm}"; exit 1; }
DIR=$RUNS/${arm}_s${s}
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$DIR/DONE"
echo "W7_DONE ${arm} seed ${s}"
