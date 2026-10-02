#!/bin/bash -l
# WAVE 7 STAGE 2 -- sign density, fresh seeds. Task = (arm, seed): 0-1 A | 2-3 D | 4-5 DL  (seeds 303/404)
#   A  = adopted corpus (Stage 1's, reused)      D = the same sentences with --drop_function_signs (Stage 1's, reused)
#   DL = D with each clip's sign lengths rescaled so its total frame count equals its A counterpart (built in w7s2_prep)
# Same recipe, snapshot and flags as Stage 1 (kaya/w7_density.sbatch), including NO --vocab_file, so seeds 303/404 pool
# with 42/101/202 for the 5-seed adoption test. Pre-registration: results/PREREG_WAVE7_DENSITY_2026-09-23.md (section 10).
# Submit with --dependency=afterok:<prep> --export=ALL,PREP=<prep job id>.
# ondemand-gpu at 12 h (its cap), as for waves 9 and 10: the gpu partition had 45 jobs pending on 3 Oct, and this recipe
# took 1.25-1.4 h per run in Stage 1. Any V100.
#SBATCH --job-name=w7s2-train
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(303 404); ARMS=(A D DL)
arm=${ARMS[$((T / 2))]}; s=${SEEDS[$((T % 2))]}
W7=$G/wave7; S2=$G/wave7s2; RUNS=$W7/runs
echo "task $T -> arm $arm seed $s"
[ -n "${PREP:-}" ] && [ -f "$S2/PREP_OK_$PREP" ] || { echo "FAIL: prep gate $S2/PREP_OK_${PREP:-unset} missing"; exit 1; }
echo "d87466e755f80981ee73b0f334994161  $W7/joint_A.tsv" | md5sum -c -
echo "7e2fbc72e70e58d22f89589e58203454  $W7/joint_D.tsv" | md5sum -c -
case $arm in
  DL) JOINT=$S2/joint_DL.tsv ;;
  *)  JOINT=$W7/joint_$arm.tsv ;;
esac
DIR=$RUNS/${arm}_s${s}
[ ! -e "$DIR" ] || { echo "FAIL: $DIR already exists; refusing to overwrite a run"; exit 1; }
[ "$(grep -c . "$JOINT")" = 26199 ] || { echo "FAIL: $JOINT is not 26,199 rows"; exit 1; }
ALLOW_PREFIX="/scratch/pmc099/pprasad/wave7/corpora/ /scratch/pmc099/pprasad/wave7s2/corpora/" \
  assert_group "$JOINT" "$M/val.tsv" "$M/test.tsv"
w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF \
  || { echo "[FAIL] train ${arm}"; exit 1; }
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$DIR/DONE"
echo "W7S2_DONE ${arm} seed ${s}"
