#!/bin/bash -l
# WAVE 10 -- does the ORDER of the stitched signs matter? Task = (arm, seed): 0-2 A | 3-5 S | 6-8 SC | 9-11 R
# (seeds 42/101/202).
#   A  = adopted recipe (real + synthetic signed in English word order)
#   S  = the SAME synthetic clips, signs shuffled; CTC targets unchanged (English order), as any reordered corpus would get
#   SC = S with each synthetic sentence's CTC target re-ordered to follow its signs (--gloss_map, built in prep), so the
#        translator's use of order can be separated from the CTC alignment
#   R  = real-only (the size of the stitched-data gain in this batch)
# English targets, exemplars, durations, sign counts and the real half are identical between A, S and SC (gated in prep).
# Pre-registration: results/PREREG_WAVE10_SIGNORDER_2026-10-01.md. Submit with
#   --dependency=afterok:<prep> --export=ALL,PREP=<prep job id>.
# ondemand-gpu at 12 h (its cap) by Preet's choice on 2026-10-01, as for wave 9: the gpu partition estimated a 14 Oct
# start; wave-9 runs of this recipe took ~1.5 h on ondemand-gpu. Any V100 (the recipe ran on 16 GB cards in waves 6-7).
#SBATCH --job-name=w10-order
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202); ARMS=(A S SC R)
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
W10=$G/wave10; RUNS=$W10/runs; mkdir -p "$RUNS"
echo "task $T -> arm $arm seed $s"
[ -n "${PREP:-}" ] && [ -f "$W10/PREP_OK_$PREP" ] || { echo "FAIL: prep gate $W10/PREP_OK_${PREP:-unset} missing"; exit 1; }
echo "9c042dcc10e8763b14af9b06cb423727  $W5/manifests/vocab.json" | md5sum -c -
EXTRA=()
case $arm in
  R)  TRAIN=$M/train.tsv; N=12439 ;;
  SC) TRAIN=$W10/joint_S.tsv; N=26199
      # train_ctc.py silently ignores a --gloss_map path that does not exist, so check it here
      [ "$(grep -c . "$W10/ctc_map_S.tsv")" -ge 13700 ] || { echo "FAIL: $W10/ctc_map_S.tsv missing or short"; exit 1; }
      EXTRA=(--gloss_map "$W10/ctc_map_S.tsv") ;;
  *)  TRAIN=$W10/joint_$arm.tsv; N=26199 ;;
esac
[ "$(grep -c . "$TRAIN")" = "$N" ] || { echo "FAIL: $TRAIN is not $N rows"; exit 1; }
assert_group "$TRAIN" "$M/val.tsv" "$M/test.tsv"
w4run "$arm" "$RUNS" "$TRAIN" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF --vocab_file "$W5/manifests/vocab.json" \
  "${EXTRA[@]}" || { echo "[FAIL] train ${arm}"; exit 1; }
DIR=$RUNS/${arm}_s${s}
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
if [ "$arm" = SC ]; then   # the trainer prints this only when it read the map
  grep -q "^\[pgg\] loaded 1[0-9][0-9][0-9][0-9] " "$G/code/logs/w10-order-${SLURM_ARRAY_JOB_ID}_${T}.out" \
    || { echo "FAIL: the CTC map was not loaded"; exit 1; }
fi
touch "$DIR/DONE"
echo "W10_DONE ${arm} seed ${s}"
