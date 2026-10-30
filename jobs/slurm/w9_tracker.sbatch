#!/bin/bash -l
# WAVE 9 -- does a correct signer track change BLEU? Task = (arm, seed): 0-2 C0 | 3-5 V1 | 6-8 V2 (seeds 42/101/202).
#   C0 = detection 0 per frame (the current pipeline)   V1 = wave-3 tracker   V2 = tracker v2 (held-out wrong-person
#   rate 0.7% vs v1 9.9% vs index-0 23.3%). All three come from ONE extraction; English text and the synthetic half are
#   identical; each arm is trained AND evaluated on its own tracker's pose, as a deployed system would be.
# Pre-registration: results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md (§5 + wave-9 addendum). Submit with
#   --dependency=afterok:<prep> --export=ALL,PREP=<prep job id>.
#SBATCH --job-name=w9-tracker
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-8
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202); ARMS=(c0 v1 v2)
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
W9=$G/wave9; MW=$W9/manifests; RUNS=$W9/runs; mkdir -p "$RUNS"
echo "task $T -> arm $arm seed $s"
[ -n "${PREP:-}" ] && [ -f "$W9/PREP_OK_$PREP" ] || { echo "FAIL: prep gate $W9/PREP_OK_${PREP:-unset} missing"; exit 1; }
echo "9c042dcc10e8763b14af9b06cb423727  $W5/manifests/vocab.json" | md5sum -c -
for f in "joint_$arm:26199" "${arm}_val:800" "${arm}_test:800"; do
  [ "$(grep -c . "$MW/${f%%:*}.tsv")" = "${f##*:}" ] || { echo "FAIL: $MW/${f%%:*}.tsv is not ${f##*:} rows"; exit 1; }
done
assert_group "$MW/joint_$arm.tsv" "$MW/${arm}_val.tsv" "$MW/${arm}_test.tsv"
w4run "$arm" "$RUNS" "$MW/joint_$arm.tsv" "$MW/${arm}_val.tsv" "$MW/${arm}_test.tsv" "$s" 40 -- \
    $AUG_DEF --vocab_file "$W5/manifests/vocab.json" || { echo "[FAIL] train ${arm}"; exit 1; }
DIR=$RUNS/${arm}_s${s}
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$DIR/DONE"; echo "W9_DONE ${arm} seed ${s}"
