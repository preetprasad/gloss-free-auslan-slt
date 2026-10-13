#!/bin/bash -l
# WAVE 12b -- removing wrong-sense signs from the synthetic clips. Task = (arm, seed): 0-2 A | 3-5 W   (seeds 42/101/202)
#   A = the adopted recipe on the literal adopted manifest (12,439 real + 13,760 synthetic, 1.1:1): in-job control
#   W = the SAME 13,760 sentences with the 14 wrong-sense signs not rendered (3,254 of 94,428 signs; English unchanged)
# Both arms: pose augmentation, 40 epochs, deterministic kernels, the one fixed vocabulary of waves 5-10, DWPose 133 /
# coord_norm frame, batch 64, lr 5e-4, warmup 800, cosine, CTC weight 0.3; code = the frozen wave-11 snapshot (the trainer
# needs no change). The corpora are built and gated by kaya/w12b_prep.sbatch (CPU). Submit with
#   --dependency=afterok:<prep> --export=ALL,PREP=<prep job id>
# This is a correctness arm: W is adopted unless clearly harmful (3-seed mean of W - A <= -0.5 on beam-5 validation AND
# on beam-5 test). Pre-registration: results/PREREG_WAVE12B_WRONGSENSE_2026-10-13.md
# About 1.5 h per task on a V100 (wave 10, job 49135: 1:25 to 1:32 for the same recipe and corpus size).
#SBATCH --job-name=w12b-wrongsense
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh   # env, G, M, POSE, REC, DET, AUG_DEF; cd $G/wave11/code
set -u                                                             # after the library: module/activate are not -u clean
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202); ARMS=(A W)
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
W12B=$G/wave12b; RUNS=$W12B/runs; ADOPTED=$G/stitch_scale2/joint_13760.tsv; VOCAB=$G/wave5/manifests/vocab.json
echo "task $T -> arm $arm seed $s"
[ "$DET" = "--deterministic True" ] || { echo "FAIL: deterministic mode must be on in every arm (DET='$DET')"; exit 1; }
[ "$POSE" = "--num_keypoints 133 --keypoint_subset all --coord_norm frame" ] || { echo "FAIL: POSE='$POSE'"; exit 1; }
LOG=$G/code/logs/w12b-wrongsense-${SLURM_ARRAY_JOB_ID}_${T}.out   # read after training; must be this task's own log
[ -f "$LOG" ] || { echo "FAIL: $LOG is not this task's log (job name or --output was changed)"; exit 1; }
[ -n "${PREP:-}" ] && [ -f "$W12B/PREP_OK_$PREP" ] || { echo "FAIL: prep gate $W12B/PREP_OK_${PREP:-unset} missing"; exit 1; }

# the code this task runs (frozen wave-11 snapshot = the wave-5 trainer), the vocabulary and the manifests
md5sum -c - <<EOF || { echo "FAIL: code, vocabulary or manifest is not the pinned version"; exit 1; }
75783cd2aaddb8d5822f0c5d1a4f10e5  train_ctc.py
2e54dd6d2a5e3f99572e74bb086c749e  config.py
1187c9ff21c8de97b6ba98dfedf3897f  slt/data.py
076d9b1ac5e0008a895b3a5bcfe4f470  slt/model.py
8399ac01b7f856e774b3f31d87dcf385  slt/engine.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
6553f52a3f0fbeb57feef6510289d66c  tools/eval_beam.py
2fd20e3989b8e20a9857f6d90106da42  weight_average.py
9c042dcc10e8763b14af9b06cb423727  $VOCAB
f81defc699396b920ac69e16a125af92  $ADOPTED
e67f2eb08b62fe62124a35afc9d4eeab  $M/train.tsv
b612245b71a57e4b9abdc2abc47172f8  $M/val.tsv
f73404a27780df006107a1d9e55db263  $M/test.tsv
EOF
if [ "$arm" = A ]; then TRAIN=$ADOPTED; else
  TRAIN=$W12B/joint_W.tsv
  (cd "$W12B" && md5sum -c joint_W.md5) || { echo "FAIL: joint_W.tsv changed since the prep gate"; exit 1; }
fi
[ "$(grep -c . "$TRAIN")" = 26199 ] || { echo "FAIL: $TRAIN is not 26,199 rows"; exit 1; }
eng () { cut -f2- "$1" | md5sum | cut -d' ' -f1; }
[ "$(eng "$TRAIN")" = "$(eng "$ADOPTED")" ] || { echo "FAIL: $TRAIN differs from the adopted manifest in English text"; exit 1; }
[ "$(head -n 12439 "$TRAIN" | md5sum | cut -d' ' -f1)" = e67f2eb08b62fe62124a35afc9d4eeab ] \
  || { echo "FAIL: the real half of $TRAIN is not the official training manifest"; exit 1; }
# no validation or test clip may be a training clip, and the two evaluation sets must be disjoint
leak=$( { cut -f1 "$TRAIN"; cut -f1 "$M/val.tsv"; cut -f1 "$M/test.tsv"; } | sort | uniq -d | wc -l)
[ "$leak" -eq 0 ] || { echo "FAIL: $leak clips are shared between training, validation and test"; exit 1; }
assert_group "$TRAIN" "$M/val.tsv" "$M/test.tsv"

DIR=$RUNS/${arm}_s${s}
[ ! -e "$DIR/DONE" ] || { echo "FAIL: $DIR is already finished; refusing to overwrite it"; exit 1; }
mkdir -p "$RUNS"
w4run "$arm" "$RUNS" "$TRAIN" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF --vocab_file "$VOCAB" \
  || { echo "[FAIL] train ${arm}"; exit 1; }
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
grep -q "^\[determinism\] ON" "$LOG" || { echo "FAIL: the trainer did not report deterministic mode"; exit 1; }
grep -q "^\[vocab\] fixed vocabulary" "$LOG" || { echo "FAIL: the trainer did not report the fixed vocabulary"; exit 1; }
touch "$DIR/DONE"
echo "W12B_DONE ${arm} seed ${s}"
