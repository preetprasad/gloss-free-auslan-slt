#!/bin/bash -l
# WAVE 8 -- how BLEU-4 scales with the amount of REAL Communication pairs, and how much of the official score is
# episode familiarity. A MEASUREMENT, not a lever: nothing is adopted. Real-only training, adopted recipe otherwise
# (pose-aug, deterministic, DWPose 133 frame-norm, CTC 0.3, one fixed vocabulary). Every arm gets ~the same budget
# (995k-1,005k clip-passes; 15,600-16,000 optimiser steps) and exactly 40 validation evaluations.
#   tasks  0-4  R025  within-video 25%   3,141 clips  320 ep  eval every 8   seeds 42/101/202/303/404
#   tasks  5-9  R050  within-video 50%   6,241 clips  160 ep  eval every 4
#   tasks 10-14 R100  all real          12,439 clips   80 ep  eval every 2
#   tasks 15-19 V050  whole videos       6,241 clips  160 ep  eval every 4   (in-batch comparator: R050, same seed)
# The subset seed equals the training seed. Builder: $W8/tools/real_subsets.py (self-tested; groups by video_\d+).
# Pre-registration: results/PREREG_WAVE8_REALDATA_2026-09-24.md
#SBATCH --job-name=w8-realdata
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-19
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202 303 404)
ARMS=(R025 R050 R100 V050); FRACS=(0.25 0.5 1.0 -); SIZES=(3141 6241 12439 6241)
EPOCHS=(320 160 80 160); EVERY=(8 4 2 4)
i=$((T / 5)); s=${SEEDS[$((T % 5))]}; arm=${ARMS[$i]}
W8=$G/wave8; MW=$W5/manifests; RUNS=$W8/runs; MAN=$W8/manifests; mkdir -p "$RUNS" "$MAN"
V="--vocab_file $MW/vocab.json"
echo "task $T -> arm $arm seed $s | ${SIZES[$i]} clips | ${EPOCHS[$i]} ep | eval every ${EVERY[$i]}"
[ -f "$MW/vocab.json" ] || { echo "FAIL: fixed vocabulary missing"; exit 1; }
[ -f "$W8/tools/real_subsets.py" ] || { echo "FAIL: builder not deployed to $W8/tools"; exit 1; }
[ "$(md5sum "$W8/tools/real_subsets.py" | cut -d' ' -f1)" = "c338e5f9bc108fda1ea7683a0aac60f4" ] || { echo "FAIL: builder changed"; exit 1; }
[ "$(grep -c . "$M/train.tsv")" = 12439 ] || { echo "FAIL: train manifest is not 12,439 rows"; exit 1; }

# --- this task's manifest (one file per task, so array tasks never race) ---
SUB=$MAN/${arm}_s${s}.tsv
if [ "$arm" = V050 ]; then
  python "$W8/tools/real_subsets.py" --train "$M/train.tsv" --val "$M/val.tsv" --test "$M/test.tsv" \
    --sampler video --size 6241 --seed "$s" --out "$SUB" --expect 6241
else
  python "$W8/tools/real_subsets.py" --train "$M/train.tsv" --val "$M/val.tsv" --test "$M/test.tsv" \
    --sampler within --frac "${FRACS[$i]}" --seed "$s" --out "$SUB" --expect "${SIZES[$i]}"
fi
assert_group "$SUB" "$M/val.tsv" "$M/test.tsv"

w4run "$arm" "$RUNS" "$SUB" "$M/val.tsv" "$M/test.tsv" "$s" "${EPOCHS[$i]}" -- $AUG_DEF $V --eval_every "${EVERY[$i]}" \
  || { echo "[FAIL] train ${arm}"; exit 1; }
D=$RUNS/${arm}_s${s}
python - "$D/progress.json" <<'PY'
import json, sys
h = [r for r in json.load(open(sys.argv[1]))["history"] if "bleu4" in r]
assert len(h) == 40, f"expected 40 validation evaluations, got {len(h)}"
print(f"validation evaluations: {len(h)} (OK)")
PY
for f in beam5_val_best beam5_test_best; do [ -s "$D/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$D/DONE"
echo "W8_DONE ${arm} seed ${s}"
