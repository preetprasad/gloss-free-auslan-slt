#!/bin/bash -l
# WAVE 12f, step 2 of 3 (GPU): poses for the two measurements. No model is trained.
#   task 0  DWPose of the 3,209 isolated-sign clips with the PRODUCTION worker and settings
#           (results/dwpose_experiment/dwpose_extract_worker.py, balanced mode, first detection per frame).
#           71,126 frames; about 1 GPU-hour at the 21.7 frames per second measured for this worker (job 1081793).
#   task 1  measurement 2: rtmlib Wholebody3d (RTMW3D-x) on the 929 Communication TRAIN clips that carry a spotting span.
#           Its weights (369 MB, Apache-2.0, huggingface.co/Soykaf/RTMW3D-x) are third-party and are on no machine of
#           this project. They are fetched ONLY with W12F_ALLOW_DOWNLOAD=1; otherwise the task says so and ends
#           without a DONE marker, and step 3 runs measurement 1 alone.
#   sbatch --dependency=afterok:<stage job> kaya/w12f_extract.sbatch
#   with the download approved:  sbatch --dependency=afterok:<stage job> --export=ALL,W12F_ALLOW_DOWNLOAD=1 kaya/w12f_extract.sbatch
# The two tasks are independent: step 3 is submitted with afterany and needs only task 0's marker for measurement 1.
# Pre-registration: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md
#SBATCH --job-name=w12f-extract
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # modules, environment, cd into the snapshot
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12flib.sh      # paths, code pins, gates (the snapshot holds no job scripts)
set -u
T=$SLURM_ARRAY_TASK_ID
w12f_code_gate
[ -f "$OUT/DONE_stage" ] || fail "step 1 (kaya/w12f_stage.sbatch) has not finished"
w12f_gpu_gate
export ORT_THREADS=3
t0=$(date +%s)

if [ "$T" = 0 ]; then
  rm -f "$OUT/DONE_dwpose"
  [ "$(find "$ISO" -maxdepth 1 -name '*.mp4' | wc -l)" = "$N_ISO" ] || fail "isolated clip count"
  mkdir -p "$DW_ISO"
  find "$DW_ISO" -maxdepth 1 -name '*.writing.npz' -delete      # half-written file of a killed earlier run: it would break the count below
  RGB_DIR="$ISO" OUT_DIR="$DW_ISO" CLIP_SUFFIX=.mp4 DWPOSE_DEVICE=cuda \
      python -u "$WK" --shard 0 --nshards 1 --mode balanced --device cuda
  n=$(find "$DW_ISO" -maxdepth 1 -name '*.npz' | wc -l)
  [ "$n" = "$N_ISO" ] || fail "DWPose files for the isolated clips: $n of $N_ISO"
  echo "isolated clips: $n pose files in $(( $(date +%s) - t0 )) s"
  touch "$OUT/DONE_dwpose"
  echo "W12F_DWPOSE_DONE"; exit 0
fi

[ "$T" = 1 ] || fail "unknown array task $T"
rm -f "$OUT/DONE_pose3d"
[ "$(grep -c . "$OUT/pose3d_ids.txt")" = "$N_SPAN_CLIPS" ] || fail "list of span clips"
if [ ! -f "$POSE3D_ONNX" ]; then
  if [ "${W12F_ALLOW_DOWNLOAD:-0}" != 1 ]; then
    echo "W12F_POSE3D_SKIPPED: $POSE3D_ONNX is absent and the download is not approved (W12F_ALLOW_DOWNLOAD=1)."
    echo "Measurement 2 (estimated depth) is not run; measurement 1 does not need it."
    exit 0
  fi
  mkdir -p "$(dirname "$POSE3D_ONNX")"
  echo "fetching $POSE3D_URL"
  curl -L --fail --retry 3 -o "$POSE3D_ONNX.part" "$POSE3D_URL" || fail "download of the 3-D pose weights"
  mv "$POSE3D_ONNX.part" "$POSE3D_ONNX"
fi
echo "$POSE3D_SHA256  $POSE3D_ONNX" | sha256sum -c - || { rm -f "$POSE3D_ONNX"; fail "sha256 of the 3-D pose weights (the file was removed)"; }
mkdir -p "$P3D"
find "$P3D" -maxdepth 1 -name '*.writing.npz' -delete
python -u tools/w12f_extract3d.py --rgb_dir "$B/rgb_signer/Signer" --list "$OUT/pose3d_ids.txt" --out_dir "$P3D" \
    --pose "$POSE3D_ONNX" --device cuda
n=$(find "$P3D" -maxdepth 1 -name '*.npz' | wc -l)
[ "$n" = "$N_SPAN_CLIPS" ] || fail "3-D pose files: $n of $N_SPAN_CLIPS"
echo "span clips: $n 3-D pose files in $(( $(date +%s) - t0 )) s"
touch "$OUT/DONE_pose3d"
echo "W12F_POSE3D_DONE"
