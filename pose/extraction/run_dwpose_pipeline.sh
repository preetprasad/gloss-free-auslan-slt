#!/usr/bin/env bash
# Unattended DWPose full re-extraction + retrain pipeline (Mac, ~10-14h).
# Resumable: re-running skips already-extracted mp4s and npz. Wrap with caffeinate
# so the Mac does not idle-sleep mid-run.
set -uo pipefail
H="$HOME/Honours"
PY="$HOME/.pyenv/versions/3.14.3/bin/python"
RC="$H/tools/rclone/rclone"; CFG="$H/data/rclone.conf"
DE="$H/results/dwpose_experiment"
LOG="$DE/pipeline.log"
ZIP="auslan_drive:Honours Datasets/Auslan-Daily/Communication/Signer Only Video Clip/Signer.zip"
say(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "=== PIPELINE START ==="

# 1) extract ALL clip videos from the zip (stream, no full-disk copy)
NCLIP=$(find "$DE/rgb_clips" -iname '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
say "have $NCLIP mp4s locally"
if [ "$NCLIP" -lt 12000 ]; then
  say "streaming full 17.4GB zip and extracting all signer clips..."
  "$RC" --config "$CFG" cat "$ZIP" 2>>"$LOG" | bsdtar -x -f - -C "$DE/rgb_clips" 2>>"$LOG" || say "bsdtar stream ended"
  NCLIP=$(find "$DE/rgb_clips" -iname '*.mp4' | wc -l | tr -d ' ')
  say "extraction done: $NCLIP mp4s"
fi

# 2) parallel DWPose extraction (RTMW-DW-x, best hands)
N=10
say "launching $N DWPose workers over $NCLIP clips..."
pids=()
for i in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=1 nohup "$PY" "$DE/dwpose_extract_worker.py" --shard $i --nshards $N --mode balanced \
    > "$DE/worker_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
NPZ=$(ls "$DE/dwpose_clips"/*.npz 2>/dev/null | wc -l | tr -d ' ')
say "DWPose extraction complete: $NPZ npz"

# 3) manifests repointed to DWPose pose
"$PY" "$DE/build_dwpose_manifests.py" 2>&1 | tee -a "$LOG"

# 4) retrain the identical baseline on DWPose pose (133 kpts, frame-norm)
say "retraining translator on DWPose pose..."
cd "$H/src"
"$PY" train.py \
  --train_manifest "$DE/manifests/train.tsv" \
  --val_manifest   "$DE/manifests/val.tsv" \
  --test_manifest  "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 32 --epochs 40 --device mps --num_workers 0 \
  --out_dir comm_dwpose_full 2>&1 | tee -a "$LOG"
say "retrain done"

# 5) beam decode for the literature-standard number
"$PY" tools/eval_beam.py --ckpt comm_dwpose_full/best.pt \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device mps 2>&1 | tee -a "$LOG"
say "=== PIPELINE COMPLETE ==="
