#!/usr/bin/env bash
# DWPose extraction (8 thread-capped workers) -> retrain -> beam -> CTC experiment.
# Strictly sequential heavy jobs: extraction is CPU (16 threads total), retrain/CTC
# are MPS and only run after extraction finishes. Resumable (skips done npz).
set -uo pipefail
H="$HOME/Honours"; PY="$HOME/.pyenv/versions/3.14.3/bin/python"
DE="$H/results/dwpose_experiment"; LOG="$DE/pipeline.log"
say(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

N=8
say "=== EXTRACTION v2: $N workers x 2 ORT threads ==="
pids=()
for i in $(seq 0 $((N-1))); do
  ORT_THREADS=2 nohup "$PY" "$DE/dwpose_extract_worker.py" --shard $i --nshards $N --mode balanced \
    > "$DE/worker_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
say "extraction complete: $(ls "$DE/dwpose_clips"/*.npz 2>/dev/null | wc -l) npz"

"$PY" "$DE/build_dwpose_manifests.py" 2>&1 | tee -a "$LOG"

say "retraining translator on DWPose pose (MPS)..."
cd "$H/src"
"$PY" train.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 32 --epochs 40 --device mps --num_workers 0 \
  --out_dir comm_dwpose_full 2>&1 | tee -a "$LOG"
"$PY" tools/eval_beam.py --ckpt comm_dwpose_full/best.pt \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device mps 2>&1 | tee -a "$LOG"
say "=== DWPose result done; starting CTC pseudo-gloss experiment (MPS) ==="

"$PY" train_ctc.py \
  --train_manifest ../comm_run/manifests/train.tsv --val_manifest ../comm_run/manifests/val.tsv \
  --test_manifest ../comm_run/manifests/test.tsv \
  --keypoint_subset body_hands --batch_size 32 --epochs 40 --ctc_weight 0.3 \
  --device mps --num_workers 0 --out_dir comm_ctc_pseudogloss 2>&1 | tee -a "$LOG"
say "=== ALL DONE (DWPose retrain + CTC experiment) ==="
