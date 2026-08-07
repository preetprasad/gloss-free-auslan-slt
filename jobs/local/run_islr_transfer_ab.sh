#!/usr/bin/env bash
# Paired ISLR->CSLT transfer test on Auslan-Daily Communication (official split).
#   Arm A (control)  : translator trained from scratch.
#   Arm B (transfer) : identical translator + MM-WLAuslan ISLR-pretrained encoder.
# Only --init_encoder and --out_dir differ; same seed, split, and hyperparameters,
# so the test BLEU-4 delta isolates the value of ISLR pretraining.
#
# Encoder match: --keypoint_subset body_hands => 204-dim input, matching the
# d256 generic encoder saved by run_islr_pretrain.sh (55 encoder tensors transfer).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/.pyenv/versions/3.14.3/bin/python}"
ENC="mmwl_islr_pretrain/islr_encoder_best.pt"

COMMON=(
  --train_manifest ../comm_run/manifests/train.tsv
  --val_manifest   ../comm_run/manifests/val.tsv
  --test_manifest  ../comm_run/manifests/test.tsv
  --keypoint_subset body_hands
  --batch_size 32
  --epochs 40
  --device mps
  --num_workers 0
)

echo "########## ARM A: CONTROL (from scratch) ##########"
"$PY" train.py "${COMMON[@]}" --out_dir comm_scratch_control

echo "########## ARM B: TRANSFER (ISLR-pretrained encoder) ##########"
"$PY" train.py "${COMMON[@]}" --init_encoder "$ENC" --out_dir comm_islr_transfer

echo "########## DONE ##########"
echo "control : comm_scratch_control/metrics.json"
echo "transfer: comm_islr_transfer/metrics.json"
