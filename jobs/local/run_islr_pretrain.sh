#!/usr/bin/env bash
# MM-WLAuslan ISLR pretraining (pose -> 3,215-gloss recognition) -> transferable encoder.
#
# Stage 1 of the ISLR->CSLT transfer experiment (08-06 factorial arm 2). Trains the
# SAME pose encoder used by the Auslan-Daily translator on word-level Auslan
# recognition, then saves mmwl_islr_pretrain/islr_encoder_best.pt, loadable by
# train.py --init_encoder for the downstream fine-tune.
#
# IMPORTANT: --model_arch, --d_model and --keypoint_subset MUST match the downstream
# Auslan-Daily run for the encoder tensors to transfer 1:1 (defaults below match the
# 9M generic body+hands baseline). Data was fetched to ../mmwl_pose via bundled rclone.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/.pyenv/versions/3.14.3/bin/python}"

"$PY" pretrain_islr.py \
  --pose_pkl     ../mmwl_pose/pose_cam1_trainval.pkl \
  --train_labels ../mmwl_pose/Train.json \
  --val_pose_pkl ../mmwl_pose/pose_cam1_trainval.pkl \
  --val_labels   ../mmwl_pose/Valid.json \
  --keypoint_subset body_hands \
  --model_arch generic \
  --d_model 256 \
  --batch_size 32 \
  --epochs 30 \
  --lr 5e-4 \
  --lr_schedule cosine \
  --warmup_steps 500 \
  --label_smoothing 0.1 \
  --grad_clip 1.0 \
  --device mps \
  --num_workers 0 \
  --out_dir mmwl_islr_pretrain

echo "done -> mmwl_islr_pretrain/islr_encoder_best.pt"
echo "downstream transfer: train.py --init_encoder mmwl_islr_pretrain/islr_encoder_best.pt \\"
echo "  --model_arch generic --d_model 256 --keypoint_subset body_hands (must match)"
