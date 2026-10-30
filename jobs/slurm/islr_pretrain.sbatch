#!/bin/bash -l
# MM-WLAuslan ISLR pretraining on DWPose 133-kpt pose (representation parity with the
# DWPose translator). Produces islr_encoder_best.pt loadable by train_ctc.py --init_encoder.
# Prereq: mmwl-dw-train + mmwl-dw-valid finished (DWPose npz under scratch/mmwl/dwpose_clips/*).
#SBATCH --job-name=islr-pretrain
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
RC="rclone --config /group/pmc099/pprasad/code/rclone.conf"
M=/scratch/pmc099/pprasad/mmwl
LAB=$M/labels; mkdir -p "$LAB"
[ -f "$LAB/Train.json" ] || $RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Annotation/Labels+Split/Train.json" "$LAB/Train.json"
[ -f "$LAB/Valid.json" ] || $RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Annotation/Labels+Split/Valid.json" "$LAB/Valid.json"
cd /group/pmc099/pprasad/code/phase2_baseline

# SMOKE (few clips, 1 epoch) — set -e blocks the full run if the pipeline is broken.
echo "=== SMOKE ==="
python pretrain_islr.py \
  --pose_dir "$M/dwpose_clips/valid" --train_labels "$LAB/Valid.json" \
  --num_keypoints 133 --keypoint_subset all --use_confidence True \
  --limit_samples 64 --epochs 1 --batch_size 16 --device cuda --out_dir "$M/runs/islr_smoke"

echo "=== QC: DWPose-confidence quality gate -> keep-list (cleaner pretraining signal) ==="
QC=/group/pmc099/pprasad/code/results/dwpose_experiment/mmwl_qc.py
python "$QC" --pose_dir "$M/dwpose_clips/train" --out "$M/qc/train" --min_hand_conf 0.30 --max_zero_frac 0.30
python "$QC" --pose_dir "$M/dwpose_clips/valid" --out "$M/qc/valid" --min_hand_conf 0.30 --max_zero_frac 0.30
cat "$M/qc/train_keep.txt" "$M/qc/valid_keep.txt" > "$M/qc/keep_all.txt"
cp -r "$M/qc" /group/pmc099/pprasad/mmwl_qc

echo "=== FULL: pretrain on Train (QC-filtered), validate on Valid ==="
python pretrain_islr.py \
  --pose_dir "$M/dwpose_clips/train" --train_labels "$LAB/Train.json" \
  --val_pose_dir "$M/dwpose_clips/valid" --val_labels "$LAB/Valid.json" \
  --keep_list "$M/qc/keep_all.txt" \
  --num_keypoints 133 --keypoint_subset all --use_confidence True \
  --batch_size 128 --epochs 30 --lr 5e-4 --lr_schedule cosine --warmup_steps 500 \
  --device cuda --num_workers 8 --out_dir "$M/runs/islr_dwpose"
cp -r "$M/runs/islr_dwpose" /group/pmc099/pprasad/
echo "ISLR_PRETRAIN_DONE -> islr_encoder_best.pt (use as train_ctc.py --init_encoder)"
