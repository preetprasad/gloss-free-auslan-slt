#!/bin/bash -l
# Estimator-only ablation — isolate the DWPose gain from the +face / frame-vs-bbox-norm confounds.
# The Week-4 breakthrough changed THREE things at once (AlphaPose->DWPose, +face keypoints, bbox->frame
# norm). This decomposes them by reducing DWPose to the SAME articulator set as the AlphaPose baseline
# (body+hands, no face) under both norms. Same CTC recipe as comm_dwpose_ctc (9.63) — only subset/norm change.
#
# Reference cells for the thesis table:
#   AlphaPose  body_hands  bbox   = 2.87  (historical control)
#   DWPose     all(+face)  frame  = 9.63  (existing comm_dwpose_ctc)
#   DWPose     body_hands  bbox   = RUN 1 (this job)  <- matched-to-AlphaPose config; estimator effect
#   DWPose     body_hands  frame  = RUN 2 (this job)  <- vs RUN 1 = norm effect; vs 9.63 = face effect
#SBATCH --job-name=est-ablation
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd $G/code/phase2_baseline
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== RUN 1: DWPose body_hands (no face) + BBOX-norm  (matched to AlphaPose config) ==="
python train_ctc.py $COM $REC \
  --num_keypoints 133 --keypoint_subset cwb_body_hands --coord_norm bbox \
  --out_dir "$DE/runs/est_dwpose_bodyhands_bbox"
python tools/eval_beam.py --ckpt "$DE/runs/est_dwpose_bodyhands_bbox/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/est_dwpose_bodyhands_bbox" "$G/" 2>/dev/null || true

echo "=== RUN 2: DWPose body_hands (no face) + FRAME-norm  (isolates the norm effect) ==="
python train_ctc.py $COM $REC \
  --num_keypoints 133 --keypoint_subset cwb_body_hands --coord_norm frame \
  --out_dir "$DE/runs/est_dwpose_bodyhands_frame"
python tools/eval_beam.py --ckpt "$DE/runs/est_dwpose_bodyhands_frame/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/est_dwpose_bodyhands_frame" "$G/" 2>/dev/null || true

echo "EST_ABLATION_DONE"
