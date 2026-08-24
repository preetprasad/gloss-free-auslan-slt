#!/bin/bash -l
# Estimator-vs-normalisation disambiguation — the cell owed by est-ablation (job 1100623).
# Completes the 2x2: {AlphaPose, DWPose} x {bbox, frame} under one identical pipeline/recipe.
#   DWPose  body_hands bbox  = 1.45 (collapsed)   DWPose  body_hands frame = 10.03   [already have]
#   AlphaPose body_hands bbox = RUN 1 (re-confirm ~2.87)  AlphaPose body_hands frame = RUN 2  [KEY]
# If RUN 2 stays ~3, the ESTIMATOR is the driver (DWPose-frame 10.03 >> AlphaPose-frame ~3, same norm).
# If RUN 2 jumps toward ~10, the NORMALISATION was doing the work, not the estimator.
#SBATCH --job-name=ap-ablation
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
AM=/scratch/pmc099/pprasad/alphapose/manifests; R=/scratch/pmc099/pprasad/alphapose/runs; G=/group/pmc099/pprasad
cd $G/code/phase2_baseline
COM="--train_manifest $AM/ap_train.tsv --val_manifest $AM/ap_val.tsv --test_manifest $AM/ap_test.tsv"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== RUN 1: AlphaPose body_hands + BBOX-norm (re-confirm ~2.87 control, matched pipeline) ==="
python train_ctc.py $COM $REC --num_keypoints 136 --keypoint_subset body_hands --coord_norm bbox \
  --out_dir "$R/ap_bodyhands_bbox"
python tools/eval_beam.py --ckpt "$R/ap_bodyhands_bbox/best.pt" --manifest $AM/ap_test.tsv --beam 5 --device cuda
cp -r "$R/ap_bodyhands_bbox" "$G/" 2>/dev/null || true

echo "=== RUN 2: AlphaPose body_hands + FRAME-norm (THE disambiguating cell) ==="
python train_ctc.py $COM $REC --num_keypoints 136 --keypoint_subset body_hands --coord_norm frame \
  --out_dir "$R/ap_bodyhands_frame"
python tools/eval_beam.py --ckpt "$R/ap_bodyhands_frame/best.pt" --manifest $AM/ap_test.tsv --beam 5 --device cuda
cp -r "$R/ap_bodyhands_frame" "$G/" 2>/dev/null || true
echo "AP_ABLATION_DONE"
