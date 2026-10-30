#!/bin/bash -l
# ISLR->CTC transfer: warm-start the DWPose+CTC translator from the MM-WLAuslan ISLR encoder
# (val top-1 80.6% on 3,215 signs). Clean isolation of the ISLR-pretraining effect vs the
# from-scratch DWPose+CTC baseline (9.63) and the VLP best (10.27).
#SBATCH --job-name=islr-transfer
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
cd /group/pmc099/pprasad/code/phase2_baseline
INIT=/group/pmc099/pprasad/islr_dwpose/islr_encoder_best.pt

echo "=== ISLR->CTC transfer (init_encoder = $INIT) ==="
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --init_encoder "$INIT" \
  --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_islr_ctc"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_islr_ctc/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/comm_dwpose_islr_ctc" /group/pmc099/pprasad/
echo "ISLR_TRANSFER_DONE"
