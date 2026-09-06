#!/bin/bash -l
# PoseStitch steps B+C — pre-train the translator on the stitched synthetic corpus, then fine-tune on
# real Auslan-Daily. Isolates the PoseStitch effect vs plain DWPose+CTC (9.63, no pre-train). Full
# encoder+decoder transfer via --resume_checkpoint. Validation-selected on Comm val; eval Comm test.
#SBATCH --job-name=posestitch-train
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --seed 42 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== Stage 1: PRE-TRAIN on stitched synthetic corpus (val on Comm to track real progress) ==="
python train_ctc.py $POSE $REC --epochs 30 \
  --train_manifest "$DE/posestitch/posestitch_train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --out_dir "$DE/runs/ctc_stitch_pretrain"

echo "=== Stage 2: FINE-TUNE on real Auslan-Daily (full-model warm-start from stitched) ==="
python train_ctc.py $POSE $REC --epochs 40 \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --resume_checkpoint "$DE/runs/ctc_stitch_pretrain/best.pt" \
  --out_dir "$DE/runs/ctc_stitch_ft"
python tools/eval_beam.py --ckpt "$DE/runs/ctc_stitch_ft/best.pt" --manifest "$DE/manifests/test.tsv" \
  --beam 5 --device cuda || true
cp -r "$DE/runs/ctc_stitch_ft" "$G/comm_dwpose_ctc_stitch_ft" 2>/dev/null || true
echo "POSESTITCH_TRAIN_DONE"
