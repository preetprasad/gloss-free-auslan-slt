#!/bin/bash -l
# Stack-1: DWPose pose + pseudo-gloss CTC grounding (combine the two anti-collapse wins).
#SBATCH --job-name=dwpose-ctc
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --epochs 40 --ctc_weight 0.3 \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_ctc"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_ctc/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/comm_dwpose_ctc" /group/pmc099/pprasad/
echo "DWPOSE_CTC_DONE"
