#!/bin/bash -l
# Stack-1: scaled translator (d512, 6+6 layers) on DWPose pose. "Bigger didn't help" was
# true with broken pose; retest now the encoder carries real hand signal.
#SBATCH --job-name=dwpose-d512
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
python train.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --d_model 512 --num_encoder_layers 6 --num_decoder_layers 6 --dim_feedforward 2048 \
  --batch_size 32 --epochs 40 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_d512"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_d512/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/comm_dwpose_d512" /group/pmc099/pprasad/
echo "DWPOSE_D512_DONE"
