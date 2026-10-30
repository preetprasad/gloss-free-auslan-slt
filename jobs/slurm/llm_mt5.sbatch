#!/bin/bash -l
# LLM decoder (mT5-base + LoRA on the DWPose pose encoder) — the big lever toward 13+.
# Installs transformers/peft, smoke-tests OpenCode's train_llm.py on 64 real clips, then
# runs full. set -e => if the smoke crashes, the full run never starts.
#SBATCH --job-name=llm-mt5
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
pip install -q "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" sentencepiece   # torch 2.5.1: peft>=0.15 calls torch.distributed.tensor.DTensor (absent); 4.x transformers needs <5
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline

echo "=== SMOKE (64 real DWPose clips) ==="
python train_llm.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --llm_name google/mt5-base --lora_r 16 --lora_alpha 32 \
  --limit_samples 64 --epochs 1 --batch_size 4 --device cuda --out_dir "$DE/runs/llm_smoke"

echo "=== FULL RUN ==="
python train_llm.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --llm_name google/mt5-base --lora_r 16 --lora_alpha 32 --llm_lr 1e-4 \
  --batch_size 8 --grad_accum_steps 2 --epochs 20 --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_llm_mt5"
cp -r "$DE/runs/comm_dwpose_llm_mt5" /group/pmc099/pprasad/
echo "LLM_MT5_DONE"
