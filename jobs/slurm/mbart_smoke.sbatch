#!/bin/bash -l
# mBART decoder SMOKE: 64 clips, 1 epoch — validate the pose->adapter->mBART(BPE) pipeline runs
# end-to-end (forward loss, generate, BLEU) before the multi-day full run. set -e gates it.
#SBATCH --job-name=mbart-smoke
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" sentencepiece sacremoses 2>/dev/null || true
export HF_HOME=/scratch/pmc099/pprasad/hf_cache
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
echo "=== SMOKE (64 clips, 1 ep) ==="
python train_mbart.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --mbart_name facebook/mbart-large-50 --mbart_lora_r 32 \
  --init_encoder "$G/comm_dwpose_ctc_vlplong80/best.pt" \
  --limit_samples 64 --epochs 1 --batch_size 4 --grad_accum 2 --llm_lr 1e-4 \
  --num_beams 3 --device cuda --num_workers 4 --out_dir "$DE/runs/mbart_smoke"
echo "MBART_SMOKE_DONE"
