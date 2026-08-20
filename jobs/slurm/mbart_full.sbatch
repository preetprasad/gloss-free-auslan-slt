#!/bin/bash -l
# mBART decoder FULL run (Stage 2): pose encoder (warm from VLP-80) -> adapter -> mBART(BPE)+LoRA,
# full official split. The strong-single-model lever; report vs 10.885 single / 11.50 ensemble.
#SBATCH --job-name=mbart-full
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
pip install -q "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" sentencepiece sacremoses 2>/dev/null || true
export HF_HOME=/scratch/pmc099/pprasad/hf_cache
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
python train_mbart.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --mbart_name facebook/mbart-large-50 --mbart_lora_r 32 \
  --init_encoder "$G/comm_dwpose_ctc_vlplong80/best.pt" \
  --epochs 20 --batch_size 4 --grad_accum 4 --llm_lr 1e-4 --warmup_steps 500 \
  --num_beams 5 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_mbart"
cp -r "$DE/runs/comm_dwpose_mbart" "$G/" 2>/dev/null || true
echo "MBART_FULL_DONE"
