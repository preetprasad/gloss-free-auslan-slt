#!/bin/bash -l
# Stage 2b — GFSLT-VLP-style ALIGNMENT then fine-tune for the mBART decoder.
# Phase A (align): mBART fully FROZEN, train only pose-encoder + adapter so they map pose features
#   into the pretrained decoder's space (fixes the Stage-2a val->test overfit from a cold adapter).
# Phase B (fine-tune): resume from A, unfreeze LoRA, fine-tune everything.
#SBATCH --job-name=mbart-align
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
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --mbart_name facebook/mbart-large-50 --mbart_lora_r 32"

echo "=== Phase A: ALIGN (mBART frozen; train adapter + pose encoder) ==="
python train_mbart.py $COM $POSE --mbart_train_lora False \
  --init_encoder "$G/comm_dwpose_ctc_vlplong80/best.pt" \
  --epochs 12 --batch_size 8 --grad_accum 2 --llm_lr 2e-4 --warmup_steps 300 \
  --num_beams 5 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/mbart_align"

echo "=== Phase B: FINE-TUNE (resume from align; unfreeze LoRA) ==="
python train_mbart.py $COM $POSE --mbart_train_lora True \
  --resume_checkpoint "$DE/runs/mbart_align/best.pt" \
  --epochs 15 --batch_size 4 --grad_accum 4 --llm_lr 7e-5 --warmup_steps 300 \
  --num_beams 5 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_mbart_aligned"
cp -r "$DE/runs/comm_dwpose_mbart_aligned" "$G/" 2>/dev/null || true
echo "MBART_ALIGN_DONE"
