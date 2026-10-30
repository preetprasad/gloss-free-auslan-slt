#!/bin/bash -l
# LLM-VQ (OpenCode build): Stage A VQ tokeniser over DWPose pose -> discrete sign tokens,
# Stage B mT5-base+LoRA decodes English from the token sequence. The architecturally-different
# route to 15+ (literature ~23.4). Watch Stage A codebook PERPLEXITY (collapse risk).
#SBATCH --job-name=vq-llm
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
pip install -q "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" sentencepiece jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"

echo "=== Stage A: VQ tokeniser (pose, 1024 codes) — WATCH perplexity ==="
python train_vq.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  $POSE --vq_modality pose \
  --vq_codebook_size 1024 --vq_code_dim 256 --vq_num_codebooks 1 --vq_ema_decay 0.99 \
  --vq_commitment_cost 0.25 --batch_size 64 --epochs 50 --lr 3e-4 --lr_schedule cosine \
  --device cuda --num_workers 8 --out_dir "$DE/runs/vq_pose"

echo "=== Stage B SMOKE (64 clips, 1 ep) — set -e gates the full run ==="
python train_vq_llm.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" $POSE --vq_modality pose \
  --vq_encoder_ckpt "$DE/runs/vq_pose/best.pt" --vq_llm_name google/mt5-base \
  --vq_lora_r 16 --vq_lora_alpha 32 --vq_llm_lr 1e-4 \
  --limit_samples 64 --epochs 1 --batch_size 4 --device cuda --out_dir "$DE/runs/vq_llm_smoke"

echo "=== Stage B FULL: mT5+LoRA over VQ tokens ==="
python train_vq_llm.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" $POSE --vq_modality pose \
  --vq_encoder_ckpt "$DE/runs/vq_pose/best.pt" --vq_llm_name google/mt5-base \
  --vq_lora_r 16 --vq_lora_alpha 32 --vq_llm_lr 1e-4 \
  --batch_size 8 --epochs 20 --device cuda --num_workers 8 \
  --bleu_tokenize 13a --detokenize True --out_dir "$DE/runs/comm_dwpose_vq_llm"
cp -r "$DE/runs/comm_dwpose_vq_llm" /group/pmc099/pprasad/
echo "VQ_LLM_DONE"
