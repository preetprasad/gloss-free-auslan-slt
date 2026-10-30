#!/bin/bash -l
# GRPO RL fine-tuning of the VLP best translator (10.27). The 2025 open-domain SOTA mechanism
# (SLT-GRPO): reward = 0.5*sentenceBLEU + 0.5*ROUGE-L, group-relative advantage, KL to frozen ref.
# Current best to beat = 10.335 (RGB fusion); this isolates RL's effect on the pose model first.
#SBATCH --job-name=grpo
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
pip install -q sacrebleu jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
INIT=/group/pmc099/pprasad/comm_dwpose_vlp_ctc/best.pt

python train_grpo.py --init_ckpt "$INIT" \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --grpo_group 8 --grpo_beta 0.2 --grpo_temp 0.9 \
  --batch_size 8 --epochs 3 --lr 5e-7 --warmup_steps 100 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_grpo_v2"
cp -r "$DE/runs/comm_dwpose_grpo" /group/pmc099/pprasad/
echo "GRPO_DONE"
