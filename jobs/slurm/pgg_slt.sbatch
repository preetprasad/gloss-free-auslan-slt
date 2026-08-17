#!/bin/bash -l
# PGG-SLT: LLM pseudo-gloss (Qwen2.5-3B) -> CTC. Top training lever from the research.
# Stage A: generate ordered gloss for every train subtitle. Stage B: DWPose+CTC supervised by
# those glosses, warm-started from the VLP 10.27 encoder. Beats the rule-based pseudo-gloss?
#SBATCH --job-name=pgg-slt
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
pip install -q "transformers==4.46.3" "accelerate==1.1.1" sentencepiece jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline

echo "=== Stage A: generate LLM pseudo-glosses (Qwen2.5-3B) ==="
python gen_pseudo_gloss.py --manifest "$DE/manifests/train.tsv" \
  --out "$DE/runs/pgg_glosses.tsv" --model Qwen/Qwen2.5-3B-Instruct --device cuda
echo "sample:"; head -5 "$DE/runs/pgg_glosses.tsv"

echo "=== Stage B: DWPose + CTC on LLM pseudo-glosses (warm-start VLP 10.27) ==="
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --gloss_map "$DE/runs/pgg_glosses.tsv" --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" \
  --batch_size 64 --epochs 40 --ctc_weight 0.5 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_pgg_ctc"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_pgg_ctc/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda 2>/dev/null || true
cp -r "$DE/runs/comm_dwpose_pgg_ctc" /group/pmc099/pprasad/
echo "PGG_SLT_DONE"
