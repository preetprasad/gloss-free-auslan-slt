#!/bin/bash -l
# Full GASLT recipe: deformable gloss (local temporal) attention encoder on DWPose pose + CTC.
# Addresses the temporal-redundancy / collapse gap. Attention already numerically verified.
#SBATCH --job-name=gaslt
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
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
python train_ctc.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --model_arch gaslt --gaslt_num_keys 7 --d_model 256 --num_encoder_layers 4 --num_decoder_layers 4 \
  --batch_size 32 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_gaslt"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_gaslt/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda 2>/dev/null || true
cp -r "$DE/runs/comm_dwpose_gaslt" /group/pmc099/pprasad/
echo "GASLT_DONE"
