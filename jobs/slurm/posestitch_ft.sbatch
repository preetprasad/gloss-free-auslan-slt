#!/bin/bash -l
# PoseStitch Stage 2 RE-RUN — fine-tune on real Auslan-Daily from the completed stitched pre-train
# (ctc_stitch_pretrain/best.pt), now with the shape-filtered resume fix. Eval Comm test.
#SBATCH --job-name=posestitch-ft
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
python train_ctc.py $POSE --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine \
  --warmup_steps 800 --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest "$DE/manifests/test.tsv" \
  --resume_checkpoint "$DE/runs/ctc_stitch_pretrain/best.pt" \
  --out_dir "$DE/runs/ctc_stitch_ft"
python tools/eval_beam.py --ckpt "$DE/runs/ctc_stitch_ft/best.pt" --manifest "$DE/manifests/test.tsv" \
  --beam 5 --device cuda || true
cp -r "$DE/runs/ctc_stitch_ft" "$G/comm_dwpose_ctc_stitch_ft" 2>/dev/null || true
echo "POSESTITCH_FT_DONE"
