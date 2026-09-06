#!/bin/bash -l
# R-Drop consistency regularisation — attacks the one untouched wall (gap #7): single-reference CE +
# exposure bias + overfitting on 12k sentences. Two dropout passes + symmetric KL between them.
# Reported +2 BLEU on low-resource IWSLT. Matched DWPose+CTC recipe (alpha=0 = the 9.63 baseline).
#SBATCH --job-name=rdrop
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
for A in 1.0 5.0; do
  echo "=== R-Drop alpha=$A (DWPose+CTC; baseline control = 9.63 at alpha=0) ==="
  python train_ctc.py $COM $POSE $REC --r_drop_alpha $A --out_dir "$DE/runs/ctc_rdrop_$A"
  python tools/eval_beam.py --ckpt "$DE/runs/ctc_rdrop_$A/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$DE/runs/ctc_rdrop_$A" "$G/comm_dwpose_ctc_rdrop_$A" 2>/dev/null || true
done
echo "RDROP_DONE"
