#!/bin/bash -l
# ctc_weight was fixed at 0.3 in EVERY run and never tuned — a genuine one-and-done default.
# Sweep the CE<->CTC balance on the matched DWPose+CTC recipe (0.3 = the 9.63 baseline control).
# Validation-selected; if a weight beats 0.3 it transfers to the VLP recipe next.
#SBATCH --job-name=ctcweight-sweep
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
REC="--batch_size 64 --epochs 40 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --seed 42 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
for W in 0.1 0.3 0.5 0.7; do
  echo "=== ctc_weight=$W (0.3 = established baseline control) ==="
  python train_ctc.py $COM $POSE $REC --ctc_weight $W --out_dir "$DE/runs/ctcw_$W"
  python tools/eval_beam.py --ckpt "$DE/runs/ctcw_$W/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$DE/runs/ctcw_$W" "$G/comm_dwpose_ctcw_$W" 2>/dev/null || true
done
echo "CTCWEIGHT_SWEEP_DONE"
