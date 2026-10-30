#!/bin/bash -l
# BLEU push — temporal frame compression (#1 architecture gap: frames-per-sign redundancy → collapse).
# Average-pool consecutive DWPose frames toward sign rate BEFORE the encoder. Interface-preserving
# (only the input sequence shortens; CTC/decoder unchanged). Matched DWPose+CTC recipe, only
# temporal_pool differs. pool=1 = established baseline (9.63). Validation selects; test reported for
# reference but NOT used to choose the factor.
#SBATCH --job-name=temporal-pool-ctc
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

echo "=== SMOKE (pool=2, 64 clips, 1 ep) — fail fast if the pooling path is broken ==="
python train_ctc.py $COM $POSE $REC --temporal_pool 2 --epochs 1 --limit_samples 64 \
  --out_dir "$DE/runs/tpool_smoke"

for K in 1 2 3; do
  echo "=== temporal_pool=$K (matched DWPose+CTC) ==="
  python train_ctc.py $COM $POSE $REC --temporal_pool $K --out_dir "$DE/runs/tpool_$K"
  python tools/eval_beam.py --ckpt "$DE/runs/tpool_$K/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$DE/runs/tpool_$K" "$G/comm_dwpose_tpool_$K" 2>/dev/null || true
done
echo "TEMPORAL_POOL_DONE"
