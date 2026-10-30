#!/bin/bash -l
# Frequency-aware CTC weighting — targets the diagnosed long-tail vocabulary bottleneck (unseen ~6).
# Upweights pseudo-gloss targets by inverse corpus frequency so the model attends to rare content words.
# Matched DWPose+CTC recipe; alpha=0 = the 9.63 baseline (identical code path). The metric that matters
# is the UNSEEN-target BLEU (via metric_suite), not just overall. Validation-selected.
#SBATCH --job-name=ctc-freqwt
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
pip install -q jiwer sacrebleu 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
DUMP=$G/code/results/semantic_grading/dump_hyps.py
cd "$G/code/phase2_baseline"
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
mkdir -p "$DE/fw_hyps"

for A in 0.0 0.5 1.0; do
  echo "=== frequency-aware CTC, alpha=$A (0.0 = baseline control) ==="
  OUT="$DE/runs/ctc_fw_$A"
  python train_ctc.py $COM $POSE $REC --ctc_freq_weight $A --out_dir "$OUT"
  echo "--- seen/unseen breakdown (alpha=$A) ---"
  python "$DUMP" --ckpt "$OUT/best.pt" --test_manifest "$DE/manifests/test.tsv" \
    $POSE --beam 5 --device cuda --out "$DE/fw_hyps/hyps_fw_$A.jsonl" || true
  python metric_suite.py --in "$DE/fw_hyps/hyps_fw_$A.jsonl" --field hyp_beam \
    --train_manifest "$DE/manifests/train.tsv" || true
  cp -r "$OUT" "$G/comm_dwpose_ctc_fw_$A" 2>/dev/null || true
done
echo "CTC_FREQWEIGHT_DONE"
