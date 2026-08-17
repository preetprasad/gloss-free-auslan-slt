#!/bin/bash -l
# MULTI-EXPERIMENT 10h GPU batch — amortises a scarce V100 slot across several experiments.
# All hyperparameters MATCH the proven winning recipes (VLP->CTC = 10.27, RGB-fusion = 10.335):
#   VLP  : batch64 ep40 lr5e-4 cosine     CTC : batch64 ep40 ctc0.3 lr5e-4 cosine warmup800
#   fusion: batch32 ep40 ctc0.3 lr5e-4 cosine warmup800   (only the seed differs)
# Measured wallclock: VLP+CTC ~42 min, fusion ~35 min  => a full seed pipeline ~77 min.
# Contents (well within 10h; each guarded, models copied out on completion):
#   * seeds 101/202/303 x {VLP->CTC, RGB-fusion}  -> 6 new members + n=4 error bars for both recipes
#   * VLP-long (80 ep) -> CTC                      -> does more pose-text pretraining beat 10.27?
# NO `set -e`: one failure never aborts the rest. Ordered so an early cut still yields >=2 seeds.
#SBATCH --job-name=gpu-batch
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"

vlp_ctc () {  # $1=seed  $2=tag  $3=vlp_epochs
  local S=$1 T=$2 VE=${3:-40}
  echo "===== [$(date +%T)] VLP seed=$S ($T) ep=$VE ====="
  python pretrain_vlp.py $COM $POSE --seed $S --batch_size 64 --epochs $VE --lr 5e-4 \
    --lr_schedule cosine --device cuda --num_workers 8 --out_dir "$DE/runs/vlp_$T" \
    && echo "[ok] vlp_$T" || { echo "[FAIL] vlp_$T"; return; }
  echo "===== [$(date +%T)] CTC seed=$S ($T) ====="
  python train_ctc.py $COM $POSE --init_encoder "$DE/runs/vlp_$T/vlp_encoder_best.pt" --seed $S \
    --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
    --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$DE/runs/ctc_$T" \
    && cp -r "$DE/runs/ctc_$T" "$G/comm_dwpose_ctc_$T" && echo "[ok] ctc_$T copied" || echo "[FAIL] ctc_$T"
  python tools/eval_beam.py --ckpt "$DE/runs/ctc_$T/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda 2>/dev/null || true
}

fusion () {  # $1=seed  $2=tag  $3=warmstart ckpt
  local S=$1 T=$2 INIT=$3
  echo "===== [$(date +%T)] RGB-fusion seed=$S ($T) ====="
  python train_ctc.py $COM $POSE --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 \
    --init_encoder "$INIT" --seed $S --batch_size 32 --epochs 40 --ctc_weight 0.3 --lr 5e-4 \
    --lr_schedule cosine --warmup_steps 800 --num_beams 5 --length_penalty 0.6 \
    --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$DE/runs/fus_$T" \
    && cp -r "$DE/runs/fus_$T" "$G/comm_dwpose_fus_$T" && echo "[ok] fus_$T copied" || echo "[FAIL] fus_$T"
}

# --- 3 seeds, full proven recipe (40 ep) => error bars + ensemble members ---
vlp_ctc 101 s101; fusion 101 s101 "$DE/runs/ctc_s101/best.pt"
vlp_ctc 202 s202; fusion 202 s202 "$DE/runs/ctc_s202/best.pt"
vlp_ctc 303 s303; fusion 303 s303 "$DE/runs/ctc_s303/best.pt"

# --- stronger-VLP probe: 80-epoch pose-text pretraining, then standard CTC ---
vlp_ctc 101 vlplong80 80

echo "GPU_BATCH_DONE $(date +%T)"
echo "Members in /group: comm_dwpose_{ctc,fus}_s101/s202/s303 + comm_dwpose_ctc_vlplong80"
echo "Next (CPU): ensemble_decode over existing {rgb_fusion,vlp_ctc} + the new members, val-select beam/LP + weights."
