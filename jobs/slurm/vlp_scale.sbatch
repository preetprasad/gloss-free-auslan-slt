#!/bin/bash -l
# VLP-pretraining SCALING: 40->80 gave 10.27->10.71, so push further. Trains VLP for 128 and 256
# epochs (seed 101, same as the 80-ep run) then the standard 40-ep CTC fine-tune each. Maps the
# VLP-epochs -> BLEU curve and yields (hopefully) stronger ensemble members.
#SBATCH --job-name=vlp-scale
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

vlp_ctc () {  # $1=vlp_epochs  $2=tag
  local VE=$1 T=$2
  echo "===== [$(date +%T)] VLP ${VE}ep ($T) ====="
  python pretrain_vlp.py $COM $POSE --seed 101 --batch_size 64 --epochs $VE --lr 5e-4 \
    --lr_schedule cosine --device cuda --num_workers 8 --out_dir "$DE/runs/vlp_$T" \
    && echo "[ok] vlp_$T" || { echo "[FAIL] vlp_$T"; return; }
  echo "===== [$(date +%T)] CTC from $T ====="
  python train_ctc.py $COM $POSE --init_encoder "$DE/runs/vlp_$T/vlp_encoder_best.pt" --seed 101 \
    --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
    --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$DE/runs/ctc_$T" \
    && cp -r "$DE/runs/ctc_$T" "$G/comm_dwpose_ctc_$T" && echo "[ok] ctc_$T copied" || echo "[FAIL] ctc_$T"
  python tools/eval_beam.py --ckpt "$DE/runs/ctc_$T/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda 2>/dev/null || true
}

vlp_ctc 128 vlp128
vlp_ctc 256 vlp256
echo "VLP_SCALE_DONE $(date +%T)"
echo "Curve: 40ep=10.27  80ep=10.71  128ep=see ctc_vlp128  256ep=see ctc_vlp256 (beam-5 test)"
