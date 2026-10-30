#!/bin/bash -l
# SWA (within-run checkpoint averaging): re-run the two best recipes saving the last-5 epoch
# checkpoints, average each into a single stronger member (shared-trajectory averaging works,
# unlike the failed independent-seed soup). Then compare vs best.pt; feeds a stronger ensemble.
#SBATCH --job-name=swa
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer sacrebleu 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
K=5

echo "===== SWA member 1: VLP-80 -> CTC (last-$K avg) ====="
ENC=$DE/runs/vlp_vlplong80/vlp_encoder_best.pt
[ -f "$ENC" ] || ENC=$G/comm_dwpose_ctc_vlplong80/best.pt   # fallback: warm from the CTC model
python train_ctc.py $COM $POSE --init_encoder "$ENC" --seed 101 --batch_size 64 --epochs 40 \
  --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --swa_last_k $K \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$DE/runs/swa_ctc_vlp80" \
  && CK=$(ls "$DE"/runs/swa_ctc_vlp80/swa_ep*.pt | paste -sd,) \
  && python weight_average.py --ckpts "$CK" --out "$G/comm_dwpose_swa_vlp80.pt" \
  && echo "--- SWA-VLP80 test beam-5 ---" \
  && python ensemble_decode.py --ckpts "$G/comm_dwpose_swa_vlp80.pt" $POSE \
       --test_manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || echo "[FAIL] swa_vlp80"

echo "===== SWA member 2: RGB-fusion (last-$K avg) ====="
python train_ctc.py $COM $POSE --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 \
  --init_encoder "$G/comm_dwpose_vlp_ctc/best.pt" --seed 101 --batch_size 32 --epochs 40 \
  --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --swa_last_k $K \
  --num_beams 5 --length_penalty 0.6 --bleu_tokenize 13a --detokenize True \
  --device cuda --num_workers 8 --out_dir "$DE/runs/swa_fus" \
  && CK=$(ls "$DE"/runs/swa_fus/swa_ep*.pt | paste -sd,) \
  && python weight_average.py --ckpts "$CK" --out "$G/comm_dwpose_swa_fus.pt" \
  && echo "--- SWA-fus test beam-5 ---" \
  && python ensemble_decode.py --ckpts "$G/comm_dwpose_swa_fus.pt" $POSE \
       --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 \
       --test_manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || echo "[FAIL] swa_fus"

echo "SWA_DONE. Next (CPU): ensemble {swa_vlp80, swa_fus, VLP-80, RGB-fusion, seeds}, val-select."
