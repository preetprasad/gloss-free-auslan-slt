#!/bin/bash -l
# Signer/session-disjoint generalisation — the honest proxy for "signer-disjoint".
# Auslan-Daily's official split is VIDEO-OVERLAPPING (all 90 test videos also appear in train, split at
# the clip level) and the public release has NO per-signer label. So we build a VIDEO-DISJOINT split
# (whole conversations held out: no clip from a test video appears in train) and retrain the SAME
# DWPose+CTC recipe that scored 9.63 on the official split. The gap = generalisation to unseen
# sessions/signers. Split: train 11341 (73 vids) / val 1444 (9) / test 1254 (8), verified disjoint.
#SBATCH --job-name=video-disjoint
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd $G/code/phase2_baseline

echo "=== DWPose+CTC on the VIDEO-DISJOINT split (same recipe as official-split 9.63) ==="
python train_ctc.py \
  --train_manifest "$DE/manifests/vd_train.tsv" --val_manifest "$DE/manifests/vd_val.tsv" \
  --test_manifest "$DE/manifests/vd_test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --epochs 40 --ctc_weight 0.3 \
  --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_ctc_videodisjoint"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_ctc_videodisjoint/best.pt" \
  --manifest "$DE/manifests/vd_test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/comm_dwpose_ctc_videodisjoint" "$G/" 2>/dev/null || true
echo "VIDEO_DISJOINT_DONE"
