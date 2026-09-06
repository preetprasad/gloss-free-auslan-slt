#!/bin/bash -l
# VLP x PoseStitch — 2x2 factorial. Stacks the project's two validated wins for the first time.
#
# WHY: our best single model is VLP-80 -> CTC (10.71), and PoseStitch v2 joint training is
# +0.71 over its own control (10.59 vs 9.88) -- but the two have never been combined. If the
# effects are even roughly additive this lands a SINGLE model near the 11.50 ensemble.
#
# This is NOT the v1 failure mode. v1 did disjoint synthetic-pretrain -> real-finetune, which is
# the arm that collapsed on PHOENIX (11.38 vs 21.38). Here pretraining is CLIP-style pose-text
# VLP, and the synthetic data enters as JOINT training data -- the configuration that won.
#
# 2x2: {VLP on real, VLP on real+synthetic} x {CTC on real, CTC on real+synthetic}. Two pretrains
# feed four finetunes, so it separates "synthetic helps representation learning" from "synthetic
# helps translation training" instead of confounding them.
#
#   A1 VLP(real)  -> CTC(real)      in-batch control; should land near the historical 10.71
#   A2 VLP(real)  -> CTC(joint)     THE KEY ARM
#   A3 VLP(joint) -> CTC(joint)     synthetic in both stages
#   A4 VLP(joint) -> CTC(real)      synthetic in pretraining only
#
# Arms are ordered so a timeout still leaves the decisive A1/A2 pair complete. A2 is compared to
# A1 (same pretrain, only the finetune manifest differs) -- internally controlled, nothing else moves.
# VLP epochs = 80: the measured peak of the inverted-U scaling curve (40->10.27, 80->10.71,
# 128->9.69, 256->9.66). Corpus is the SAME 6,250-clip set that produced 10.59, so the non-VLP
# result stays a valid reference point.
#SBATCH --job-name=vlp-stitch
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"

REAL="$DE/manifests/train.tsv"
JOINT="$ST/joint_train.tsv"
test -s "$ST/corpus/posestitch_v2_train.tsv" || { echo "FAIL: synthetic corpus missing"; exit 1; }
test -s "$JOINT" || cat "$REAL" "$ST/corpus/posestitch_v2_train.tsv" > "$JOINT"
echo "real=$(wc -l < "$REAL") joint=$(wc -l < "$JOINT")"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
VAL="--val_manifest $DE/manifests/val.tsv"
TEST="--test_manifest $DE/manifests/test.tsv"
FT="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
    --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

vlp_pretrain () {   # $1 = name, $2 = train manifest
  echo "=== VLP-80 contrastive pretraining: $1 ($2) ==="
  python pretrain_vlp.py --train_manifest "$2" $VAL $POSE \
    --batch_size 64 --epochs 80 --lr 5e-4 --lr_schedule cosine \
    --device cuda --num_workers 8 --out_dir "$ST/runs/$1"
}

finetune () {       # $1 = arm name, $2 = CTC train manifest, $3 = VLP encoder ckpt
  echo "=== ARM $1: CTC on $(basename "$2") from $(basename "$(dirname "$3")") ==="
  python train_ctc.py --train_manifest "$2" $VAL $TEST $POSE \
    --init_encoder "$3" $FT --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
  cp -r "$ST/runs/$1" "$G/comm_dwpose_$1" 2>/dev/null || true
}

# ---- pretrain on REAL, then the decisive pair -------------------------------
vlp_pretrain vlp80_real "$REAL"
VR="$ST/runs/vlp80_real/vlp_encoder_best.pt"
finetune vs_A1_vlpreal_ctcreal  "$REAL"  "$VR"
finetune vs_A2_vlpreal_ctcjoint "$JOINT" "$VR"

# ---- pretrain on REAL+SYNTHETIC, then the remaining pair ---------------------
vlp_pretrain vlp80_joint "$JOINT"
VJ="$ST/runs/vlp80_joint/vlp_encoder_best.pt"
finetune vs_A3_vlpjoint_ctcjoint "$JOINT" "$VJ"
finetune vs_A4_vlpjoint_ctcreal  "$REAL"  "$VJ"

echo "=== SUMMARY — A2 vs A1 is the decisive comparison (same pretrain, only finetune data differs) ==="
for R in vs_A1_vlpreal_ctcreal vs_A2_vlpreal_ctcjoint vs_A3_vlpjoint_ctcjoint vs_A4_vlpjoint_ctcreal; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('%-26s val_best=%.3f@ep%-3d test_greedy=%.3f wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e:
    print('%-26s NO RESULT %s' % ('$R', e))
"
done
echo "references (beam-5): no-VLP control 9.88 | no-VLP joint 10.59 | historical VLP-80 10.71 | S6 ensemble 11.50"
echo "VLP_STITCH_2X2_DONE"
