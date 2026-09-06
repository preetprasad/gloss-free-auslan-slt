#!/bin/bash -l
# STACK the duration-modelled stitch corpus onto the BEST recipe (VLP-80), and build an
# ensemble-ready member. Both are untried.
#
# WHY NOW: duration-modelled joint training replicates 3/3 paired seeds at mean beam-5 10.97
# (+1.48 over paired controls), every arm beating the previous best single model 10.71 (VLP-80).
# But every stitch experiment so far has run on the plain DWPose+CTC recipe. The earlier VLP x
# stitch 2x2 showed the BASE corpus already gave +0.98 on the VLP recipe (A2-A1, validation-
# supported); the duration-modelled corpus is ~3x stronger, and that combination is untested.
#
# Arms (all seed 42, VLP-80 pretraining on REAL data shared by both finetunes so only the
# finetune manifest moves -- the same internally-controlled design as the earlier 2x2):
#   V1  VLP-80(real) -> CTC(real)                in-batch control for this recipe
#   V2  VLP-80(real) -> CTC(real + duration-modelled synthetic)     THE TEST
# Ordered so a timeout still leaves the decisive pair complete.
#SBATCH --job-name=dur-vlp
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
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad; ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"
test -s "$ST/joint_dur.tsv" || { echo "FAIL: duration-modelled manifest missing"; exit 1; }

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
VAL="--val_manifest $DE/manifests/val.tsv"
TEST="--test_manifest $DE/manifests/test.tsv"
FT="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
    --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== VLP-80 contrastive pretraining on REAL data (shared by both arms) ==="
python pretrain_vlp.py --train_manifest "$DE/manifests/train.tsv" $VAL $POSE \
  --batch_size 64 --epochs 80 --lr 5e-4 --lr_schedule cosine \
  --device cuda --num_workers 8 --out_dir "$ST/runs/vlp80_dur_base"
VR="$ST/runs/vlp80_dur_base/vlp_encoder_best.pt"

run () {  # $1 name, $2 finetune manifest
  echo "=== ARM $1 ($2) ==="
  python train_ctc.py --train_manifest "$2" $VAL $TEST $POSE --init_encoder "$VR" $FT \
    --out_dir "$ST/runs/$1" --swa_last_k 5
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$ST/runs/$1" "$G/comm_dwpose_$1" 2>/dev/null || true   # make it ensemble-reachable
}
run dvlp_V1_ctcreal "$DE/manifests/train.tsv"
run dvlp_V2_ctcdur  "$ST/joint_dur.tsv"

echo "=== SUMMARY — V2 vs V1 is the decisive pair (shared pretrain, only finetune data differs) ==="
for R in dvlp_V1_ctcreal dvlp_V2_ctcdur; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('%-18s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e: print('%-18s NO RESULT %s' % ('$R', e))
"
done
echo "refs (beam-5): DWPose+CTC control ~9.5 | dur-stitch on DWPose+CTC 10.97 mean (3 seeds)"
echo "              | historical VLP-80 10.71 | SWA-VLP80 10.885 | S6 ensemble 11.80"
echo "DUR_VLP_DONE"
