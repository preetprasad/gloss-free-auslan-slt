#!/bin/bash -l
# HAND-LOCAL NORMALISATION — the first representation-level lever with a measured, pre-registered
# target since the DWPose swap.
#
# WHY THIS EXISTS (results/CONTINUOUS_ID_PROBE_2026-09-09.md), all measured WITHOUT training anything:
#   The project's bottleneck is visual-lexical identification (70.1% of clips get zero content words
#   right; the model emits the right frequent word 85% of the time but lands it on the right clip
#   20.8%). Every prior visual probe was CROSS-domain (isolated->continuous, spotting recall 0.05),
#   which cannot separate "the representation is too weak" from "isolated->continuous transfer fails".
#   The continuous-domain probe settles it, using Auslan-Daily's own Sign_Spotting spans:
#     same descriptor, same 45 glosses, 8 exemplars/class, CLIP-DISJOINT folds, chance 2.22%
#       ISOLATED   (MM-WLAuslan citation) hands top-1 40.0%  top-5 70.3%
#       CONTINUOUS (real Auslan-Daily)    hands top-1 18.1%  top-5 34.7%   <- loses 55%
#   So the pose representation DOES carry sign identity; continuous signing is what destroys it, and
#   HANDS degrade most (-21.9 pts vs body -12.4) despite carrying the most signal.
#   Hand-local normalisation (each hand around its own wrist at palm scale) recovers:
#       CONTINUOUS hands top-1 18.1% -> 24.9%  (+6.8 = 31% of the domain gap), top-5 34.7% -> 46.9%
#       stable over 5 seeds (24.6-25.8%)
#   and it does NOT help the isolated domain (40.0% -> 39.2%), i.e. it is specifically a
#   continuous-signing fix -- which is the domain we translate.
#
# The flag is a 133-keypoint port of the hand half of `hierarchical_pose_norm`, which hard-requires
# HALPE-136 and has therefore NEVER executed since the DWPose switch -- a dormant improvement.
#
# A BETTER PROBE NUMBER IS NOT A BLEU RESULT. Pre-registered rule, same as every other lever here:
# the arm must beat its OWN same-seed control on BOTH validation and test, in all three pairs.
#SBATCH --job-name=hand-local
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
G=/group/pmc099/pprasad; B=$G/scratch_backup
cd "$G/code/phase2_baseline"
OUT=$G/hand_local; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

# --- DATA ROOT IS /group, NOT /scratch ------------------------------------------------------------
# Manifests ship absolute /scratch clip paths, so rewrite them onto the durable backup at runtime and
# verify every referenced clip exists BEFORE training -- otherwise a purge would silently reduce the
# training set instead of failing. This makes the job immune to the 21-day scratch purge.
SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in train val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  elif [ -f "$B/manifests/$f.tsv" ]; then cp "$B/manifests/$f.tsv" "$M/$f.tsv"
  else echo "FAIL: cannot source manifest $f.tsv"; exit 1; fi
done
wc -l "$M"/train.tsv "$M"/val.tsv "$M"/test.tsv
python - "$M" <<'PYCHK'
import os, sys
root = sys.argv[1]
bad = tot = 0
for f in ("train", "val", "test"):
    for line in open(os.path.join(root, f + ".tsv")):
        if "\t" not in line:
            continue
        tot += 1
        if not os.path.exists(line.split("\t", 1)[0]):
            bad += 1
print(f"manifest clip check: {tot} referenced, {bad} missing")
sys.exit(1 if bad else 0)
PYCHK


# fail loudly rather than silently training the control twice
python - <<'PY'
import sys; sys.path.insert(0,'.')
from config import Config
assert hasattr(Config(), 'hand_local_norm'), "hand_local_norm missing -- sync config.py"
print("hand_local_norm flag present")
PY

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --train_manifest $M/train.tsv --val_manifest $M/val.tsv --test_manifest $M/test.tsv"

run () {  # $1=tag $2=seed $3=hand_local
  echo "=== ARM $1 (seed $2, hand_local_norm=$3) ==="
  python train_ctc.py $POSE $REC --seed "$2" --hand_local_norm "$3" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}

for S in 42 101 202; do
  run "hl_ctrl_s$S"  "$S" False   # in-batch control: identical recipe, flag off
  run "hl_hand_s$S"  "$S" True
done

echo "=== SUMMARY — rule: hand-local must beat its OWN same-seed control on val AND test, 3/3 ==="
for S in 42 101 202; do
  for A in hl_ctrl_s$S hl_hand_s$S; do
    python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-14s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-14s NO RESULT %s' % ('$A', e))
"
  done
done
echo "HAND_LOCAL_DONE"
