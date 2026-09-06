#!/bin/bash -l
# PoseStitch v2 — SEED REPLICATION. Decides whether the +0.71 beam-5 joint gain is real.
#
# The 2026-09-04b run gave (beam-5, seed 42): control 9.88 / real-doubled 8.72 / joint 10.59.
# +0.71 sits at the EDGE of this project's ±0.4-0.8 batch-noise band, so a single seed cannot
# settle it. This runs PAIRED control-vs-joint at two fresh seeds on the SAME 6,250-clip corpus,
# changing nothing but --seed. Corpus size is deliberately held fixed here; the "does more
# synthetic help" question is a separate job (posestitch_v2_scale.sbatch) so the two effects
# never get confounded.
#
# Read the result as: joint MUST beat its own same-seed control in BOTH pairs for the gain to
# be called real. One win and one loss = noise, and the lever goes back on the shelf.
#SBATCH --job-name=stitch-seeds
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

SYN="$ST/corpus/posestitch_v2_train.tsv"
test -s "$SYN" || { echo "FAIL: synthetic corpus missing at $SYN"; exit 1; }
test -s "$ST/joint_train.tsv" || cat "$DE/manifests/train.tsv" "$SYN" > "$ST/joint_train.tsv"
echo "corpus: $(wc -l < "$SYN") synthetic clips | joint: $(wc -l < "$ST/joint_train.tsv") | real: $(wc -l < "$DE/manifests/train.tsv")"

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

run_arm () {   # $1 = name, $2 = train manifest, $3 = seed
  echo "=== ARM: $1 (seed $3) ==="
  python train_ctc.py --train_manifest "$2" $VAL $POSE $BASE --seed "$3" --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
}

# Paired, seed by seed, so a partial run still yields a complete pair.
for S in 101 202; do
  run_arm "sv2_control_s$S" "$DE/manifests/train.tsv"  "$S"
  run_arm "sv2_joint_s$S"   "$ST/joint_train.tsv"      "$S"
done

echo "=== SUMMARY — joint must beat its OWN same-seed control in BOTH pairs ==="
for R in sv2_control_s101 sv2_joint_s101 sv2_control_s202 sv2_joint_s202; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('%-20s val_best=%.3f@ep%-3d test_greedy=%.3f wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e:
    print('%-20s NO RESULT %s' % ('$R', e))
"
done
echo "(seed-42 reference: control 9.88 / joint 10.59 beam-5)"
echo "STITCH_SEEDS_DONE"
