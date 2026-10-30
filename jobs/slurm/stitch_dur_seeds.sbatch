#!/bin/bash -l
# SEED REPLICATION of the duration-modelled stitch corpus.
#
# Job 1175390 (seed 42) gave beam-5 11.08 vs an in-batch control of 9.49 (+1.59), with validation
# ordering the arms MONOTONICALLY with temporal fidelity: control 9.908 < no-resample 9.987 <
# fps-corrected 10.363 < duration-modelled 10.625, as mean clip length went 246 -> 206 -> 108
# frames against a real Auslan-Daily mean of 91.2. That is a dose-response, not one lucky arm.
#
# But it is ONE SEED, and this project's rule is that a stitch result is not claimable until it
# beats its OWN same-seed control in every pair -- the discipline that made the base +0.56 lever
# credible (3/3 seeds) and that has caught test-tuned numbers before. Same protocol here.
#
# Reuses the seed-42 corpus at $ST/corpus_dur (built with --sign_dur), so only --seed moves.
#SBATCH --job-name=dur-seeds
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
test -s "$ST/joint_dur.tsv" || { echo "FAIL: duration-modelled joint manifest missing"; exit 1; }
wc -l "$DE/manifests/train.tsv" "$ST/joint_dur.tsv"

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
run () {  # $1 name, $2 manifest, $3 seed
  echo "=== ARM $1 (seed $3) ==="
  python train_ctc.py --train_manifest "$2" $VAL $POSE $BASE --seed "$3" --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
}
# paired, seed by seed, so a partial run still leaves a COMPLETE pair
for S in 101 202; do
  run "dur_control_s$S" "$DE/manifests/train.tsv" "$S"
  run "dur_joint_s$S"   "$ST/joint_dur.tsv"       "$S"
done

echo "=== SUMMARY — duration-modelled joint must beat its OWN same-seed control in BOTH pairs ==="
for R in dur_control_s101 dur_joint_s101 dur_control_s202 dur_joint_s202; do
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
echo "seed-42 reference (beam-5): control 9.49 | duration-modelled joint 11.08 (+1.59)"
echo "DUR_SEEDS_DONE"
