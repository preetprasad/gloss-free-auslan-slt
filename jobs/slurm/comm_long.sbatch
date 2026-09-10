#!/bin/bash -l
# COMMUNICATION AT A LONGER SCHEDULE — does the whole ladder move?
#
# WHY (results/EPOCH_BUDGET_AUDIT_2026-09-09.md). An audit of all 140 saved validation histories found
# 35 runs peaked in the final 5% of their schedule, and a proper OLS slope + SE over the final 20
# epochs shows the plain pose+CTC line is STILL CLIMBING at cut-off: graphfair_2 t=5.42,
# ctc_fw_0.5 t=6.45, ctc_rdrop_5.0 t=12.08, dur_joint_s101 t=2.33, at roughly +0.05-0.10 BLEU/epoch.
# Controls rise at the same rate, so the project's PAIRED DELTAS are unbiased and no closed verdict
# needs reopening -- but every ABSOLUTE number, including the 11.80 headline, is understated.
#
# This is the only cheap untested lever that could lift the entire ladder rather than one arm.
#
# DESIGN: three schedules (40 / 80 / 150) at two seeds, everything else identical, with the 40-epoch
# arm as the IN-BATCH control reproducing the standard recipe. Three points rather than two because:
#   (a) the LR trajectory is CONFOUNDED with epoch count -- `train_ctc.py:165` sets
#       steps = epochs * len(train_dl) and WarmupCosine anneals to 0.05*base_lr over that period, so at
#       epoch 40 a 150-epoch run sits at ~4.2e-4 against a 40-epoch run's ~2.5e-5, a ~17x difference.
#       "Train longer" therefore cannot be a pure epoch ablation; it is the schedule anyone would
#       actually run. Three points show whether the effect is monotonic in budget rather than a
#       one-off LR artefact.
#   (b) they reveal SATURATION, which a two-point comparison cannot.
# DO NOT extrapolate the +0.06/epoch slope: saturation is expected and is precisely what this measures.
#
# Selection stays validation-only. The test number reported per arm is its val-selected checkpoint.
#SBATCH --job-name=comm-long
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
OUT=$G/comm_long; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

# data root is /group so the job is immune to the 21-day scratch purge whenever it is scheduled
SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in train val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  else echo "FAIL: cannot source $SRC/$f.tsv"; exit 1; fi
done
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
wc -l "$M"/train.tsv "$M"/val.tsv "$M"/test.tsv

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
      --train_manifest $M/train.tsv --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
run () {   # $1=tag $2=epochs $3=seed
  echo "=== ARM $1 (epochs=$2, seed=$3) ==="
  python train_ctc.py $POSE $BASE --epochs "$2" --seed "$3" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}
for S in 42 101; do
  run "cl_e040_s$S"  40  "$S"      # in-batch control: the standard recipe
  run "cl_e080_s$S"  80  "$S"
  run "cl_e150_s$S" 150  "$S"
done

echo "=== SUMMARY — is the gain monotonic in budget, and where does validation turn over? ==="
for S in 42 101; do
  for A in cl_e040_s$S cl_e080_s$S cl_e150_s$S; do
    python -c "
import json
d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
h=[e for e in d.get('history',[]) if isinstance(e,dict) and e.get('bleu4') is not None]
print('%-14s val_best=%6.3f @ep%-4d of %-4d test_greedy=%6.3f wer=%.3f' % (
    '$A', d['best']['bleu4'], d['best']['epoch'], d['epochs'],
    t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
if h: print('    last 8 val:', [(e['epoch'], round(e['bleu4'],2)) for e in h[-8:]])
"
  done
done
echo "reference: the standard 40-epoch recipe scores beam-5 ~9.6-9.9 on this split"
echo "COMM_LONG_DONE"
