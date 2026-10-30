#!/bin/bash -l
# COMPOUND TWO CONFIRMED LEVERS: train the duration-stitch recipe at the newly-measured 80-epoch optimum.
#
# Three results now point at this experiment:
#   (1) duration-stitch joint training = +1.48 beam-5, replicated 3/3 (2026-09-06);
#   (2) `comm-long` (1196386) finds an INTERIOR epoch optimum on the plain recipe -- beam-5
#       40ep 9.21 -> 80ep 9.83 (+0.62) -> 150ep 8.31. 80 is right, 150 overtrains badly;
#   (3) the new best operating point, F @ 12.114 test, is an ensemble whose duration members were ALL
#       trained at 40 epochs -- i.e. at a budget now known to be suboptimal for this architecture.
# So the members carrying the current record were undertrained. This retrains them at 80.
#
# WHY IT IS NOT A FOREGONE CONCLUSION. The 80-epoch gain was measured on REAL-ONLY data. The duration
# arm trains on 19,319 clips (real + 6,880 synthetic), so it already sees ~55% more data per epoch and
# may sit at a different point on the same curve -- more data can substitute for more epochs. It could
# equally overtrain sooner, since `comm-long` shows the penalty past the optimum is steep (-1.52).
# That is exactly why a 40-epoch in-batch control is included rather than reusing the existing members.
#
# ARMS: 40ep seed 42 (in-batch control, must land near the recorded 10.625 val / 11.08 beam-5)
#       80ep seeds 42, 101, 202
# Reusing the existing 40-epoch members as the control would be a cross-batch comparison, and this
# session has already produced two cases where that hid a real difference.
#SBATCH --job-name=dur-long
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
OUT=$G/dur_long; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in train val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  else sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$B/dwpose/manifests/$f.tsv" > "$M/$f.tsv"; fi
done

# the duration corpus: reuse the backed-up seed-42 build that produced the replicated +1.48, so the
# ONLY thing moving in this job is the epoch budget.
CORP="$B/stitch_meta/stitch_v2/corpus_dur/posestitch_v2_train.tsv"
[ -f "$CORP" ] || { echo "FAIL: duration corpus missing at $CORP"; exit 1; }
sed "s#/scratch/pmc099/pprasad/stitch_v2/corpus_dur#$B/stitch_meta/stitch_v2/corpus_dur#g" "$CORP" > "$OUT/syn.tsv"
NS=$(wc -l < "$OUT/syn.tsv"); echo "synthetic clips: $NS"
[ "$NS" -ge 6800 ] || { echo "FAIL: expected ~6880 synthetic clips, got $NS"; exit 1; }
# verify the synthetic clips actually exist on /group before training on them
python - "$OUT/syn.tsv" <<'PYCHK'
import os, sys
bad = tot = 0
for line in open(sys.argv[1]):
    if "\t" not in line: continue
    tot += 1
    if not os.path.exists(line.split("\t", 1)[0]): bad += 1
print(f"synthetic clip check: {tot} referenced, {bad} missing")
sys.exit(1 if bad else 0)
PYCHK
cat "$M/train.tsv" "$OUT/syn.tsv" > "$OUT/joint.tsv"; wc -l "$M/train.tsv" "$OUT/joint.tsv"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
      --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
run () {  # $1 tag  $2 epochs  $3 seed
  echo "=== ARM $1 (epochs=$2, seed=$3) ==="
  python train_ctc.py --train_manifest "$OUT/joint.tsv" $POSE $BASE --epochs "$2" --seed "$3" \
    --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}
run "dl_e040_s42" 40 42
for S in 42 101 202; do run "dl_e080_s$S" 80 "$S"; done

echo "=== SUMMARY — does the 80-epoch optimum carry to the duration recipe? ==="
for A in dl_e040_s42 dl_e080_s42 dl_e080_s101 dl_e080_s202; do
  python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-14s val_best=%6.3f @ep%-4d test_greedy=%6.3f wer=%.3f' % (
        '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-14s NO RESULT %s' % ('$A', e))
"
done
echo "recorded 40-epoch duration members: val 10.625/9.715/10.113, beam-5 mean 10.97 (controls 9.49)"
echo "if 80ep members beat 40ep, re-run the F ensemble with them -- F is currently 12.114 on test"
echo "DUR_LONG_DONE"
