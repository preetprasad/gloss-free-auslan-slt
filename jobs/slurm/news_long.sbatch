#!/bin/bash -l
# NEWS, TRAINED TO CONVERGENCE — follow-up to news-base (1191190), which was UNDERTRAINED.
#
# WHAT 1191190 SHOWED. Validation-selected arm news_512: val 1.557, test beam-5 1.39, WER 1.095
# (news_256: val 1.400, test 1.26). Against published News pose baselines SL-Luong 2.81 / GASLT 2.72 /
# SL-Transf 2.11, that is roughly HALF the weakest published number.
#
# BUT BOTH ARMS PEAKED AT EPOCH 40 -- THE LAST EPOCH -- AND WERE STILL RISING:
#   news_512 val: ep35 1.458, ep36 1.401, ep37 1.415, ep38 1.430, ep39 1.496, ep40 1.557
#   news_256 val: ... ep38 1.342, ep39 1.276, ep40 1.400
# The 40-epoch budget was tuned on COMMUNICATION, where validation peaks at epoch 8-24 and then
# overfits. News has 17.0-word targets against Communication's 7.1 and a 12,409-word vocabulary
# against 3,058, so the same budget overfits one split and undertrains the other. **1.39 is a floor
# produced by a stopped run, not a converged result**, and it must not be quoted as News performance.
#
# The 512-frame arm also beat 256 on BOTH validation (1.557 vs 1.400) and test (1.391 vs 1.267),
# confirming that the max_src_frames=256 default -- a Communication-tuned choice that truncates 33%
# of News clips -- was a real constraint. Both arms are carried forward at the longer schedule so the
# frame effect is re-measured at convergence rather than assumed to persist.
#
# EARLY STOPPING IS DELIBERATELY NOT ADDED: the point is to observe where validation actually turns
# over. Selection remains validation-only; the test number reported is the val-selected checkpoint's.
#SBATCH --job-name=news-long
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
OUT=$G/news_long; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

# News clips are on scratch AND in the durable backup; prefer the backup so a purge cannot bite.
SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in news_train news_val news_test; do
  [ -f "$SRC/$f.tsv" ] || { echo "FAIL: missing $SRC/$f.tsv"; exit 1; }
  sed "s#/scratch/pmc099/pprasad/news/dwpose_clips#$B/news/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
done
python - "$M" <<'PYCHK'
import os, sys
root = sys.argv[1]
bad = tot = 0
for f in ("news_train", "news_val", "news_test"):
    for line in open(os.path.join(root, f + ".tsv")):
        if "\t" not in line:
            continue
        tot += 1
        if not os.path.exists(line.split("\t", 1)[0]):
            bad += 1
print(f"manifest clip check: {tot} referenced, {bad} missing")
sys.exit(1 if bad else 0)
PYCHK
wc -l "$M"/news_train.tsv "$M"/news_val.tsv "$M"/news_test.tsv

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 150 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --train_manifest $M/news_train.tsv --val_manifest $M/news_val.tsv --test_manifest $M/news_test.tsv"
run () {
  echo "=== ARM $1 (max_src_frames=$2, 150 epochs) ==="
  python train_ctc.py $POSE $REC --max_src_frames "$2" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/news_test.tsv" --beam 5 --device cuda || true
}
run news_long_512 512
run news_long_256 256

echo "=== SUMMARY — did validation turn over, and where? ==="
for A in news_long_512 news_long_256; do
  python -c "
import json
d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
h=d.get('history') or []
vs=[(e.get('epoch'), e.get('val_bleu4') or e.get('bleu4')) for e in h if isinstance(e,dict)]
vs=[(a,b) for a,b in vs if b is not None]
print('%-16s val_best=%6.3f @ep%-4d test_greedy=%6.3f wer=%.3f' % (
    '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
if vs: print('    last 10:', [(a,round(b,2)) for a,b in vs[-10:]])
"
done
echo "40-epoch reference (job 1191190): news_512 val 1.557 @ep40 (STILL RISING) test 1.39 | news_256 val 1.400 @ep40 test 1.26"
echo "published News pose baselines: SL-Luong 2.81 | GASLT 2.72 | SL-Transf 2.11"
echo "NEWS_LONG_DONE"
