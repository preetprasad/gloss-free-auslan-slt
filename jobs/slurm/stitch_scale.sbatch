#!/bin/bash -l
# DOES THE DURATION-STITCH LEVER STILL SATURATE AT ~0.5:1?
#
# The stitch lever was measured to saturate immediately at ~0.5:1 (2026-09-05): ratio 0.5:1, 1:1,
# k=1 and k=2 all landed in 10.3-10.6. That measurement was made on the OLD corpus, whose clips
# averaged 246 frames against a real 91.2 -- roughly 3x too long. The duration fix then nearly
# TRIPLED the lever (+0.56 -> +1.48 mean, replicated 3/3), and the project log's 2026-09-06 entry records
# the open question verbatim: "re-check whether the ratio/sentence-count saturation still holds now
# that the temporal statistics are correct -- the earlier flat result may have been an artefact of
# the wrong durations." THAT HAS NEVER BEEN RUN. It is the strongest untested lever on the board and
# it exploits MM-WLAuslan (45,010 clips, 14 exemplars/gloss) far harder than anything so far.
#
# It is newly answerable because constrained decoding broke the sentence bottleneck: 3,125 -> 20,608
# sentences at 100% content coverage by construction, enough for 1.66:1 at k=1. The old ratio test
# could not reach past ~1:1 and had to buy scale with repeated realisations.
#
# ARMS (seed 42 gets all five; seed 101 repeats the three load-bearing ones):
#   ctrl   real only                                  -- in-batch baseline
#   r055   +6,880 synthetic  = 0.55:1                 -- reproduces the +1.48 dose in-batch
#   r111   +13,760           = 1.11:1
#   r166   +20,608           = 1.66:1
#   cm166  real duplicated to r166's size             -- COMPUTE-MATCHED control
#
# cm166 is mandatory. At fixed 40 epochs a bigger corpus gets proportionally more optimisation steps,
# and that exact confound previously made a compute-matched arm drop 1.16 below its control. Without
# it, "more synthetic data helps" cannot be separated from "more updates helps".
#
# PRE-REGISTERED READING:
#   r111 > r055 AND r166 > r055 on validation  -> saturation WAS an artefact of the wrong durations;
#                                                 scaling is a live lever and the corpus should grow.
#   r166 ~= r055                               -> saturation is real and independent of temporal fidelity.
#   r166 > cm166                               -> the gain is the synthetic DATA, not the extra compute.
# Validation selects; test is reported alongside. Do NOT compare across batches to 10.97 or 11.80.
#SBATCH --job-name=stitch-scl2
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
OUT=$G/stitch_scale2; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

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
MM="$B/mmwl"
# count with find, not a glob: `ls dir/*.npz` returns NOTHING on the 38,580-file train directory
# because the expansion exceeds ARG_MAX, which looks identical to missing data.
for d in train valid; do
  n=$(find "$MM/dwpose_clips/$d" -name "*.npz" | wc -l)
  echo "mmwl $d: $n npz"
  [ "$n" -gt 1000 ] || { echo "FAIL: MM-WLAuslan $d clips missing or too few ($n)"; exit 1; }
done


# ---- sentence pool: the constrained 20,608, which is what makes 1.66:1 reachable ----
SENTS="$B/stitch_meta/stitch_v3/sentences_constrained.txt"
[ -f "$SENTS" ] || { echo "FAIL: constrained sentence pool missing"; exit 1; }
NSENT=$(wc -l < "$SENTS")
echo "sentence pool: $SENTS ($NSENT lines)"
[ "$NSENT" -ge 20000 ] || { echo "FAIL: pool has $NSENT sentences, need >=20000 for the 1.66:1 arm"; exit 1; }

COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --sentences $SENTS --min_cov 0.9 --min_signs 3 --realizations 1 --seed 42 \
  --morphology --no_paren_alias --sign_dur"

# Build ONCE at the largest size, then take prefixes. Three separate builds at different --max_clips
# would draw different sentences and different exemplars, confounding ratio with corpus composition;
# prefixes guarantee r055 is a strict subset of r111 is a strict subset of r166.
echo "=== building the 1.66:1 corpus once (prefixes give the smaller ratios) ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus" --max_clips 20608
FULL="$OUT/corpus/posestitch_v2_train.tsv"
NSYN=$(wc -l < "$FULL"); echo "synthetic clips built: $NSYN"
# a short corpus silently under-doses an arm; job 1193265 lost its interpretability exactly this way
[ "$NSYN" -ge 18000 ] || { echo "FAIL: only $NSYN clips built, expected ~20608"; exit 1; }

REAL="$M/train.tsv"; NREAL=$(wc -l < "$REAL")
for R in 6880 13760 20608; do
  head -n "$R" "$FULL" > "$OUT/syn_$R.tsv"
  cat "$REAL" "$OUT/syn_$R.tsv" > "$OUT/joint_$R.tsv"
  echo "ratio $(python -c "print(f'{$R/$NREAL:.2f}')"):1  -> $(wc -l < "$OUT/joint_$R.tsv") lines"
done
# compute-matched: real data repeated to the SAME total size as the 1.66:1 arm, so identical step count
# with zero new information.
TARGET=$(wc -l < "$OUT/joint_20608.tsv")
: > "$OUT/cm_train.tsv"
while [ "$(wc -l < "$OUT/cm_train.tsv")" -lt "$TARGET" ]; do cat "$REAL" >> "$OUT/cm_train.tsv"; done
head -n "$TARGET" "$OUT/cm_train.tsv" > "$OUT/cm_tmp" && mv "$OUT/cm_tmp" "$OUT/cm_train.tsv"
wc -l "$OUT/joint_20608.tsv" "$OUT/cm_train.tsv"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"

run () {
  echo "=== ARM $1 (seed $3) ==="
  python train_ctc.py --train_manifest "$2" $POSE $REC --seed "$3" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}

for S in ; do
  run "sc_ctrl_s$S"  "$REAL"                "$S"
  run "sc_r055_s$S"  "$OUT/joint_6880.tsv"  "$S"
  run "sc_r111_s$S"  "$OUT/joint_13760.tsv" "$S"
  run "sc_r166_s$S"  "$OUT/joint_20608.tsv" "$S"
  run "sc_cm166_s$S" "$OUT/cm_train.tsv"    "$S"
done
# r111 MUST be here. The first run (1196567) omitted it at seed 101 because the three "load-bearing"
# arms were chosen BEFORE the result was known -- and r111 then turned out to be the best arm at seed
# 42 (+1.63 vs r055's +1.01), leaving the winner unreplicated. Choose replication arms to cover the
# whole axis, not the subset you expect to matter.
for S in 101 202; do
  run "sc_ctrl_s$S" "$REAL"                "$S"
  run "sc_r055_s$S" "$OUT/joint_6880.tsv"  "$S"
  run "sc_r111_s$S" "$OUT/joint_13760.tsv" "$S"
  run "sc_r166_s$S" "$OUT/joint_20608.tsv" "$S"
done

echo "=== SUMMARY — does BLEU rise with ratio once durations are correct? ==="
for A in sc_ctrl_s42 sc_r055_s42 sc_r111_s42 sc_r166_s42 sc_cm166_s42 sc_ctrl_s101 sc_r055_s101 sc_r166_s101; do
  python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-14s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-14s NO RESULT %s' % ('$A', e))
"
done
echo "reference (do NOT compare across batches): duration-stitch 3-seed mean beam-5 10.97, controls 9.49"
echo "STITCH_SCALE_DONE"
