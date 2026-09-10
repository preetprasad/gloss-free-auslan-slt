#!/bin/bash -l
# THE DECISIVE STITCHING CONTROL — proposed 2026-09-05, built 2026-09-09, never run before.
#
# QUESTION: WHY does PoseStitch help? Two mechanisms fit everything measured so far:
#   (A) VISUAL-LEXICAL GROUNDING -- the model learns "this pose means WORD" from synthetic signs.
#   (B) SEQUENCE STRUCTURE / REGULARISATION -- it learns "a pose sequence of this length with this
#       many signs maps to a sentence of this shape", plus generic regularisation from more data.
# They imply opposite next steps: (A) says keep improving stitcher fidelity; (B) says that work is
# spent. Evidence currently leans (B) but cannot separate them: the gain is beam-only (+0.71 beam vs
# +0.04 greedy), every scaling axis is flat (ratio/sentences/realisations all land 10.3-10.6),
# continuous->citation retrieval is 3.5% against 43.0% within-domain, and E0c measured synthetic
# within-class variability at 0.14x real. This job separates them directly instead of arguing.
#
# THE CONTROL: --scramble_glosses renders each sentence from RANDOMLY CHOSEN signs while holding the
# English target, sign COUNT, sampled per-sign DURATIONS and TRANSITION budgets bit-identical
# (verified locally: 200/200 on all four, 99.9% of glosses changed). Only WHICH sign is shown differs.
#
# PRE-REGISTERED READING, fixed before the run:
#   scrambled ~= joint             -> gain is (B); stitcher fidelity has no lexical headroom.
#   scrambled ~= real-only control -> gain is (A); grounding is real, fidelity work is justified.
#   in between                     -> report the split; both channels contribute.
# Both outcomes are informative; neither is a failure.
#
# WALLTIME: ALWAYS REQUEST 24h (Preet's standing instruction, 2026-09-09). Backfill does reward a
# shorter request -- in our own queue that day ens-dur (12h) was scheduled 12:10 and news-base (24h)
# 15:08 at equal priority, and comparable 6-arm jobs actually run 2:49-3:01 -- but a TIMEOUT destroys
# the whole run and every arm in it, while a later start only costs wall-clock. Do not "optimise"
# this down.
#
# DATA ROOT IS /group, NOT /scratch: everything read here is on durable storage, so the job is immune
# to the 21-day scratch purge whenever it happens to be scheduled.
#SBATCH --job-name=stitch-scram
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
OUT=$G/stitch_scram; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

# --- point the manifests at the durable clip copy ------------------------------------------------
SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in train val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  else
    echo "FAIL: cannot source manifest $f.tsv from $SRC"; exit 1
  fi
done
wc -l "$M"/train.tsv "$M"/val.tsv "$M"/test.tsv
# every referenced clip must exist, or we would silently train on a subset
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
SENTS="$B/stitch_meta/stitch_v2/sentences.txt"
[ -f "$SENTS" ] || SENTS=/scratch/pmc099/pprasad/stitch_v2/sentences.txt
[ -f "$SENTS" ] || { echo "FAIL: sentence pool missing"; exit 1; }
echo "sentence pool: $SENTS ($(wc -l < "$SENTS") lines)"
# count with find, not a glob: `ls dir/*.npz` silently returns nothing on the 38,580-file train
# directory because the expansion exceeds ARG_MAX, which would look identical to missing data.
for d in train valid; do
  n=$(find "$MM/dwpose_clips/$d" -name "*.npz" | wc -l)
  echo "mmwl $d: $n npz"
  [ "$n" -gt 1000 ] || { echo "FAIL: MM-WLAuslan $d clips missing or too few ($n)"; exit 1; }
done

# --- build the two synthetic corpora; ONLY gloss identity differs --------------------------------
build () {
  python tools/build_posestitch_corpus_v2.py \
    --labels "$MM/labels/Train.json" "$MM/labels/Valid.json" \
    --npz_dirs "$MM/dwpose_clips/train" "$MM/dwpose_clips/valid" \
    --sentences "$SENTS" --out_dir "$OUT/$1" \
    --min_cov 0.9 --min_signs 3 --realizations 1 --max_clips 6880 --seed 42 --sign_dur "${@:2}"
  cat "$M/train.tsv" "$OUT/$1/posestitch_v2_train.tsv" > "$OUT/joint_$1.tsv"
}
build real
build scram --scramble_glosses
wc -l "$OUT/joint_real.tsv" "$OUT/joint_scram.tsv"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
run () {
  echo "=== ARM $1 (seed $3) ==="
  python train_ctc.py --train_manifest "$2" $POSE $REC --seed "$3" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}
for S in 42 101; do
  run "scr_ctrl_s$S"  "$M/train.tsv"          "$S"
  run "scr_joint_s$S" "$OUT/joint_real.tsv"   "$S"
  run "scr_scram_s$S" "$OUT/joint_scram.tsv"  "$S"
done

echo "=== SUMMARY — where does SCRAMBLED land, between control and joint? ==="
for S in 42 101; do
  for A in scr_ctrl_s$S scr_joint_s$S scr_scram_s$S; do
    python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-16s val_best=%6.3f  test_greedy=%6.3f  wer=%.3f' % (
        '$A', d['best']['bleu4'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-16s NO RESULT %s' % ('$A', e))
"
  done
done
echo "STITCH_SCRAMBLE_DONE"
