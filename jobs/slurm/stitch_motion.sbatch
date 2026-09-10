#!/bin/bash -l
# E0b GATE — does the motion-corrected synthetic corpus beat the current one on BLEU?
#
# E0/E0b (results/E0_MOTION_FIDELITY_2026-09-09.md) measured, WITHOUT training anything, that the
# shipped `--sign_dur` corpus is still clearly separable from real Auslan-Daily motion, and that two
# independent defects cause it:
#   * --trim_frac 0.15 cuts the movement ONSET/OFFSET (the fastest part of a citation sign), which
#     CREATES join discontinuity: join/real-interior 4.85x -> 3.43x when the trim is removed.
#   * uniform resampling fixes total DURATION but preserves the slow citation VELOCITY PROFILE:
#     separability 0.889 -> 0.704 and pause density 36.6% -> 21.6% with motion-aware resampling.
# Combined: separability 0.872 -> 0.704 (empirical chance 0.684), velocity 0.0288 -> 0.0443 vs a real
# 0.0547. Duration is UNCHANGED (107 frames matched, real 104), so this does not disturb the lever
# that already produced +1.48.
#
# A BETTER MOTION STATISTIC IS NOT A RESULT. This job is the only thing that can promote it, and the
# rule is pre-registered: the joint arm must beat its OWN same-seed control on BOTH validation and
# test, in all three pairs, exactly as the duration lever had to.
#
# NOTE: the corpus differs from the +1.48 one in TWO ways (trim and resampler), so the in-batch
# control below is mandatory -- do not compare to the historical 10.97 across batches.
#SBATCH --job-name=stitch-mot3
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
OUT=$G/stitch_motion3; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

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

# ---- build the motion-corrected corpus (same sentence pool + budget + seed as the current one) ----
# MUST be sentences_all.txt: the +1.48 duration corpus (stitch_temporal.sbatch:36) used
# SENT="$ST/sentences_all.txt"; BUDGET=6880. Run 1193265 fell back to stitch_v2/sentences.txt,
# which holds only 3,125 lines, so --realizations 1 --max_clips 6880 could yield at most 3,125
# and produced 2,660 clips = 0.21:1 against the duration corpus's 6,880 = 0.55:1. That confounded
# the transform with a 2.6x smaller corpus, in the regime BELOW the measured ~0.5:1 saturation
# point where corpus size still moves BLEU. Pin the pool and assert it can fill the budget.
SENTS="$B/stitch_meta/stitch_v2/sentences_all.txt"
[ -f "$SENTS" ] || SENTS=/scratch/pmc099/pprasad/stitch_v2/sentences_all.txt
if [ ! -f "$SENTS" ]; then echo "FAIL: cannot find the Stage-A sentence pool"; exit 1; fi
NSENT=$(wc -l < "$SENTS")
echo "sentence pool: $SENTS ($NSENT lines)"
if [ "$NSENT" -lt 6880 ]; then
  echo "FAIL: pool has $NSENT sentences but the budget is 6880 at --realizations 1;"
  echo "      a short pool silently under-doses the corpus and confounds the transform (see 1193265)."
  exit 1
fi

COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --sentences $SENTS --min_cov 0.9 --min_signs 3 --realizations 1 --max_clips 6880 --seed 42 \
  --morphology --no_paren_alias"

# ARM B corpus: --sign_dur ONLY. This is the transform behind the replicated +1.48, rebuilt HERE with
# this job's pool and lexicon flags so it is an IN-BATCH reference. Run 1193265 lacked it, so its
# motion arm could only be compared to real-only or across batches to a corpus that differed in
# THREE ways (transform, morphology/alias, and — fatally — 2660 vs 6880 clips).
echo "=== corpus B: --sign_dur only (in-batch duration reference) ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus_dur" --sign_dur

# ARM C corpus: E0b's winner — drop the trim (it was CREATING the join defect, 4.85x -> 3.43x) and
# resample on accumulated motion (separability 0.889 -> 0.704, pause 36.6% -> 21.6%).
echo "=== corpus C: --sign_dur + --motion_resample, trim 0.0 (E0b winner) ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus_mot" \
  --sign_dur --motion_resample --trim_frac 0.0 --audit_jsonl "$OUT/corpus_mot/audit.jsonl"

python tools/audit_posestitch_motion.py \
  --audit_jsonl "$OUT/corpus_mot/audit.jsonl" --real_train_manifest "$M/train.tsv" \
  --out_json "$OUT/audit_production.json" || echo "(audit non-fatal)"

# GUARD, corrected after run 1196565 failed here. That run built 5,903 clips, not 6,880, because
# `sentences_all.txt` is coverage-filtered under the DEFAULT lexicon while this job sets
# --morphology --no_paren_alias, which changes the lexicon and drops 977 sentences below cov>=0.9.
# The original `-ge 6000` check was wrong for its own purpose: what this experiment needs is that the
# two arms are matched to EACH OTHER (B vs C is the only comparison that tests E0b), not that they
# match the historical 6,880-clip corpus. 5,903 is 0.47:1, comfortably in the dosed regime; the
# failure mode actually worth catching is a SILENT DIFFERENCE BETWEEN THE ARMS, so assert that.
NDUR=0; NMOT=0
for T in dur mot; do
  n=$(wc -l < "$OUT/corpus_$T/posestitch_v2_train.tsv")
  echo "corpus_$T: $n synthetic clips (ratio $(python -c "print(f'{$n/$(wc -l < "$M/train.tsv"):.2f}')"):1)"
  [ "$n" -ge 5500 ] || { echo "FAIL: corpus_$T has only $n clips -- under-dosed, below the ~0.5:1 regime"; exit 1; }
  [ "$T" = dur ] && NDUR=$n || NMOT=$n
  cat "$M/train.tsv" "$OUT/corpus_$T/posestitch_v2_train.tsv" > "$OUT/joint_$T.tsv"
done
# the arms must be the same size or the transform is confounded with corpus size, which is exactly
# what invalidated job 1193265
D=$(( NDUR > NMOT ? NDUR - NMOT : NMOT - NDUR ))
[ "$D" -le $(( NDUR / 50 )) ] || { echo "FAIL: corpora differ by $D clips ($NDUR vs $NMOT) -- transform would be confounded with size"; exit 1; }
echo "arms matched: dur=$NDUR mot=$NMOT (difference $D)"
wc -l "$M/train.tsv" "$OUT/joint_dur.tsv" "$OUT/joint_mot.tsv"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"

run () {  # $1=tag  $2=train manifest  $3=seed
  echo "=== ARM $1 (seed $3) ==="
  python train_ctc.py --train_manifest "$2" $POSE $REC --seed "$3" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda || true
}

# THREE arms per seed so the transform is isolated in-batch:
#   A ctrl : real only                      -> does synthetic help at all
#   B dur  : + --sign_dur corpus            -> reproduces the replicated +1.48 lever here
#   C mot  : + motion-corrected corpus      -> C vs B is the ONLY comparison that tests E0b
# Pre-registered: C must beat B on validation AND test in 3/3 seeds for motion-resampling to be
# adopted. C > A but C <= B means the gain is the duration fix already on the books, not the motion fix.
for S in 42 101 202; do
  run "mot_ctrl_s$S" "$M/train.tsv"     "$S"
  run "mot_dur_s$S"  "$OUT/joint_dur.tsv" "$S"
  run "mot_mot_s$S"  "$OUT/joint_mot.tsv" "$S"
done

echo "=== SUMMARY — rule: C(mot) must beat B(dur) on val AND test, 3/3. C>A but C<=B = duration fix only ==="
for S in 42 101 202; do
  for A in mot_ctrl_s$S mot_dur_s$S mot_mot_s$S; do
    python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-16s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-16s NO RESULT %s' % ('$A', e))
"
  done
done
echo "compare to: duration-stitch 3-seed mean beam-5 10.97 (controls 9.49) -- but ONLY via the in-batch controls above"
echo "STITCH_MOTION_DONE"
