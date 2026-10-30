#!/bin/bash -l
# DOES THE NEWS SPLIT HELP COMMUNICATION AS *PRETRAINING*, WHERE JOINT TRAINING FAILED?
#
# 2026-08-27: joint-training on Comm+News (22,088 clips, the exact recipe that scored 10.27 on
# Comm-only) scored 8.89 on Comm test -- adding News HURT in-domain BLEU by ~1.4, and that closed
# "data scaling" for the Communication headline. But it closed ONE TRAINING REGIME: naive
# concatenation, where a single model must fit broadcast and conversational register simultaneously
# and the Comm validation selector only ever sees that compromised model.
#
# PRETRAIN -> FINETUNE HAS NEVER BEEN TRIED, and the VLP x stitch 2x2 (2026-09-05) showed the two
# stages behave differently: synthetic data in the FINETUNE stage helped robustly (+0.98 test /
# +0.76 val) while the same data in the PRETRAIN stage added nothing once finetuning was already
# synthetic. Stage identity matters here, so the joint-training negative does not transfer.
#
# WHY IT COULD WORK. News supplies 9,649 extra clips of CONTINUOUS signing with real sentence-level
# translation targets. The measured barrier for MM-WLAuslan transfer is that isolated citation form
# does not survive into continuous signing (spotting recall 0.05, cross-domain retrieval 3.5%);
# News has no such gap -- it IS continuous translation data. Its register differs (all-splits:
# News hand speed 1.10x citation vs Communication 9.86x), which is exactly why it must be a
# pretraining stage that is then specialised away, not a co-training partner.
#
# ARMS (seeds 42, 101; one shared News pretrain per seed):
#   ctrl      Comm only                                        -- in-batch baseline
#   npre      News pretrain -> Comm finetune                   -- does News transfer at all
#   npre_dur  News pretrain -> (Comm + duration-stitch) finetune -- BOTH datasets stacked
#
# npre_dur is the arm that exploits both datasets at once: MM-WLAuslan supplies stitched lexical
# coverage in the finetune data, News supplies continuous-signing structure in the encoder init.
# Neither has been combined with the other.
#
# PRE-REGISTERED: npre must beat ctrl on validation AND test in 2/2 seeds for News pretraining to be
# claimed. npre_dur vs npre isolates whether News adds anything ON TOP of the replicated +1.48 stitch
# lever -- if npre_dur ~= the known duration-stitch level, News contributes nothing beyond it.
# Vocabularies differ (News 12,409 vs Comm 3,058); --resume_checkpoint shape-filters, so the encoder
# and the vocab-independent decoder body transfer and only the vocab-tied layers re-initialise.
#SBATCH --job-name=news-xfer
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
OUT=$G/news_xfer; M="$OUT/manifests"; RUNS="$OUT/runs"; mkdir -p "$M" "$RUNS"

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


# ---- News manifests, rewritten onto /group like the Communication ones ----
NM="$OUT/news_manifests"; mkdir -p "$NM"
for f in train val; do
  s="$B/dwpose/manifests/news_$f.tsv"
  [ -f "$s" ] || { echo "FAIL: missing $s"; exit 1; }
  sed "s#/scratch/pmc099/pprasad/news/dwpose_clips#$B/news/dwpose_clips#g" "$s" > "$NM/$f.tsv"
done
wc -l "$NM/train.tsv" "$NM/val.tsv"
python - "$NM" <<'PYCHK'
import os, sys
root = sys.argv[1]; bad = tot = 0
for f in ("train", "val"):
    for line in open(os.path.join(root, f + ".tsv")):
        if "\t" not in line: continue
        tot += 1
        if not os.path.exists(line.split("\t", 1)[0]): bad += 1
print(f"news clip check: {tot} referenced, {bad} missing")
sys.exit(1 if bad else 0)
PYCHK

# ---- duration-stitch corpus for the npre_dur arm, rebuilt deterministically (seed 42) ----
# The original corpus_dur clips live only on /scratch and are purge-eligible, so regenerate rather
# than depend on them. sentences_all.txt is coverage-filtered: 6,880 in -> 6,880 out, matching the
# corpus behind the replicated +1.48 exactly.
SENTS="$B/stitch_meta/stitch_v2/sentences_all.txt"
NSENT=$(wc -l < "$SENTS"); echo "sentence pool: $SENTS ($NSENT lines)"
[ "$NSENT" -ge 6880 ] || { echo "FAIL: pool too small ($NSENT) to fill the 6880 budget"; exit 1; }
python tools/build_posestitch_corpus_v2.py \
  --labels "$MM/labels/Train.json" "$MM/labels/Valid.json" \
  --npz_dirs "$MM/dwpose_clips/train" "$MM/dwpose_clips/valid" \
  --sentences "$SENTS" --out_dir "$OUT/corpus_dur" \
  --min_cov 0.9 --min_signs 3 --realizations 1 --max_clips 6880 --seed 42 --sign_dur
NSYN=$(wc -l < "$OUT/corpus_dur/posestitch_v2_train.tsv"); echo "synthetic clips: $NSYN"
[ "$NSYN" -ge 6000 ] || { echo "FAIL: only $NSYN clips built, expected 6880"; exit 1; }
cat "$M/train.tsv" "$OUT/corpus_dur/posestitch_v2_train.tsv" > "$OUT/joint_dur.tsv"
wc -l "$M/train.tsv" "$OUT/joint_dur.tsv"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
COMMVAL="--val_manifest $M/val.tsv --test_manifest $M/test.tsv"

for S in 42 101; do
  # (1) News pretrain, selected on NEWS validation -- shared by both transfer arms at this seed
  echo "=== PRETRAIN news_s$S ==="
  python train_ctc.py --train_manifest "$NM/train.tsv" --val_manifest "$NM/val.tsv" \
    $POSE $BASE --seed "$S" --out_dir "$RUNS/news_pre_s$S"
  PRE="$RUNS/news_pre_s$S/best.pt"
  [ -f "$PRE" ] || { echo "FAIL: no pretrain checkpoint at $PRE"; exit 1; }

  for A in "nx_ctrl_s$S:$M/train.tsv:" "nx_npre_s$S:$M/train.tsv:$PRE" "nx_npre_dur_s$S:$OUT/joint_dur.tsv:$PRE"; do
    tag=${A%%:*}; rest=${A#*:}; man=${rest%%:*}; init=${rest#*:}
    echo "=== ARM $tag (seed $S)${init:+ warm-started from News} ==="
    python train_ctc.py --train_manifest "$man" $COMMVAL $POSE $BASE --seed "$S" \
      ${init:+--resume_checkpoint "$init"} --out_dir "$RUNS/$tag"
    python tools/eval_beam.py --ckpt "$RUNS/$tag/best.pt" --manifest "$M/test.tsv" \
      --beam 5 --device cuda || true
  done
done

echo "=== SUMMARY — rule: npre must beat ctrl on val AND test, 2/2 seeds ==="
for A in nx_ctrl_s42 nx_npre_s42 nx_npre_dur_s42 nx_ctrl_s101 nx_npre_s101 nx_npre_dur_s101; do
  python -c "
import json
try:
    d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
    print('%-17s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$A', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
except Exception as e: print('%-17s NO RESULT %s' % ('$A', e))
"
done
echo "context (do NOT compare across batches): naive Comm+News JOINT training scored 8.89 on Comm test"
echo "                                          vs 10.27 Comm-only -- that is the regime this replaces."
echo "NEWS_XFER_DONE"
