#!/bin/bash -l
# SUBWORD (BPE) ON NEWS — a language-side lever tested on the split where language IS the bottleneck.
#
# WHY THIS, AND WHY NOW (results/ALL_SPLITS_ANALYSIS_2026-09-09.md + NEWS_CONTEXT_AND_WER_FLOOR):
# The two Auslan-Daily splits have OPPOSITE bottlenecks. Measured on the corpora themselves:
#                       Communication      News
#   content words/ref        2.55          9.23
#   train vocabulary        3,058        12,409
#   verbatim test overlap   28.8%          5.4%
#   test CONTENT tokens never seen in train  0.54%   ->   4.63%   (8.6x)
#   train vocab seen exactly once           33.9%   ->   42.9%
#   hand speed vs citation form             9.86x   ->   1.10x   (News is VISUALLY easy)
# Communication is visually hard and linguistically easy; News is the reverse.
#
# EVERY language-side lever this project has tested -- BPE, LLM/mBART decoders, paraphrase
# augmentation, the regularisation sweep -- was run ON COMMUNICATION, i.e. on the split where language
# was NOT the constraint. BPE in particular scored neutral there (V1000 8.76, V2000 8.96 against a
# word-level control ~8.8), which is the expected result when only 0.54% of test content tokens are
# out of vocabulary: there is essentially no open-vocabulary problem to solve.
#
# On News there is: **4.63% of test content tokens never appear in training**, and a word-level model
# CANNOT emit them at all, at any level of visual accuracy. Subword units can. This is the cleanest
# available test of whether the earlier BPE negative was about the method or about the split.
#
# SCHEDULE: 150 epochs, matching news-long (1196181), because news-base showed News is UNDERTRAINED at
# 40 -- both arms peaked at epoch 40/40 with a significant upward trend (t=4.92 over the final 20
# epochs). Running BPE at 40 would repeat that mistake. max_src_frames=512 because it beat 256 on both
# validation (1.557 vs 1.400) and test (1.391 vs 1.267); 33% of News clips exceed 256 frames.
#
# The word-level arm here is an IN-BATCH control, so nothing needs comparing across jobs.
# PRE-REGISTERED: BPE must beat the in-batch word-level control on VALIDATION to be claimed.
#SBATCH --job-name=news-bpe
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
pip install -q jiwer sentencepiece 2>/dev/null || true
G=/group/pmc099/pprasad; B=$G/scratch_backup
cd "$G/code/phase2_baseline"
OUT=$G/news_bpe; M="$OUT/manifests"; RUNS="$OUT/runs"; SPM="$OUT/spm"; mkdir -p "$M" "$RUNS" "$SPM"

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

# SPM is trained on the NEWS training text only -- never val/test, or the vocabulary leaks.
cut -f2- "$M/news_train.tsv" > "$SPM/news_train_text.txt"
wc -l "$SPM/news_train_text.txt"
V=4000
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(input='$SPM/news_train_text.txt', model_prefix='$SPM/news_bpe_$V',
                               vocab_size=$V, model_type='bpe', character_coverage=1.0)
print('trained SPM vocab $V on News train text only')
"
test -f "$SPM/news_bpe_$V.model" || { echo "FAIL: SPM model not produced"; exit 1; }

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 150 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --seed 42 --max_src_frames 512 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
      --train_manifest $M/news_train.tsv --val_manifest $M/news_val.tsv --test_manifest $M/news_test.tsv"
run () {  # $1=tag, $2...=extra flags
  echo "=== ARM $1 ==="
  python train_ctc.py $POSE $BASE "${@:2}" --out_dir "$RUNS/$1"
  python tools/eval_beam.py --ckpt "$RUNS/$1/best.pt" --manifest "$M/news_test.tsv" --beam 5 --device cuda || true
}
run nb_word                                                    # in-batch word-level control
run nb_bpe4000 --text_tokenization sentencepiece --sentencepiece_model "$SPM/news_bpe_$V.model"

echo "=== SUMMARY — does subword help where 4.63% of test content tokens are OOV? ==="
for A in nb_word nb_bpe4000; do
  python -c "
import json
d=json.load(open('$RUNS/$A/progress.json')); t=d.get('test',{})
print('%-12s val_best=%6.3f @ep%-4d of %-4d test_greedy=%6.3f wer=%.3f' % (
    '$A', d['best']['bleu4'], d['best']['epoch'], d['epochs'],
    t.get('bleu4',float('nan')), t.get('wer',float('nan'))))
"
done
echo "reference: BPE on COMMUNICATION was neutral (V1000 8.76 / V2000 8.96 vs word-level ~8.8)"
echo "published News pose baselines: SL-Luong 2.81 | GASLT 2.72 | SL-Transf 2.11"
echo "NEWS_BPE_DONE"
