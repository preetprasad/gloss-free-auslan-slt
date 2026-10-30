#!/bin/bash -l
# WHY DID BPE NOT HELP NEWS? A mechanism check on existing checkpoints -- not a new experiment.
#
# `news-bpe` (1196431) came back neutral: BPE-4000 vs word-level = +0.107 val / +0.089 greedy / -0.24
# beam-5, all smaller than the gap between two runs of the IDENTICAL word-level config (0.236 val /
# 0.181 greedy). The premise was that 4.63% of News test content tokens never occur in training, so a
# word-level model CANNOT emit them and subword units can. That premise is logically sound. What was
# never measured is whether the mechanism ENGAGED: did the BPE model ever emit an out-of-vocabulary word,
# and was it ever the RIGHT one?
#
# Pre-registered readings:
#   BPE emits ~0 correct OOV words  -> the open-vocabulary mechanism never engaged. Being ABLE to spell a
#                                      rare word is irrelevant if the sign cannot be identified visually;
#                                      identification binds even on the linguistically-hard split.
#   BPE emits OOV words, mostly wrong -> the mechanism engaged but produces invented/misspelled words.
#   BPE emits correct OOV words      -> it works; the BLEU effect is just too small to resolve.
#
# Needs eval_beam.py with hypotheses in --output_json (patched 2026-09-10; the analysis FAILS LOUDLY if
# the keys are absent). Re-decoding nb_word also validates the harness: it must reproduce the recorded
# beam-5 1.92, and nb_bpe4000 must reproduce 1.68.
#SBATCH --job-name=news-oov
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
OUT=$G/news_oov; mkdir -p "$OUT"
R=$G/news_bpe/runs; M=$G/news_bpe/manifests

sed "s#/scratch/pmc099/pprasad/news/dwpose_clips#$B/news/dwpose_clips#g" "$M/news_test.tsv" > "$OUT/news_test.tsv"
python - "$OUT/news_test.tsv" <<'PYCHK'
import os, sys
rows = [l.split("\t", 1)[0] for l in open(sys.argv[1]) if "\t" in l]
miss = sum(1 for p in rows if not os.path.exists(p))
print(f"news test clips: {len(rows)} referenced, {miss} missing")
sys.exit(1 if (miss or len(rows) != 699) else 0)
PYCHK

for A in nb_word nb_bpe4000; do
  [ -f "$R/$A/best.pt" ] || { echo "FAIL: missing $R/$A/best.pt"; exit 1; }
  echo "=== DECODE $A ==="
  python tools/eval_beam.py --ckpt "$R/$A/best.pt" --manifest "$OUT/news_test.tsv" \
    --beam 5 --device cuda --output_json "$OUT/$A.json"
done

echo "=== OOV MECHANISM ANALYSIS ==="
python - "$OUT" "$M/news_train.tsv" <<'PYAN'
import json, sys
sys.path.insert(0, ".")
from tools.stitch_lexicon import content_words, words
from slt.metrics import corpus_bleu4, detokenize
out, trainman = sys.argv[1], sys.argv[2]
train_vocab = set()
for line in open(trainman):
    if "\t" in line:
        train_vocab.update(words(line.split("\t", 1)[1]))
print(f"word types in News TRAIN text: {len(train_vocab)}\n")
print(f"{'model':11} {'BLEU-4':>7} {'refOOV%':>8} {'hypOOV%':>8} {'OOV emitted':>12} "
      f"{'OOV correct':>12} {'OOV recall':>11} {'IV recall':>10}")
for A in ("nb_word", "nb_bpe4000"):
    d = json.load(open(f"{out}/{A}.json"))
    if "beam_hyps" not in d or "refs" not in d:
        raise SystemExit(f"FATAL: {A}.json lacks beam_hyps/refs -- eval_beam.py patch not applied")
    hyps, refs = d["beam_hyps"], d["refs"]
    bleu = corpus_bleu4([detokenize(x) for x in hyps], [detokenize(x) for x in refs], tokenize="13a")
    hyp_w = hyp_oov = oov_ok = oov_rec = oov_tot = iv_rec = iv_tot = 0
    for h, r in zip(hyps, refs):
        hw = words(h); hset = set(hw); rset = set(words(r))
        hyp_w += len(hw)
        for w in hw:
            if w not in train_vocab:
                hyp_oov += 1
                oov_ok += (w in rset)
        for w in content_words(r):
            if w in train_vocab:
                iv_tot += 1; iv_rec += (w in hset)
            else:
                oov_tot += 1; oov_rec += (w in hset)
    print(f"{A:11} {bleu:7.2f} {100*oov_tot/max(1,oov_tot+iv_tot):7.2f}% {100*hyp_oov/max(1,hyp_w):7.2f}% "
          f"{hyp_oov:12d} {oov_ok:12d} {100*oov_rec/max(1,oov_tot):10.2f}% {100*iv_rec/max(1,iv_tot):9.2f}%")
print("\nrefOOV% should reproduce the recorded 4.63%. nb_word must reproduce beam-5 1.92, nb_bpe4000 1.68.")
print("'OOV correct' = emitted words absent from train text that DO appear in that clip's reference.")
PYAN
echo "NEWS_OOV_DONE"
