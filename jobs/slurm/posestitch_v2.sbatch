#!/bin/bash -l
# PoseStitch v2 — properly exploiting the 45,010-clip MM-WLAuslan dictionary.
#
# WHY A RE-RUN: v1 scored a neutral 9.66, but it was a weak test of stitching:
#   * used 6,430 of 45,010 clips  -> 2 exemplars/gloss (the "3 realizations" were near-identical)
#   * single-word gloss lookup    -> 612 of 3,215 glosses were structurally unmatchable
#   * coverage threshold 0.6      -> up to 40% of content words had no sign in the pose
#   * re-realised REAL sentences  -> the 2026 stitching ablation shows this exact config HURTS
#                                    (clip reassignment w/o novel sentences: 20.04 vs 21.38)
#   * disjoint synthetic pretrain -> that paper's synthetic-VLP arm COLLAPSED (11.38 vs 21.38)
# v2 fixes all five: all 45k clips (14 exemplars/gloss), phrase-aware matching (coverage
# 41.9% -> 51.1% measured), cov>=0.9, LLM-generated NOVEL sentences, and JOINT training.
#
# Arms: (0) real-only control  (1) compute-matched control (real duplicated)  (2) real + synthetic.
# Arm 1 exists so a gain cannot be explained by "twice the optimisation steps".
# Selection on the REAL val split; the REAL test split is unchanged.
#SBATCH --job-name=stitch-v2
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
pip install -q "transformers==4.46.3" "accelerate==1.1.1" sentencepiece jiwer 2>/dev/null || true
export DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
MM=/scratch/pmc099/pprasad/mmwl; export ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"
mkdir -p "$ST"
LAB="--labels $MM/labels/Train.json $MM/labels/Valid.json"
DIRS="--npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid"

echo "=== SMOKE GATE: generate 40 sentences, stitch them, load through the real pipeline ==="
python tools/gen_stitch_sentences.py $LAB --ad_manifest "$DE/manifests/train.tsv" \
  --out "$ST/sents_smoke.txt" --device cuda --target 40 --max_rounds 6
test -s "$ST/sents_smoke.txt" || { echo "SMOKE FAIL: no sentences generated"; exit 1; }
head -5 "$ST/sents_smoke.txt"
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$ST/sents_smoke.txt" \
  --out_dir "$ST/smoke" --realizations 1
python - <<'PY'
import sys; sys.path.insert(0, ".")
import os, torch
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
DE = os.environ.get("DE", "/scratch/pmc099/pprasad/dwpose")
ST = os.environ.get("ST", "/scratch/pmc099/pprasad/stitch_v2")
cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame")
txt = [l.rstrip("\n").split("\t")[1] for l in open(f"{DE}/manifests/train.tsv") if l.strip()]
vocab = Vocab.build(txt, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
known = set(vocab.itos)
man = f"{ST}/smoke/posestitch_v2_train.tsv"
ds = AuslanDailyDataset(man, vocab, cfg, "")
src, pad, tgt, texts = collate([ds[i] for i in range(min(4, len(ds)))])
assert src.shape[-1] == cfg.feature_dim() and bool(torch.isfinite(src).all()), src.shape
oov = [w for _, t in ds.samples for w in t.lower().split() if w not in known]
assert not oov, f"synthetic text leaks OOV words (would change the output space): {oov[:10]}"
print("SMOKE_OK stitched batch", tuple(src.shape), "| vocab-safe | clips:", len(ds))
PY

echo "=== STAGE A: generate novel signable sentences (target 8000) ==="
python tools/gen_stitch_sentences.py $LAB --ad_manifest "$DE/manifests/train.tsv" \
  --out "$ST/sentences.txt" --device cuda --target 8000 --min_cov 0.9 --max_rounds 320
wc -l "$ST/sentences.txt"

echo "=== STAGE B: stitch into synthetic continuous clips (~1:1 with the 12,439 real) ==="
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$ST/sentences.txt" \
  --out_dir "$ST/corpus" --min_cov 0.9 --min_signs 3 --realizations 2 --max_clips 12500
SYN="$ST/corpus/posestitch_v2_train.tsv"
wc -l "$SYN"

echo "=== build joint + compute-matched manifests ==="
cat "$DE/manifests/train.tsv" "$SYN" > "$ST/joint_train.tsv"
cat "$DE/manifests/train.tsv" "$DE/manifests/train.tsv" > "$ST/doubled_train.tsv"
wc -l "$DE/manifests/train.tsv" "$ST/joint_train.tsv" "$ST/doubled_train.tsv"

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

run_arm () {   # $1 = arm name, $2 = train manifest
  echo "=== ARM: $1 ($2) ==="
  python train_ctc.py --train_manifest "$2" $VAL $POSE $REC --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
  cp -r "$ST/runs/$1" "$G/comm_dwpose_$1" 2>/dev/null || true
}
run_arm stitchv2_control  "$DE/manifests/train.tsv"
run_arm stitchv2_doubled  "$ST/doubled_train.tsv"
run_arm stitchv2_joint    "$ST/joint_train.tsv"

echo "=== SUMMARY (joint must beat BOTH controls to count) ==="
for R in stitchv2_control stitchv2_doubled stitchv2_joint; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('$R', 'val_best=%.3f@ep%d' % (d['best']['bleu4'], d['best']['epoch']),
          'test_bleu4=%.3f' % t.get('bleu4', float('nan')), 'wer=%.3f' % t.get('wer', float('nan')))
except Exception as e:
    print('$R', 'NO RESULT', e)
"
done
echo "POSESTITCH_V2_DONE"
