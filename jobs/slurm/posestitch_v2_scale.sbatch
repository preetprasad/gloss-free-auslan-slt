#!/bin/bash -l
# PoseStitch v2 — CORPUS SCALING. Does MORE synthetic data help, or is 0.5:1 already the plateau?
#
# The 2026-09-04b run only reached 6,250 synthetic clips (0.50:1 against the 12,439 real) because
# Stage A acceptance ran ~11/round against a 320-round cap -- so the lever was never saturated.
# This extends the corpus to roughly 1:1 and trains ONE joint arm at seed 42, which is directly
# comparable to that run's seed-42 pair (control 9.88 / joint 10.59 beam-5).
#
# Generation uses a DIFFERENT --seed so the new sentences are not a re-draw of the old ones, and
# the concatenated sentence list is de-duplicated before stitching. The original corpus is left
# untouched (the seed-replication job reads it concurrently).
#SBATCH --job-name=stitch-scale
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
LAB="--labels $MM/labels/Train.json $MM/labels/Valid.json"
DIRS="--npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid"

echo "=== STAGE A2: generate ADDITIONAL novel sentences (fresh generation seed) ==="
python tools/gen_stitch_sentences.py $LAB --ad_manifest "$DE/manifests/train.tsv" \
  --out "$ST/sentences_ext.txt" --device cuda --target 4000 --min_cov 0.9 \
  --max_rounds 500 --seed 202
# merge + de-duplicate (order-preserving) so repeated sentences don't masquerade as new coverage
cat "$ST/sentences.txt" "$ST/sentences_ext.txt" | awk '!seen[$0]++' > "$ST/sentences_all.txt"
echo "sentences: was $(wc -l < "$ST/sentences.txt") | new $(wc -l < "$ST/sentences_ext.txt") | merged-unique $(wc -l < "$ST/sentences_all.txt")"

echo "=== STAGE B2: stitch the extended sentence set (target ~1:1 with 12,439 real) ==="
# Stitch seed stays 42: the merged file keeps the original sentences first and in order, so their
# exemplar draws reproduce exactly and the ONLY difference from the 6,250-clip corpus is the added data.
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$ST/sentences_all.txt" \
  --out_dir "$ST/corpus_big" --min_cov 0.9 --min_signs 3 --realizations 2 \
  --max_clips 12500 --seed 42
SYN_BIG="$ST/corpus_big/posestitch_v2_train.tsv"
cat "$DE/manifests/train.tsv" "$SYN_BIG" > "$ST/joint_big_train.tsv"
echo "synthetic: $(wc -l < "$SYN_BIG") clips (was 6250) | joint_big: $(wc -l < "$ST/joint_big_train.tsv")"

echo "=== ARM: joint_big at seed 42 (vs the seed-42 pair: control 9.88 / joint@0.5:1 10.59) ==="
python train_ctc.py --train_manifest "$ST/joint_big_train.tsv" \
  --val_manifest "$DE/manifests/val.tsv" --test_manifest "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$ST/runs/sv2_joint_big_s42"
python tools/eval_beam.py --ckpt "$ST/runs/sv2_joint_big_s42/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
cp -r "$ST/runs/sv2_joint_big_s42" "$G/comm_dwpose_sv2_joint_big" 2>/dev/null || true

echo "=== SUMMARY ==="
python -c "
import json
d=json.load(open('$ST/runs/sv2_joint_big_s42/progress.json')); t=d.get('test',{})
print('sv2_joint_big_s42 val_best=%.3f@ep%d test_greedy=%.3f wer=%.3f' % (
    d['best']['bleu4'], d['best']['epoch'], t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
print('seed-42 references -- control: val 9.675 greedy 9.65 beam 9.88 | joint@0.5:1: val 9.935 greedy 9.69 beam 10.59')
"
echo "STITCH_SCALE_DONE"
