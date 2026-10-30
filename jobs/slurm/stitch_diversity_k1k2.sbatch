#!/bin/bash -l
# THE DECISIVE A-vs-B ARM: is the PoseStitch gain LINGUISTIC DIVERSITY or just CLIP VOLUME?
#
# PoseStitch v2 joint training now REPLICATES (+0.71/+0.75/+0.23 beam-5 over three paired seeds,
# mean +0.56). But we still do not know WHICH axis produced it, and the two published sources
# disagree with what we actually ran:
#   * The PHOENIX +2.92 winning configuration used 7,000 sentences x **1 realisation each**.
#     Its ablation with clip reassignment but NO novel sentences scored 20.04 vs a 21.38
#     baseline -- i.e. visual variation ALONE was actively harmful.
#   * Data Diversification (NeurIPS 2020), the only published curve for "k renderings of the
#     same item", saturates at k=3 (29.8/30.1/30.7/30.7/30.7/30.6 for k=1..6).
#   * We ran 3,125 sentences x k=2 -- half their linguistic diversity, double the repetition.
# And the 1:1 scaling arm (job 1173726) raised validation to a record 10.901 while test FELL to
# 10.32, which is exactly what you would expect if extra CLIPS are not the active ingredient.
#
# THIS JOB HOLDS CLIP COUNT AND SYNTHETIC:REAL RATIO FIXED and moves only sentence diversity:
#   ARM P  6,880 unique sentences x 1 realisation  = 6,880 clips   <- max linguistic diversity
#   ARM Q  3,440 unique sentences x 2 realisations = 6,880 clips   <- half the diversity
# Q's sentences are a strict SUBSET of P's (the stitcher consumes the file in order and stops at
# --max_clips), so the two corpora differ in nothing except how the same clip budget is spent.
# Both are 0.553:1 against the 12,439 real clips. Whatever separates them IS the diversity effect.
#
# No published sign-language work ablates realisations-per-synthetic-sentence, so this is a
# genuine contribution either way: if P > Q the PHOENIX linguistic-diversity story replicates on
# Auslan; if P == Q the gain is clip volume and the whole diversity framing is wrong.
#SBATCH --job-name=stitch-div
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
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
MM=/scratch/pmc099/pprasad/mmwl; ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"
LAB="--labels $MM/labels/Train.json $MM/labels/Valid.json"
DIRS="--npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid"
SENT="$ST/sentences_all.txt"          # 6,880 unique, built by job 1173726
test -s "$SENT" || { echo "FAIL: merged sentence file missing at $SENT"; exit 1; }
N=$(wc -l < "$SENT"); echo "sentence pool: $N unique"

# Same clip budget for both arms. P spends it on distinct sentences, Q on repeated ones.
BUDGET=6880
echo "=== build ARM P corpus: $BUDGET sentences x 1 realisation ==="
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$SENT" \
  --out_dir "$ST/corpus_k1" --min_cov 0.9 --min_signs 3 \
  --realizations 1 --max_clips $BUDGET --seed 42
echo "=== build ARM Q corpus: $((BUDGET/2)) sentences x 2 realisations ==="
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$SENT" \
  --out_dir "$ST/corpus_k2" --min_cov 0.9 --min_signs 3 \
  --realizations 2 --max_clips $BUDGET --seed 42

P="$ST/corpus_k1/posestitch_v2_train.tsv"; Q="$ST/corpus_k2/posestitch_v2_train.tsv"
python - <<PY
import collections
for tag, p in [("P k=1", "$P"), ("Q k=2", "$Q")]:
    rows = [l.rstrip("\n").split("\t", 1)[1] for l in open(p) if "\t" in l]
    c = collections.Counter(rows)
    print("%-7s clips=%d  unique sentences=%d  mean realisations/sentence=%.2f"
          % (tag, len(rows), len(c), len(rows) / max(1, len(c))))
PY
cat "$DE/manifests/train.tsv" "$P" > "$ST/joint_k1.tsv"
cat "$DE/manifests/train.tsv" "$Q" > "$ST/joint_k2.tsv"
wc -l "$ST/joint_k1.tsv" "$ST/joint_k2.tsv"

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

run () {  # $1 name, $2 train manifest
  echo "=== ARM $1 ($2) ==="
  python train_ctc.py --train_manifest "$2" $VAL $POSE $REC --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
}
run div_control "$DE/manifests/train.tsv"
run div_k1      "$ST/joint_k1.tsv"
run div_k2      "$ST/joint_k2.tsv"

echo "=== SUMMARY — div_k1 vs div_k2 is the decisive contrast (identical clip budget) ==="
for R in div_control div_k1 div_k2; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('%-12s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e:
    print('%-12s NO RESULT %s' % ('$R', e))
"
done
echo "seed-42 refs (beam-5): real-only control 9.88 | joint@0.5:1 k=2 10.59 | joint@1:1 k~2 10.32"
echo "STITCH_DIVERSITY_DONE"
