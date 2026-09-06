#!/bin/bash -l
# TEMPORAL FIDELITY — does fixing the duration mismatch move the stitch gain?
#
# The 2026-09-05 audit found our synthetic clips are temporally wrong in two compounding ways:
#   (a) BUG: Auslan-Daily is 25 fps, MM-WLAuslan is 30 fps, and the stitcher never resampled, so
#       every synthetic clip was dilated 1.2x before any linguistic argument;
#   (b) continuous signs are ~72% SHORTER than their citation forms (paired per-gloss: 80.05 fr
#       @30fps vs 22.49 fr @25fps = 4.32x in the raw frames the model sees).
# Measured: one citation clip averages 79.4 frames while a WHOLE real sentence averages 91.2;
# our synthetic clips run 250.8.
#
# This was worth running only because the pose-reliance probe REFUTED the "it is all decoder-side
# language modelling" hypothesis: the joint models use the visual input 15-19% MORE than control,
# and the gain holds on the UNSEEN-target split (+0.71 / +0.51 in two paired seeds). So visual
# fidelity is a live lever, not a dead one.
#
# Fully internally controlled: all four arms in one job, one sentence pool, k=1, identical clip
# budget, seed 42. Only the TEMPORAL treatment moves.
#SBATCH --job-name=stitch-temporal
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
SENT="$ST/sentences_all.txt"; BUDGET=6880
COMMON="--min_cov 0.9 --min_signs 3 --realizations 1 --max_clips $BUDGET --seed 42"

echo "=== corpus: fps-corrected (25/30 = 0.8333) ==="
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$SENT" \
  --out_dir "$ST/corpus_fps" $COMMON --fps_scale 0.8333
echo "=== corpus: per-sign duration sampled from the REAL continuous distribution ==="
python tools/build_posestitch_corpus_v2.py $LAB $DIRS --sentences "$SENT" \
  --out_dir "$ST/corpus_dur" $COMMON --sign_dur

for T in fps dur; do
  cat "$DE/manifests/train.tsv" "$ST/corpus_$T/posestitch_v2_train.tsv" > "$ST/joint_$T.tsv"
done
python - <<'PY'
import numpy as np
for tag, m in [("no-resample (div_k1)", "/scratch/pmc099/pprasad/stitch_v2/corpus_k1/posestitch_v2_train.tsv"),
               ("fps-corrected", "/scratch/pmc099/pprasad/stitch_v2/corpus_fps/posestitch_v2_train.tsv"),
               ("duration-modelled", "/scratch/pmc099/pprasad/stitch_v2/corpus_dur/posestitch_v2_train.tsv")]:
    try:
        rows = [l.split("\t")[0] for l in open(m)][:300]
        L = [np.load(p)["KPS"].shape[0] for p in rows]
        print("%-24s mean %6.1f  median %5.0f frames" % (tag, np.mean(L), np.median(L)))
    except Exception as e:
        print("%-24s (unavailable: %s)" % (tag, e))
print("%-24s mean   91.2 frames  <-- REAL Auslan-Daily clips" % "target")
PY

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
run () {
  echo "=== ARM $1 ($2) ==="
  python train_ctc.py --train_manifest "$2" $VAL $POSE $REC --out_dir "$ST/runs/$1"
  python tools/eval_beam.py --ckpt "$ST/runs/$1/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
}
run tmp_control  "$DE/manifests/train.tsv"
run tmp_noresamp "$ST/joint_k1.tsv"
run tmp_fps      "$ST/joint_fps.tsv"
run tmp_dur      "$ST/joint_dur.tsv"

echo "=== SUMMARY ==="
for R in tmp_control tmp_noresamp tmp_fps tmp_dur; do
  python -c "
import json
try:
    d=json.load(open('$ST/runs/$R/progress.json')); t=d.get('test',{})
    print('%-14s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e: print('%-14s NO RESULT %s' % ('$R', e))
"
done
echo "prior refs (beam-5, same recipe/seed): div_control 9.69 | div_k1 no-resample 10.45"
echo "STITCH_TEMPORAL_DONE"
