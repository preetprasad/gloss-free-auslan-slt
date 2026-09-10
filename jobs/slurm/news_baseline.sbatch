#!/bin/bash -l
# NEWS-ONLY BASELINE — Atif action item ("just news split not done").
#
# What we HAVE done with News: joint-trained Comm+News (22,088 clips) and evaluated on the COMM
# test split -> 8.89, i.e. adding News HURT in-domain Communication BLEU by ~1.4 (register shift).
# What we have NEVER done: train on News and evaluate on NEWS. That is a separate result and it is
# directly comparable to published numbers, because the Auslan-Daily paper's Table 3 reports a
# full News column.
#
# PUBLISHED TARGETS — Auslan-Daily News, single-person gloss-free SLT (paper Table 3):
#   SL-Luong      Pose  BLEU-4 **2.81**  ROUGE-L 20.65   <- best published on News
#   GASLT         Pose  BLEU-4  2.72     ROUGE-L 18.76
#   SL-Transf     Pose  BLEU-4  2.11     ROUGE-L 20.25
#   SL-Transf     RGB   BLEU-4  2.52 | TSPNet 2.26 | MMTLB 2.31 | SL-Luong RGB 2.68
# Note how much LOWER these are than the Communication column (9.20-9.95): News is the harder split.
#
# WHY IT IS HARDER, and WHY IT IS THE MORE HONEST SPLIT — measured locally 2026-09-08 on the
# real manifests, confirming the paper's own statistics:
#   target length   Comm train 8.1 words (paper 7.8)  |  News train 17.0 (paper 16.9)
#   clip length     Comm ~73 frames                   |  News mean 220.4, median 200, p90 376
#   vocabulary      Comm train 3,064 types            |  News train 18,893 (6x), only 10% shared
#   VERBATIM TEST-TARGET OVERLAP WITH TRAIN:
#       Communication  230/800 = **28.8%**
#       News            38/700 = ** 5.4%**   <-- ~5x LESS contaminated
# That last row is the important one. Our Communication headline is inflated by memorisation
# (seen-target ~30 vs unseen ~5.8), and the PUBLISHED Communication numbers are inflated the same
# way. News is far less memorisable by construction, so a News result is a much stronger
# generalisation claim than another Communication number -- which is exactly what "prove it with
# results" needs.
#
# AND A CONCRETE CONFIG TRAP, now quantified: our `max_src_frames` default is **256**, tuned on
# Communication's ~73-frame clips. **33.0% of News clips exceed 256 frames** and are subsampled
# away before the model sees them. Arm 2 is therefore not optional -- it tests a Communication-
# specific choice that may be silently crippling the other split.
#
# Arms (matched DWPose+CTC recipe, the same one used for every Communication ablation):
#   news_256  max_src_frames 256   <- exactly the Communication recipe, the honest like-for-like
#   news_512  max_src_frames 512   <- accommodates the 214-frame mean and its tail
#SBATCH --job-name=news-base
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
cd "$G/code/phase2_baseline"

# Locate the News manifests written by tools/build_combined_manifests.py. Fail loudly rather than
# silently training on the wrong split.
M="$DE/manifests"
NTR=""; for c in "$M/news_train.tsv" "$M/news/train.tsv" "$M/News_train.tsv"; do [ -f "$c" ] && NTR="$c" && break; done
NVA=""; for c in "$M/news_val.tsv" "$M/news/val.tsv" "$M/news_dev.tsv" "$M/News_val.tsv"; do [ -f "$c" ] && NVA="$c" && break; done
NTE=""; for c in "$M/news_test.tsv" "$M/news/test.tsv" "$M/News_test.tsv"; do [ -f "$c" ] && NTE="$c" && break; done
if [ -z "$NTR" ] || [ -z "$NVA" ] || [ -z "$NTE" ]; then
  echo "FAIL: could not find News manifests. Present in $M:"; ls -1 "$M"
  echo "Rebuild with tools/build_combined_manifests.py --news_xlsx <Auslan-Daily_News.xlsx> ..."; exit 1
fi
echo "News manifests: train=$NTR val=$NVA test=$NTE"
wc -l "$NTR" "$NVA" "$NTE"
python - <<PY
import statistics as st
for tag, p in [("train","$NTR"), ("test","$NTE")]:
    w=[len(l.rstrip("\n").split("\t",1)[1].split()) for l in open(p) if "\t" in l]
    print(f"  News {tag}: {len(w)} clips, target length mean {st.mean(w):.1f} words (paper says 16.9)")
PY

COM="--train_manifest $NTR --val_manifest $NVA --test_manifest $NTE"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
run () {
  echo "=== ARM $1 (max_src_frames=$2) ==="
  python train_ctc.py $COM $POSE $REC --max_src_frames "$2" --out_dir "$DE/runs/$1"
  python tools/eval_beam.py --ckpt "$DE/runs/$1/best.pt" --manifest "$NTE" --beam 5 --device cuda || true
  cp -r "$DE/runs/$1" "$G/comm_dwpose_$1" 2>/dev/null || true
}
run news_256 256
run news_512 512

echo "=== SUMMARY — compare to PUBLISHED News pose baselines: SL-Luong 2.81 | GASLT 2.72 | SL-Transf 2.11 ==="
for R in news_256 news_512; do
  python -c "
import json
try:
    d=json.load(open('$DE/runs/$R/progress.json')); t=d.get('test',{})
    print('%-10s val_best=%6.3f @ep%-3d  test_greedy=%6.3f  wer=%.3f' % (
        '$R', d['best']['bleu4'], d['best']['epoch'],
        t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
except Exception as e: print('%-10s NO RESULT %s' % ('$R', e))
"
done
echo "for context, our Communication numbers: 11.80 ensemble / 10.97 best single vs published 9.95"
echo "NEWS_BASELINE_DONE"
