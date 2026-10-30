#!/bin/bash -l
# THREE MORE DURATION-STITCH SEEDS — to extend the ensemble that just won on validation.
#
# WHY (job 1191189, the valid half). Of six ensemble candidates, four were invalidated by a vocabulary
# bug; the two that were internally vocab-consistent stand:
#     A_S6_baseline   (the established 6-member S6)   val 11.707
#     E_dur_x3_only   (3 duration-stitch models)      val 12.280   <-- +0.573, IN-BATCH
# A 3-member ensemble of duration-stitch models beat the 6-member S6 by +0.573 on validation. The S6
# ensemble is known to be diversity-limited (6 members but only 3 recipes; the leave-one-out ablation
# showed redundant same-recipe members add nothing), and E has only THREE members. The obvious and
# cheap question is whether the winning recipe scales with more members.
#
# This job trains dur_joint at seeds 303/404/505, giving six duration models in total for a later
# ensemble sweep. It is deliberately NOT an ensemble job: the members are built here, the sweep is run
# separately once ens-remap (1196177) reports whether MIXED families work with --remap_vocab.
#
# EVERYTHING MATCHES THE EXISTING THREE EXACTLY -- same corpus ($ST/joint_dur.tsv, the seed-42
# --sign_dur build), same 40-epoch recipe, only --seed differs. **40 epochs is deliberate even though
# the epoch audit (09-09q) shows it is short**: ensemble members must be trained identically, and the
# existing s42/s101/s202 models are 40-epoch. Mixing schedules within an ensemble would repeat, in a
# different form, the mistake that made four candidates in 1191189 meaningless.
#
# Paired controls are NOT re-run: the +1.48 lever is already replicated 3/3 and these seeds exist only
# to supply ensemble members, not to re-test the lever.
#SBATCH --job-name=dur-seeds2
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
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad; ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"

# The seed-42 duration corpus lives on scratch and is ~5 days old, so it is not purge-eligible this
# week; fail loudly rather than silently training on the wrong manifest if that ever changes.
test -s "$ST/joint_dur.tsv" || { echo "FAIL: $ST/joint_dur.tsv missing -- rebuild with --sign_dur seed 42"; exit 1; }
n_missing=$(python - "$ST/joint_dur.tsv" <<'PYCHK'
import os, sys
bad = 0
for line in open(sys.argv[1]):
    if "\t" in line and not os.path.exists(line.split("\t", 1)[0]):
        bad += 1
print(bad)
PYCHK
)
[ "$n_missing" -eq 0 ] || { echo "FAIL: $n_missing clips referenced by joint_dur.tsv are missing"; exit 1; }
wc -l "$ST/joint_dur.tsv"

VAL="--val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
for S in 303 404 505; do
  echo "=== ARM dur_joint_s$S ==="
  python train_ctc.py --train_manifest "$ST/joint_dur.tsv" $VAL $POSE $BASE --seed "$S" \
    --out_dir "$ST/runs/dur_joint_s$S"
  python tools/eval_beam.py --ckpt "$ST/runs/dur_joint_s$S/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
done

echo "=== SUMMARY — six duration members now available for an ensemble sweep ==="
for S in 42 101 202 303 404 505; do
  D=$ST/runs/dur_joint_s$S; [ "$S" = "42" ] && D=$ST/runs/tmp_dur
  python -c "
import json,os
p='$D/progress.json'
if not os.path.exists(p): print('%-18s (missing)'%'dur_joint_s$S')
else:
    d=json.load(open(p)); t=d.get('test',{})
    print('%-18s val=%6.3f @ep%-3d test=%6.3f' % ('dur_joint_s$S', d['best']['bleu4'], d['best']['epoch'], t.get('bleu4',float('nan'))))
"
done
echo "next: ensemble sweep over the 6, and vs S6, once ens-remap (1196177) reports on mixed families"
echo "DUR_SEEDS2_DONE"
