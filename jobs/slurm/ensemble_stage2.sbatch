#!/bin/bash -l
# STAGE 1b + STAGE 2 for the duration-augmented ensemble.
#
# 1196177 (`ens-remap`) fixed the vocabulary bug and every previously-artefactual candidate came back
# in the 12.1-12.8 band on validation, confirming the diagnosis. Validation standings (beam-10):
#   A_S6_baseline    11.707   (from 1191189, vocab-consistent)
#   E_dur_x3_only    12.280
#   B_S6+dur42       12.251 @ LP0.6
#   C_S6+dur_x3      12.634 @ LP0.8
#   D_S6+dur_x3+vlp  12.716 @ LP0.6
#   F_dur_x3+rgbfus  12.778 @ LP0.8   <- validation argmax
#
# BUT THE LP GRID STARTED AT 0.6, AND THAT IS A REAL PROBLEM. Historical S6 peaks at LP 0.3 (val
# 12.470 / test 11.802); inside this grid the same checkpoints score only 11.707, because 0.3 was
# never tried. So the grid can miss an optimum, and F/D may also peak below 0.6. D's best is already
# AT the grid edge (0.6), which is exactly the signature of a truncated sweep.
#
# STAGE 1b therefore re-sweeps the two leading candidates down to LP 0.2 on VALIDATION ONLY. It also
# re-sweeps A_S6 over the same extended grid so the comparison is like-for-like rather than against a
# differently-gridded historical number.
# STAGE 2 then evaluates the SINGLE overall validation argmax on the official test set ONCE, at that
# one pinned LP (`--lps`), and dumps hypotheses so the operating point can finally be error-analysed
# -- the project log records that the 11.80 point "cannot be error-analysed at all" because no ensemble
# decode was ever saved.
#
# Checkpoints are read from the /group backup, not /scratch: the duration members live only on
# purge-eligible scratch and are now load-bearing for the headline.
#SBATCH --job-name=ens-stage2
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
OUT=$G/ens_stage2; M="$OUT/manifests"; mkdir -p "$M" "$OUT/dumps"

SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  elif [ -f "$B/dwpose/manifests/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$B/dwpose/manifests/$f.tsv" > "$M/$f.tsv"
  else echo "FAIL: cannot source $f.tsv"; exit 1; fi
done
wc -l "$M/val.tsv" "$M/test.tsv"

R="$B/stitch_meta/stitch_v2/runs"
D42="$R/tmp_dur/best.pt"; D101="$R/dur_joint_s101/best.pt"; D202="$R/dur_joint_s202/best.pt"
DVLP="$R/dvlp_V2_ctcdur/best.pt"
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
F_SET="$D42,$D101,$D202,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_fus_s101/best.pt"
D_SET="$S6,$D42,$D101,$D202,$DVLP"
for f in $(echo "$S6,$F_SET,$D_SET" | tr "," " "); do
  [ -f "$f" ] || { echo "FAIL: missing checkpoint $f"; exit 1; }
done
echo "all checkpoints present"

GRID="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"

echo "########## STAGE 1b — extended LP sweep, VALIDATION ONLY ##########"
for C in "A_S6:$S6" "F_dur_x3_plus_rgbfus:$F_SET" "D_S6_plus_dur_x3_vlp:$D_SET"; do
  n=${C%%:*}; k=${C#*:}
  echo "=== VAL $n ==="
  python tools/ensemble_mbr_wer.py --ckpts "$k" --beam 10 --device cuda --remap_vocab \
    --lps "$GRID" --eval_manifest "$M/val.tsv" --test_manifest "$M/val.tsv" $POSE
done

echo
echo "########## STAGE 2 — ONE locked test evaluation at the VALIDATION argmax ##########"
echo "Read the Stage-1b tables above, then run ONE of the following by hand. Stage 2 is deliberately"
echo "NOT automated: automating 'pick the best and score it on test' is how a sweep becomes test-tuning"
echo "if the selection line is ever edited. The command is printed, the human picks, it runs once."
echo
for C in "F_dur_x3_plus_rgbfus:$F_SET" "D_S6_plus_dur_x3_vlp:$D_SET" "A_S6:$S6"; do
  n=${C%%:*}; k=${C#*:}
  echo "  # $n at its validation-selected LP <LP>:"
  echo "  python tools/ensemble_mbr_wer.py --ckpts \"$k\" --beam 10 --device cuda --remap_vocab \\"
  echo "    --lps <LP> --eval_manifest \"$M/test.tsv\" --test_manifest \"$M/test.tsv\" $POSE \\"
  echo "    --dump_dir \"$OUT/dumps/$n\""
  echo
done
echo "reference, same test set, same metric, identical S6 checkpoints: S6 beam-10/LP0.3 = 11.802 / WER 0.907"
echo "ENS_STAGE2_VAL_DONE"
