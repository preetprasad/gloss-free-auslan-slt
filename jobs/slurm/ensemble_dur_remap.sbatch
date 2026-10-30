#!/bin/bash -l
# RE-RUN of the ensemble-duration candidates that job 1191189 INVALIDATED via a vocabulary bug.
#
# WHAT WENT WRONG (diagnosed 2026-09-09): tools/ensemble_mbr_wer.py applied member 0's vocabulary to
# every member. The S6 models and the duration-stitch models both have vocab size 3063 -- so nothing
# crashed and nothing warned -- but the orders differ (index 5 is 'i' for S6, 'the' for duration),
# because the duration models were trained on the joint real+synthetic manifest and Vocab.build sorts
# by frequency. Averaging their logits silently scrambled which word each index denotes.
#
# The damage scaled with the number of mismatched members, which is exactly what validation showed:
#   A_S6_baseline        11.707   (internally consistent -> VALID)
#   E_dur_x3_only        12.280   (internally consistent -> VALID, and the winner)
#   B_S6_plus_dur42      10.496   \
#   F_dur_x3_plus_rgbfus  5.482    |  ALL FOUR ARE HARNESS ARTEFACTS, not results
#   C_S6_plus_dur_x3      5.102    |  the plausible reading they invited -- "mixing model families
#   D_S6_plus_dur_x3_vlp  2.803   /   hurts ensembles" -- would have been WRONG
#
# THE FIX: --remap_vocab aligns each member into member 0's vocabulary in BOTH directions (decoder
# inputs converted to the member's ids; output distribution permuted back, with -inf for words the
# member cannot produce). Round-trip unit-tested locally. Without the flag a mismatch is now a hard
# error instead of silent corruption.
#
# THIS JOB re-runs only the four invalidated candidates. A and E need no re-run: they were already
# vocab-consistent and their numbers stand.
#
# STILL VALIDATION-ONLY. Stage 2 (one locked test) remains manual and must use the overall validation
# winner across BOTH this job and the valid A/E numbers from 1191189.
#SBATCH --job-name=ens-remap
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
G=/group/pmc099/pprasad; ST=/scratch/pmc099/pprasad/stitch_v2; DE=/scratch/pmc099/pprasad/dwpose
cd "$G/code/phase2_baseline"; M="$DE/manifests"

S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
D42="$ST/runs/tmp_dur/best.pt"; D101="$ST/runs/dur_joint_s101/best.pt"; D202="$ST/runs/dur_joint_s202/best.pt"
DVLP="$ST/runs/dvlp_V2_ctcdur/best.pt"
for f in $(echo "$S6" | tr "," " ") "$D42" "$D101" "$D202" "$DVLP"; do
  [ -f "$f" ] || { echo "FAIL: missing checkpoint $f"; exit 1; }
done

val () {
  echo "=== CANDIDATE $1 (vocab-remapped) ==="
  python tools/ensemble_mbr_wer.py --ckpts "$2" --beam 10 --device cuda \
    --remap_vocab --eval_manifest "$M/val.tsv" --test_manifest "$M/val.tsv" \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame || echo "  (candidate failed)"
}
val "B_S6_plus_dur42"       "$S6,$D42"
val "C_S6_plus_dur_x3"      "$S6,$D42,$D101,$D202"
val "D_S6_plus_dur_x3_vlp"  "$S6,$D42,$D101,$D202,$DVLP"
val "F_dur_x3_plus_rgbfus"  "$D42,$D101,$D202,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_fus_s101/best.pt"

echo "=== COMPARE AGAINST THE VALID NUMBERS FROM 1191189 ==="
echo "  A_S6_baseline  val 11.707   (vocab-consistent, no re-run needed)"
echo "  E_dur_x3_only  val 12.280   (vocab-consistent, current validation WINNER)"
echo "  historical S6 reference: val 12.470 / test 11.802 -- a DIFFERENT batch, do not mix"
echo "ENS_REMAP_DONE"
