#!/bin/bash -l
# CAN A DURATION-STITCH MEMBER BEAT THE 11.80 ENSEMBLE?
#
# The S6 ensemble (11.80 / WER 0.907, beam-10/LP0.3) has 6 members drawn from only THREE recipes
# (VLP->CTC, RGB-fusion, plain CTC) plus seed variation, and a leave-one-out ablation showed it is
# diversity-limited: redundant same-recipe members add nothing or hurt. It has NEVER contained a
# stitch-augmented model.
#
# Duration-modelled stitch members are measurably a different KIND of model: replicated +1.48 over
# paired controls (3/3 seeds, mean beam-5 10.97), output diversity 47-48% unique vs 29-36% for
# controls, and a probe shows they rely on the visual input 15-19% more. "Trained on different
# data" may be a better diversity axis than "different seed".
#
# PROTOCOL — this is the project's strictest rule and the reason several past numbers are on a
# do-NOT-report list (11.15, 11.514, 11.697, 11.70 all tested high but lost on validation):
#   * every candidate set is scored on VALIDATION only;
#   * the validation argmax is chosen;
#   * that ONE configuration is then run once on the locked test split.
# Nothing else may be quoted. LP sweep included because ensemble LP is the known lever (LP 1.0 ->
# 0.3 was worth +0.30 BLEU and -6.9pts WER). --dump_dir saves the decodes so the headline operating
# point can finally be error-analysed -- it never could before.
#SBATCH --job-name=ens-dur
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad; ST=/scratch/pmc099/pprasad/stitch_v2
cd "$G/code/phase2_baseline"

S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
D42="$ST/runs/tmp_dur/best.pt"; D101="$ST/runs/dur_joint_s101/best.pt"; D202="$ST/runs/dur_joint_s202/best.pt"
DVLP="$ST/runs/dvlp_V2_ctcdur/best.pt"

echo "##### STAGE 1 — VALIDATION ONLY. Nothing here may be quoted as a result. #####"
val () {
  echo "=== CANDIDATE $1 ==="
  python tools/ensemble_mbr_wer.py --ckpts "$2" --beam 10 --device cuda \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame \
    --eval_manifest "$DE/manifests/val.tsv" --lps "0.2,0.3,0.4,0.6,1.0" || true
}
val "A_S6_baseline"        "$S6"
val "B_S6_plus_dur42"      "$S6,$D42"
val "C_S6_plus_dur_x3"     "$S6,$D42,$D101,$D202"
val "D_S6_plus_dur_x3_vlp" "$S6,$D42,$D101,$D202,$DVLP"
val "E_dur_x3_only"        "$D42,$D101,$D202"
val "F_dur_x3_plus_rgbfus" "$D42,$D101,$D202,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_fus_s101/best.pt"

echo
echo "##### STAGE 2 — read the validation argmax above, then run ONE locked test. #####"
echo "Deliberately NOT automated: the set and LP must be chosen by eye from Stage 1 and run"
echo "separately, so no test number can be produced for a configuration that did not win on"
echo "validation. Re-run this script's val() call with --eval_manifest test.tsv and --dump_dir"
echo "\$ST/hyps/ens_<name> for the winner only."
echo "reference: S6 @ beam-10/LP0.3 = 11.802 / WER 0.9069 (val 12.470 / 0.9164)"
echo "ENS_DUR_VAL_DONE"
