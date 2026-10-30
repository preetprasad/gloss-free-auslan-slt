#!/bin/bash -l
# EXTENDED ENSEMBLE — designed from the headroom analysis, not from member count.
#
# THE RESEARCH THAT SHAPED THIS (results/ENSEMBLE_HEADROOM_2026-09-09.md):
#   * The per-sentence oracle ceiling has NOT saturated: 1 member 9.43 -> 6 members 15.14, still
#     +0.49 per member. So extra members DO add reachable headroom.
#   * But a real ensemble captures only ~30% of that gap. Reference-free selectors on the same pool:
#     plurality vote -0.7%, MBR-chrF 0.1%, MBR-sentBLEU 6.8%, LOGIT AVERAGING 30.5%. Selecting among
#     finished hypotheses is worthless; combining token-level distributions is not.
#   * Therefore each +1.0 of oracle converts to ~+0.3 actual, against measured ensemble noise of +-0.3.
#     **Adding members yields gains at or below the noise floor UNLESS they raise the oracle, which
#     requires genuine RECIPE DIVERSITY rather than more seeds.**
#   * Cross-recipe systems produce identical output only 13-26% of the time; same-recipe greedy/beam
#     pairs agree 47-50%. The existing leave-one-out ablation agrees: redundant same-recipe members add
#     nothing or hurt (VLP40 net-negative; removing CTC-s202 IMPROVED the ensemble by +0.25).
#
# SO THIS JOB IS BUILT AS A CONTROLLED TEST OF THE DIVERSITY HYPOTHESIS, not as a bigger pile:
#   X1 S6 alone                  -- in-batch reference for this batch
#   X2 DUR x3                    -- the current validation winner (12.280 vs S6 11.707 in 1191189)
#   X3 DUR x6                    -- MORE OF THE SAME RECIPE. Research predicts ~nothing over X2.
#   X4 S6 + DUR x3  (remapped)   -- a DIFFERENT recipe added. Research predicts more than X3.
#   X5 max-diversity subset      -- one strong member per distinct recipe
# X3 vs X4 is the point: same number of added members, one adds seeds and the other adds a recipe.
#
# --remap_vocab is MANDATORY here: the duration models carry a different vocabulary ORDERING (same
# size 3063, different index->word map), which silently corrupted four candidates in job 1191189.
# --dump_dir is set on every candidate so the oracle and member diversity can be MEASURED for the real
# member set afterwards -- the 11.80 operating point still cannot be error-analysed at all.
#
# VALIDATION ONLY. Stage 2 (one locked test) stays manual and must use the validation winner.
#SBATCH --job-name=ens-ext
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
cd "$G/code/phase2_baseline"; M="$DE/manifests"; OUT=$G/ens_ext; mkdir -p "$OUT"

S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
D42="$ST/runs/tmp_dur/best.pt"
DUR3="$D42,$ST/runs/dur_joint_s101/best.pt,$ST/runs/dur_joint_s202/best.pt"
DUR6="$DUR3,$ST/runs/dur_joint_s303/best.pt,$ST/runs/dur_joint_s404/best.pt,$ST/runs/dur_joint_s505/best.pt"
# one strong member per DISTINCT recipe: VLP-80, RGB-fusion, plain CTC, duration-stitch
DIV="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$D42"

for f in $(echo "$S6,$DUR6,$DIV" | tr "," " "); do
  [ -f "$f" ] || { echo "FAIL: missing checkpoint $f"; exit 1; }
done
echo "all checkpoints present"

val () {  # $1=tag $2=ckpt list
  echo "=== CANDIDATE $1 ==="
  python tools/ensemble_mbr_wer.py --ckpts "$2" --beam 10 --device cuda --remap_vocab \
    --eval_manifest "$M/val.tsv" --test_manifest "$M/val.tsv" --dump_dir "$OUT/hyps_$1" \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame || echo "  (candidate failed)"
}
val X1_S6            "$S6"
val X2_DUR3          "$DUR3"
val X3_DUR6          "$DUR6"
val X4_S6_plus_DUR3  "$S6,$DUR3"
val X5_maxdiversity  "$DIV"

echo "=== READ THIS FIRST: X3 vs X4 ==="
echo "  X3 adds 3 SAME-RECIPE members to X2; X4 adds a DIFFERENT recipe to X1."
echo "  If X4-X1 >> X3-X2, diversity is what matters and the ensemble should be built by recipe."
echo "  If both are ~0, the ensemble is saturated and this line should be closed."
echo "reference: 1191189 in-batch A_S6 11.707 / E_dur_x3 12.280; historical S6 val 12.470 / test 11.802"
echo "ENS_EXT_DONE"
