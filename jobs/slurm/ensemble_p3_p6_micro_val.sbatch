#!/bin/bash -l
# P7: validation-only, low-weight complementarity gate for P3/P6.
# The prior equal-weight S6+P3 mixture was negative; this tests whether either
# member contributes useful diversity only at a deliberately small weight.
#SBATCH --job-name=auslan-ensemble-p3p6-micro-val
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
P3="$DE/runs/comm_dwpose_rgb_temporal_ctc_p3/best.pt"
P6="$DE/runs/comm_dwpose_gated_temporal_ctc_p6/best.pt"
COMMON="--eval_manifest $DE/manifests/val.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --rgb_encoder_layers 2 --beam 10 --length_penalty 0.3 --device cpu"

run () {
  local label="$1" ckpts="$2" weights="$3"
  echo "### ${label} validation-only ###"
  python ensemble_decode.py --ckpts "$ckpts" --weights "$weights" $COMMON
}
run "S6+P3@0.10" "$S6,$P3" "1,1,1,1,1,1,0.10"
run "S6+P6@0.10" "$S6,$P6" "1,1,1,1,1,1,0.10"
run "S6+P3+P6@0.10" "$S6,$P3,$P6" "1,1,1,1,1,1,0.10,0.10"
echo "P7_MICRO_ENSEMBLE_DONE validation_only=true test_set_used=false"
