#!/bin/bash -l
# Validation-only gate: does P3's temporal-pyramid + CTC checkpoint complement
# the established S6 ensemble?  It does not read the official test manifest.
#SBATCH --job-name=auslan-ensemble-p3-val
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"

S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
P3="$DE/runs/comm_dwpose_rgb_temporal_ctc_p3/best.pt"

# S6 is already fixed: LP 0.2 = 12.389 / 0.9089 and LP 0.3 = 12.470 / 0.9164
# on validation.  Only decode the genuinely new S6+P3 candidate.
for LP in 0.2 0.3; do
  python ensemble_decode.py \
    --ckpts "$S6,$P3" --eval_manifest "$DE/manifests/val.tsv" \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame \
    --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --rgb_encoder_layers 2 \
    --beam 10 --length_penalty "$LP" --device cpu
done
echo "ENSEMBLE_P3_VAL_DONE test_set_used=false"
