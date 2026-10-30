#!/bin/bash -l
# Clean ensemble ablation (validation) — incremental 1..6 + leave-one-out, one decode pass.
# The 6-member set was already selected; this is descriptive (member contribution), beam-5/LP1.0
# to match how 11.50 is reported. Members ordered strongest-single first.
#SBATCH --job-name=ens-ablation
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
CK="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
NM="VLP80,CTCs202,RGBfus,VLP40,fus101,fus202"
python tools/ensemble_ablation.py \
  --ckpts "$CK" --names "$NM" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 \
  --eval_manifest "$DE/manifests/val.tsv" --beam 5 --length_penalty 1.0 --device cpu
echo "ENS_ABLATION_DONE"
