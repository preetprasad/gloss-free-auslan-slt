#!/bin/bash -l
# SWA ensemble (CPU): swap the stronger SWA members into the ensemble, val-selected vs the 11.50 best.
#SBATCH --job-name=swa-ens
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q sacrebleu jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
BASE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --beam 5 --length_penalty 1.0"
SV=$G/comm_dwpose_swa_vlp80.pt; SF=$G/comm_dwpose_swa_fus.pt; VL=$G/comm_dwpose_ctc_vlplong80/best.pt
R0=$G/comm_dwpose_rgb_fusion/best.pt; V0=$G/comm_dwpose_vlp_ctc/best.pt
C2=$G/comm_dwpose_ctc_s202/best.pt; F1=$G/comm_dwpose_fus_s101/best.pt; F2=$G/comm_dwpose_fus_s202/best.pt

run(){ echo "### $1 ($4) ###"; python ensemble_decode.py --ckpts "$2" $BASE --eval_manifest "$3" --device cpu; }
for M in val test; do
  MF="$DE/manifests/$M.tsv"; echo "########## ${M^^} ##########"
  run "A:SV+R0"                  "$SV,$R0"                 "$MF" A
  run "B:SV+R0+V0+C2+F1+F2"      "$SV,$R0,$V0,$C2,$F1,$F2" "$MF" B
  run "C:SV+SF+V0+C2+F1+F2"      "$SV,$SF,$V0,$C2,$F1,$F2" "$MF" C
  run "D:SV+VL+R0+V0+C2+F1+F2"   "$SV,$VL,$R0,$V0,$C2,$F1,$F2" "$MF" D
done
echo "SWA_ENS_DONE"
