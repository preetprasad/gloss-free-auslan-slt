#!/bin/bash -l
# Ensemble maximisation (CPU): weighted logit-averaging x member-set x beam/LP. VAL-selected.
# Current best = 6-member uniform = 11.50 test / 12.17 val. Push via dev-tuned weights + bigger pool.
#SBATCH --job-name=ens-max
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q sacrebleu jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
BASE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152"
VL=$G/comm_dwpose_ctc_vlplong80/best.pt; R0=$G/comm_dwpose_rgb_fusion/best.pt
V0=$G/comm_dwpose_vlp_ctc/best.pt; C1=$G/comm_dwpose_ctc_s101/best.pt
C2=$G/comm_dwpose_ctc_s202/best.pt; C3=$G/comm_dwpose_ctc_s303/best.pt
F1=$G/comm_dwpose_fus_s101/best.pt; F2=$G/comm_dwpose_fus_s202/best.pt; F3=$G/comm_dwpose_fus_s303/best.pt
V128=$G/comm_dwpose_ctc_vlp128/best.pt; V256=$G/comm_dwpose_ctc_vlp256/best.pt

S6="$VL,$R0,$V0,$C2,$F1,$F2";                  W6="1.3,1.3,1.0,0.9,0.9,0.9"
S8="$VL,$R0,$V0,$C2,$F1,$F2,$V128,$V256";       W8="1.3,1.3,1.0,0.9,0.9,0.9,0.6,0.6"
S9="$VL,$R0,$V0,$C2,$F1,$F2,$C1,$C3,$F3";        W9="1.3,1.3,1.0,0.9,0.9,0.9,0.8,0.8,0.9"

run () {  # $1 label  $2 ckpts  $3 weights(""=uniform)  $4 beam  $5 lp  $6 manifest
  echo "### $1 (beam=$4 lp=$5 w=${3:-uniform}) ###"
  python ensemble_decode.py --ckpts "$2" $BASE --beam $4 --length_penalty $5 \
    ${3:+--weights "$3"} --eval_manifest "$6" --device cpu
}
for M in val test; do
  MF="$DE/manifests/$M.tsv"
  echo "########## ${M^^} ##########"
  run "S6-uniform"     "$S6" ""   5  1.0 "$MF"
  run "S6-weighted"    "$S6" "$W6" 5  1.0 "$MF"
  run "S6-weighted-b10" "$S6" "$W6" 10 1.0 "$MF"
  run "S8-weighted"    "$S8" "$W8" 5  1.0 "$MF"
  run "S9-weighted"    "$S9" "$W9" 5  1.0 "$MF"
done
echo "ENS_MAX_DONE"
