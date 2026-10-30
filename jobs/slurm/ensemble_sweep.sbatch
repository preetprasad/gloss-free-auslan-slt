#!/bin/bash -l
# Ensemble sweep over the new diverse members (CPU, no GPU contention). VAL-selected, then TEST.
# Star new member: VLP-long pose model (10.71). Combine with RGB-fusion (10.335) + others.
#SBATCH --job-name=ens-sweep
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
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --beam 5 --length_penalty 1.0"
VL=$G/comm_dwpose_ctc_vlplong80/best.pt; R0=$G/comm_dwpose_rgb_fusion/best.pt
V0=$G/comm_dwpose_vlp_ctc/best.pt; C2=$G/comm_dwpose_ctc_s202/best.pt
F1=$G/comm_dwpose_fus_s101/best.pt; F2=$G/comm_dwpose_fus_s202/best.pt

run () { echo "### $1 :: $2 ###"; python ensemble_decode.py --ckpts "$3" $POSE --eval_manifest "$4" --device cpu; }

echo "########## VAL selection ##########"
run VAL A:VL+R0            "$VL,$R0" "$DE/manifests/val.tsv"
run VAL B:VL+R0+V0+F1      "$VL,$R0,$V0,$F1" "$DE/manifests/val.tsv"
run VAL C:VL+R0+V0+C2+F1+F2 "$VL,$R0,$V0,$C2,$F1,$F2" "$DE/manifests/val.tsv"
echo "########## TEST (all candidates; report the VAL-best one) ##########"
run TEST A:VL+R0            "$VL,$R0" "$DE/manifests/test.tsv"
run TEST B:VL+R0+V0+F1      "$VL,$R0,$V0,$F1" "$DE/manifests/test.tsv"
run TEST C:VL+R0+V0+C2+F1+F2 "$VL,$R0,$V0,$C2,$F1,$F2" "$DE/manifests/test.tsv"
echo "ENS_SWEEP_DONE"
