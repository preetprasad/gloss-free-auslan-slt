#!/bin/bash -l
# Extended decode-time sweep — LP 0.6 was the EDGE of the previous grid and won, so the optimum may
# be lower. Sweep LP 0.2-0.7 (finer, lower) + MBR-WER + MBR-chrF on the S6 ensemble. VALIDATION ONLY.
#SBATCH --job-name=mbr-wer-val2
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
python tools/ensemble_mbr_wer.py \
  --ckpts "$S6" --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" --rgb_feature_dim 1152 --device cpu --beam 10 \
  --eval_manifest "$DE/manifests/val.tsv" --lps 0.2,0.3,0.4,0.5,0.6,0.7
echo "MBR_WER_VAL2_DONE"
