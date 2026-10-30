#!/bin/bash -l
# LOCKED test confirm of the extended-sweep selection. Validation (job 1101909) showed the LP optimum
# is BELOW 0.6: LP 0.3 = BLEU-max (12.470 / WER 0.9164), LP 0.2 = WER-min (12.389 / 0.9089), both
# dominating the prior LP 0.6 (12.216 / 0.9629) and the LP 1.0 baseline (11.500 / 1.0441) on BOTH
# metrics. Confirm the val-Pareto endpoints (0.2, 0.3) on the official test ONCE; 0.6 for continuity.
#SBATCH --job-name=mbr-wer-test2
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
  --eval_manifest "$DE/manifests/test.tsv" --lps 0.2,0.3,0.6
echo "MBR_WER_TEST2_DONE"
