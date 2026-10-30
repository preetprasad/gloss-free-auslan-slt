#!/bin/bash -l
# LOCKED test confirmation of the validation-selected decoding config.
# Validation (job 1101821) selected LP=0.6 as the WER-minimising point that also improves BLEU
# (val: BLEU 12.216 / WER 0.9629 vs the LP=1.0 baseline 11.500 / 1.0441). This scores the official
# test split ONCE for LP=0.6 (the selection) with LP=1.0 alongside as the matched within-run baseline.
# No further searching. Established reference: beam-5 / LP 1.0 = 11.499 BLEU / 0.976 WER (job wer-s6).
#SBATCH --job-name=mbr-wer-test
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
  --eval_manifest "$DE/manifests/test.tsv" --lps 0.6,1.0
echo "MBR_WER_TEST_DONE"
