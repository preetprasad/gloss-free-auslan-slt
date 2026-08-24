#!/bin/bash -l
# WER-lowering (decode-time, no retraining) on the S6 ensemble — VALIDATION ONLY.
# Length-penalty sweep + MBR-WER over a single n-best decode. Selection is on val;
# the official test is scored only once, for the chosen config, in a separate step.
#SBATCH --job-name=mbr-wer-val
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
COMMON="--ckpts $S6 --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --device cpu --beam 10"

echo "=== SMOKE (24 val clips) — fail fast if broken ==="
python tools/ensemble_mbr_wer.py $COMMON --eval_manifest "$DE/manifests/val.tsv" --limit 24

echo "=== FULL VALIDATION sweep (LP grid + MBR-WER) ==="
python tools/ensemble_mbr_wer.py $COMMON --eval_manifest "$DE/manifests/val.tsv" \
  --lps 0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0
echo "MBR_WER_VAL_DONE"
