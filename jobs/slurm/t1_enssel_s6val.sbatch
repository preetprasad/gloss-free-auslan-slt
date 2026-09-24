#!/bin/bash -l
# t1-enssel Stage 1: the S6 ensemble's VALIDATION n-best dump (only the test dump exists), so selective-prediction
# thresholds can be fitted on validation and carried to test for the headline system itself. Same members, order,
# beam 10 and LP 0.3 as the recorded 11.802 decode (kaya/mbr_wer_test.sbatch); --rgb_feat_dir is mandatory (09-10m).
#SBATCH --job-name=t1-enssel
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; cd "$G/code/phase2_baseline"
M=$G/scratch_backup/dwpose/manifests; RGB=$G/scratch_backup/dwpose/rgb_feats
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
python tools/ensemble_mbr_wer.py --ckpts "$S6" --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$RGB" --rgb_feature_dim 1152 --device cuda --beam 10 \
  --eval_manifest "$M/val.tsv" --lps 0.3 --dump_dir "$G/enssel/dumps_S6_val" 2>&1 | tail -12
echo "T1_ENSSEL_DONE"
