#!/bin/bash -l
# Decode-time ensemble (training-free): average logits across our best same-vocab checkpoints.
# Current best single = RGB fusion 10.335. Test whether ensembling beats it for free.
#SBATCH --job-name=ensemble
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
G=/group/pmc099/pprasad
COMMON="--test_manifest $DE/manifests/test.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --beam 5 --length_penalty 0.6 --device cpu"

echo "### E1: RGB-fusion(10.335) + VLP(10.27) ###"
python ensemble_decode.py --ckpts "$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt" $COMMON
echo "### E2: RGB-fusion + VLP + seed13 ###"
python ensemble_decode.py --ckpts "$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_seed13/best.pt" $COMMON
echo "### E3: pose-only VLP + 3 seeds ###"
python ensemble_decode.py --ckpts "$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_seed13/best.pt,$G/comm_dwpose_ctc_seed42/best.pt,$G/comm_dwpose_ctc_seed7/best.pt" $COMMON
echo "ENSEMBLE_DONE"
