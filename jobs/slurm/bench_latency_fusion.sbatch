#!/bin/bash -l
#SBATCH --job-name=bench-fus
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
CK=/group/pmc099/pprasad/comm_dwpose_rgb_fusion/best.pt   # BEST BLEU (10.335 beam)
COMMON="--init_ckpt $CK --test_manifest $DE/manifests/test.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame --rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152 --device cpu --limit 60"
echo "### RGB-FUSION (best BLEU 10.335) GREEDY, CPU ###"; python bench_latency.py $COMMON --beam 0
echo "### RGB-FUSION BEAM-5, CPU ###"; python bench_latency.py $COMMON --beam 5
echo "BENCH_FUS_DONE"
