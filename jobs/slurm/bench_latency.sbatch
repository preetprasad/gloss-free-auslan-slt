#!/bin/bash -l
#SBATCH --job-name=bench-lat
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
CK=/group/pmc099/pprasad/comm_dwpose_vlp_ctc/best.pt
COMMON="--init_ckpt $CK --test_manifest $DE/manifests/test.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame --device cpu --limit 60"
echo "### GREEDY (single-stream, CPU) ###"; python bench_latency.py $COMMON --beam 0
echo "### BEAM-5 (single-stream, CPU) ###"; python bench_latency.py $COMMON --beam 5
echo "BENCH_DONE"
