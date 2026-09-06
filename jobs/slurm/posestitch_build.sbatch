#!/bin/bash -l
# PoseStitch step A — generate the stitched synthetic continuous corpus from MM-WLAuslan valid clips.
#SBATCH --job-name=posestitch-build
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
M=/scratch/pmc099/pprasad/mmwl; DE=/scratch/pmc099/pprasad/dwpose
cd /group/pmc099/pprasad/code/phase2_baseline
python tools/build_posestitch_corpus.py \
  --valid_labels "$M/labels/Valid.json" \
  --valid_npz_dir "$M/dwpose_clips/valid" \
  --ad_manifest "$DE/manifests/train.tsv" \
  --out_dir "$DE/posestitch" \
  --min_cov 0.6 --min_signs 2 --realizations 3
echo "POSESTITCH_BUILD_DONE"
