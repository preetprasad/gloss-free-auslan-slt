#!/bin/bash -l
#SBATCH --job-name=agreement
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
export RGB_DIR=/scratch/pmc099/pprasad/dwpose/rgb_clips
export OUT_DIR=/scratch/pmc099/pprasad/dwpose
export DWPOSE_DEVICE=cuda
python /group/pmc099/pprasad/code/results/dwpose_experiment/agreement_rtmlib.py \
    --n_clips 150 --frames 6 --device cuda
