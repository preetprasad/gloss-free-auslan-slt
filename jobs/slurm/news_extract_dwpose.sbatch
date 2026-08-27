#!/bin/bash -l
# DATA-SCALING SWING, step 2/4 — DWPose-extract the ~11k News clips (RTMW-DW-x on CUDA),
# identical pipeline to the 14k Communication extraction. GPU array, resumable (skips written npz).
#SBATCH --job-name=news-dwpose
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-15%4
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
export RGB_DIR=/scratch/pmc099/pprasad/news/rgb_clips
export OUT_DIR=/scratch/pmc099/pprasad/news/dwpose_clips
export CLIP_SUFFIX=_signer.mp4
export DWPOSE_DEVICE=cuda
mkdir -p "$OUT_DIR"
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 16 --device cuda --mode balanced
echo "shard $SLURM_ARRAY_TASK_ID done; total npz = $(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)"
