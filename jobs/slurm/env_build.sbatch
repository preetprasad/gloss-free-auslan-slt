#!/bin/bash -l
#SBATCH --job-name=env-build
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
mkdir -p /group/pmc099/pprasad/code/logs
module purge
module load Anaconda3/2024.06
python -m venv /group/pmc099/pprasad/envs/auslan
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install rtmlib opencv-python-headless numpy sacrebleu jiwer
# rtmlib pulls CPU onnxruntime which SHADOWS the GPU build; purge and pin the CUDA-12 build.
# (unpinned onnxruntime-gpu == 1.28 needs CUDA 13 -> libcublasLt.so.13 missing on Kaya)
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
pip install "onnxruntime-gpu==1.19.2"
python - <<'PY'
import torch, onnxruntime, rtmlib, cv2, sacrebleu, numpy
print("torch", torch.__version__)
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
print("ENV_OK")
PY
