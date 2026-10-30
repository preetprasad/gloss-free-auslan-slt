#!/bin/bash -l
# GFSLT-VLP environment build (Phase-A prerequisite). The repo pins an OLD stack
# (py3.8, torch 1.11 / cu113, torchtext 0.12, tf 2.9) — build it in an isolated conda env,
# NOT the auslan venv. This is the finicky step; babysit the first run for wheel/CUDA conflicts.
# CPU node is fine for the install; the MBart trim also just needs internet.
#SBATCH --job-name=gfslt-env
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
REPO=/group/pmc099/pprasad/code/GFSLT-VLP
ENVDIR=/group/pmc099/pprasad/envs/gfslt
cd "$REPO"

if [ ! -d "$ENVDIR" ]; then
  conda create -y -p "$ENVDIR" python=3.8
fi
source activate "$ENVDIR"

# torch 1.11 + cu113 (matches torchvision/torchtext pins in requirements.txt)
pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-cache-dir -r requirements.txt || {
  echo "[warn] some pins may conflict on this cluster — resolve the offenders (tensorflow/torchtext) by hand"; }

echo "=== MBart weights: trim facebook/mbart-large-cc25 (HF download, compute node has internet) ==="
export HF_HOME=/scratch/pmc099/pprasad/hf_cache
python trim_model.py 2>&1 | tail -20 || echo "[note] if trim_model.py needs the raw mbart first, HF will fetch it"

echo "=== sanity ==="
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
ls -R pretrain_models 2>/dev/null | head -30
echo "GFSLT_ENV_BUILD_DONE"
