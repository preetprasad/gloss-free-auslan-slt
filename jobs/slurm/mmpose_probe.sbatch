#!/bin/bash -l
# Probe whether MMPose (HRNet-W48, ViTPose+ whole-body) installs cleanly on Kaya,
# so we can add cross-architecture estimators to the agreement check. Isolated venv.
#SBATCH --job-name=mmpose-probe
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge
module load python/3.9.22 cuda/12.6.3    # py3.9 avoids the py3.12 pkg_resources/ImpImporter break
python3 -m venv /group/pmc099/pprasad/envs/mmpose
source /group/pmc099/pprasad/envs/mmpose/bin/activate
pip install -q --upgrade pip "setuptools<81" wheel
pip install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -q -U openmim
mim install -q "mmengine"
mim install -q "mmcv>=2.0.1,<2.2.0"   # mim resolves the cu121/torch2.2 prebuilt
mim install -q "mmdet>=3.1.0"
mim install -q "mmpose>=1.3.0"
# Torch 2.2's binary extensions in this environment require the NumPy 1.x ABI.
# Installing unpinned NumPy previously selected 2.x and made imports only
# superficially succeed while emitting an ABI failure warning.
pip install -q --force-reinstall "numpy<2" opencv-python-headless
python - <<'PY'
import cv2
import mmdet
import mmcv
import mmpose
import numpy as np
import torch

array = np.zeros((2, 3), dtype=np.float32)
tensor = torch.from_numpy(array)
assert tensor.numpy().shape == (2, 3)
print(
    "MMPOSE_NUMPY_OK",
    f"mmpose={mmpose.__version__}",
    f"mmcv={mmcv.__version__}",
    f"numpy={np.__version__}",
    f"torch={torch.__version__}",
    f"opencv={cv2.__version__}",
)
PY
echo "PROBE_DONE"
