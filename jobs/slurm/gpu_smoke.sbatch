#!/bin/bash -l
#SBATCH --job-name=gpu-smoke
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
# onnxruntime and onnxruntime-gpu share one package dir; purge both, reinstall GPU only.
pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
pip install -q "onnxruntime-gpu==1.19.2" 2>&1 | tail -1   # CUDA 12 + cuDNN 9 build (matches torch cu121)
# onnxruntime-gpu needs CUDA/cuDNN 12 libs; torch's bundled nvidia-* wheels provide them.
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import onnxruntime as ort, numpy as np
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP still missing"
from rtmlib import Wholebody
m = Wholebody(mode="balanced", backend="onnxruntime", device="cuda")
provs = None
for attr in ("pose_model", "pose", "det_model"):
    o = getattr(m, attr, None)
    if o is not None and hasattr(o, "session"):
        provs = o.session.get_providers(); break
print("pose session providers:", provs)
frame = (np.random.rand(480, 640, 3) * 255).astype("uint8")
kp, sc = m(frame)
print("rtmlib run; kpts:", None if kp is None else np.asarray(kp).shape)
assert provs and "CUDAExecutionProvider" in provs, "FELL BACK TO CPU (CUDA not active)"
print("GPU_SMOKE_OK")
PY
