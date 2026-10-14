# Wave 12f (two pose measurements, no training): paths, code pins and gates shared by
#   kaya/w12f_stage.sbatch -> kaya/w12f_extract.sbatch -> kaya/w12f_probe.sbatch
# Sourced by absolute path from the shared tree AFTER kaya/_w12lib.sh (which sets G, W12, M, pin_pose and cds into $W12/code).
# Pre-registration: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md
B=$G/scratch_backup/dwpose                      # production DWPose (dwpose_clips), signer videos (rgb_signer), on /group
MM=$G/scratch_backup/mmwl                       # MM-WLAuslan DWPose and labels (used by wave 7, job 31219)
W12F=$W12/w12f
RAW=$W12F/raw                                   # the three files fetched from Preet's Drive mirror
ISO=$W12F/rgb_isolated/New_Gloss_RGB_Data       # 3,209 isolated-sign clips <Video_ID>.mp4
DW_ISO=$W12F/dwpose_isolated                    # their DWPose, <Video_ID>.npz
P3D=$W12F/pose3d                                # measurement 2: <clip>.npz with KPS and Z
OUT=$W12F/out
WK=$G/code/results/dwpose_experiment/dwpose_extract_worker.py      # the production extraction worker, unchanged
POSE3D_NAME=rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx
POSE3D_ONNX=$W12F/models/$POSE3D_NAME
POSE3D_URL=https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/$POSE3D_NAME   # the URL rtmlib 0.0.16 itself uses
POSE3D_SHA256=4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd      # LFS object id on the model page, 369,330,857 bytes
N_ISO=3209; N_SIGNER=14039; N_SPAN_CLIPS=929

fail () { echo "FAIL: $*"; exit 1; }

# The code this experiment runs must be the code that was verified locally (md5 written by kaya/w12f_dryrun.sh --pins).
w12f_code_gate () {
  md5sum -c - <<'EOF' || fail "the experiment's code differs from the files that were verified locally"
8bb68543b3393965f9614138b4c3a523  tools/continuous_id_probe.py
d99bc83156b1c1367bc5ba863dc0b740  tools/w12f_probe.py
54ec8e25b62a6b1f7fc3dc5962f23323  tools/w12f_locate.py
93d4470117f34a7b7129da980607bfd6  tools/w12f_extract3d.py
EOF
  echo "69d8bd99cd79758fd0399c378ae25c3d  $WK" | md5sum -c - || fail "the production DWPose worker changed"
  if [ -f "$W12/code.md5" ]; then      # compiled files are not code: a .pyc that Python rewrites must not stop the chain
    grep -v '__pycache__' "$W12/code.md5" | md5sum --quiet -c - || fail "snapshot differs from $W12/code.md5"
    echo "snapshot matches $W12/code.md5"
  fi
  python tools/w12f_locate.py self_test && python tools/w12f_probe.py self_test && python tools/w12f_extract3d.py --self_test \
    || fail "a self-test failed on this node"
}

# onnxruntime falls back to the CPU without an error when its CUDA libraries do not load (0.2 frames per second
# instead of 20): open the cached person detector on the GPU and require that it really is there.
w12f_gpu_gate () {
  local SP; SP=$(python -c "import site;print(site.getsitepackages()[0])")
  export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
  python - <<'PY' || fail "onnxruntime cannot run on the GPU of this node"
import os, onnxruntime as ort
p = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx")
assert os.path.exists(p), "the cached person detector is missing: " + p
s = ort.InferenceSession(p, providers=["CUDAExecutionProvider"])
assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
print("onnxruntime", ort.__version__, "on", s.get_providers()[0])
PY
}
