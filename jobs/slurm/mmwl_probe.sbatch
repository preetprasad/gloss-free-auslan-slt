#!/bin/bash -l
# MM-WLAuslan layout probe (evidence-first, before writing the DWPose extractor).
# Fetches the SMALL Valid Kinect_F rgb.zip, lists its internal structure without full
# extraction, and dumps the pose-pkl key format + one array shape so we know:
#   (a) rgb.zip = per-sign videos or frame folders? naming convention?
#   (b) how the gloss label is encoded in the sample id
#   (c) shipped pose keypoint count (confirm 136 AlphaPose vs needed 133 DWPose)
#SBATCH --job-name=mmwl-probe
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
RC="rclone --config /group/pmc099/pprasad/code/rclone.conf"
DEST=/scratch/pmc099/pprasad/mmwl
mkdir -p "$DEST"
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

echo "=== 1) fetch Valid Kinect_F rgb.zip (2.4GB) ==="
$RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Valid/Kinect_F/rgb.zip" "$DEST/valid_kinectF_rgb.zip" --progress 2>&1 | tail -3

echo "=== 2) rgb.zip internal listing (first 40) ==="
unzip -l "$DEST/valid_kinectF_rgb.zip" | head -40
echo "=== rgb.zip entry count + extensions ==="
unzip -l "$DEST/valid_kinectF_rgb.zip" | awk '{print $4}' | grep -oE "\.[A-Za-z0-9]+$" | sort | uniq -c | sort -rn | head

echo "=== 3) fetch + inspect one small pose pkl (Test cam1 is smallest) ==="
$RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Annotation/Pose/Test/pose_cam1.pkl" "$DEST/test_pose_cam1.pkl" 2>&1 | tail -2 || \
$RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Annotation/Pose/Train+Valid/pose_cam1.pkl" "$DEST/tv_pose_cam1.pkl" 2>&1 | tail -2
python - "$DEST" <<'PY'
import sys, gzip, pickle, glob, numpy as np
d = sys.argv[1]
pk = sorted(glob.glob(d+"/*pose_cam1.pkl"))
if not pk: print("no pose pkl fetched"); sys.exit()
p = pk[0]
try:
    obj = pickle.load(gzip.open(p,"rb"))
except Exception:
    obj = pickle.load(open(p,"rb"))
print("pose obj type:", type(obj).__name__, "n=", len(obj) if hasattr(obj,"__len__") else "?")
if isinstance(obj, dict):
    ks = list(obj)[:8]
    print("sample keys (gloss encoding?):"); [print("  ", k) for k in ks]
    v = obj[ks[0]]
    def shape_of(x, depth=0):
        import numpy as _np
        if isinstance(x,_np.ndarray): return x.shape
        if isinstance(x,(list,tuple)) and x: return f"{type(x).__name__}[{len(x)}] -> {shape_of(x[0],depth+1)}"
        if isinstance(x,dict):
            kk=list(x)[0]; return f"dict{list(x)[:3]} -> {shape_of(x[kk],depth+1)}"
        return type(x).__name__
    print("first value shape:", shape_of(v))
PY

echo "=== 4) labels/split location scan ==="
$RC lsl "auslan_drive:Honours Datasets/MM-WLAuslan/Annotation" 2>&1 | head
echo "MMWL_PROBE_DONE"
