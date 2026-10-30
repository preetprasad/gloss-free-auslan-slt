#!/bin/bash -l
# MM-WLAuslan DWPose extraction — TRAIN (full ISLR pretraining set). Fetches Train Kinect_F
# rgb.zip from Preet's Drive mirror -> scratch, unzips, DWPose -> (T,133,3) npz keyed by <id>.
# GPU array; shard 0 fetches+unzips, others wait on the marker.
#SBATCH --job-name=mmwl-dw-train
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
export LD_LIBRARY_PATH=$(python -c "import os,glob,nvidia;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))"):$LD_LIBRARY_PATH
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
export ORT_THREADS=3
RC="rclone --config /group/pmc099/pprasad/code/rclone.conf"

M=/scratch/pmc099/pprasad/mmwl
ZIP=$M/train_kinectF_rgb.zip
RGB=$M/rgb_clips/train
OUT=$M/dwpose_clips/train
mkdir -p "$RGB" "$OUT"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  if [ ! -f "$RGB/UNZIP_DONE" ]; then
    [ -f "$ZIP" ] || $RC copyto "auslan_drive:Honours Datasets/MM-WLAuslan/Train/Kinect_F/rgb.zip" "$ZIP" 2>&1 | tail -2
    echo "[shard0] unzipping"; unzip -o -q "$ZIP" -d "$RGB"; touch "$RGB/UNZIP_DONE"
  fi
else
  for i in $(seq 1 360); do [ -f "$RGB/UNZIP_DONE" ] && break; sleep 20; done
fi
echo "shard $SLURM_ARRAY_TASK_ID: $(find "$RGB" -name '*_kf_rgb.mp4' | wc -l) mp4s"

export RGB_DIR="$RGB" OUT_DIR="$OUT" CLIP_SUFFIX="_kf_rgb.mp4" DWPOSE_DEVICE=cuda
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 8 --mode balanced --device cuda
echo "shard $SLURM_ARRAY_TASK_ID DONE: $(ls "$OUT" | wc -l) npz"
