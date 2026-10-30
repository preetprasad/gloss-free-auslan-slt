#!/bin/bash -l
# MM-WLAuslan DWPose extraction — VALID PILOT (validates the whole chain cheaply before Train).
# Reuses the Valid rgb.zip already fetched by the probe. rgb/<id>_kf_rgb.mp4 -> DWPose (T,133,3)
# npz keyed by bare <id> (matches Labels+Split/Valid.json keys). GPU array; shard 0 unzips.
#SBATCH --job-name=mmwl-dw-valid
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
export LD_LIBRARY_PATH=$(python -c "import os,glob,nvidia;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))"):$LD_LIBRARY_PATH
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
export ORT_THREADS=3

M=/scratch/pmc099/pprasad/mmwl
ZIP=$M/valid_kinectF_rgb.zip
RGB=$M/rgb_clips/valid
OUT=$M/dwpose_clips/valid
mkdir -p "$RGB" "$OUT"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  if [ ! -f "$RGB/UNZIP_DONE" ]; then
    echo "[shard0] unzipping $ZIP"; unzip -o -q "$ZIP" -d "$RGB"; touch "$RGB/UNZIP_DONE"
  fi
else
  for i in $(seq 1 120); do [ -f "$RGB/UNZIP_DONE" ] && break; sleep 10; done
fi
echo "shard $SLURM_ARRAY_TASK_ID: $(find "$RGB" -name '*_kf_rgb.mp4' | wc -l) mp4s visible"

export RGB_DIR="$RGB" OUT_DIR="$OUT" CLIP_SUFFIX="_kf_rgb.mp4" DWPOSE_DEVICE=cuda
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 4 --mode balanced --device cuda
echo "shard $SLURM_ARRAY_TASK_ID DONE: $(ls "$OUT" | wc -l) npz"
