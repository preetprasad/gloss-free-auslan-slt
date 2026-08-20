#!/bin/bash -l
# One-frame, detector-bypassed HRNet-W48 vs DWPose compatibility probe.
# This is deliberately CPU-only: it validates layout/coordinates before any
# costly cross-estimator extraction or translation retraining is considered.
#SBATCH --job-name=mmpose-hrnet-frame
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail

module purge
module load python/3.9.22 cuda/12.6.3
source /group/pmc099/pprasad/envs/mmpose/bin/activate

CODE=/group/pmc099/pprasad/code/phase2_baseline
MMP=/group/pmc099/pprasad/envs/mmpose/lib/python3.9/site-packages/mmpose/.mim
MODELS=/group/pmc099/pprasad/models/mmpose
VIDEO=/scratch/pmc099/pprasad/dwpose/rgb_clips/Signer/video_1_15_signer.mp4
DWPOSE=/scratch/pmc099/pprasad/dwpose/dwpose_clips/video_1_15.npz
CONFIG=$MMP/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-256x192.py
WEIGHTS=$MODELS/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth
OUT=/group/pmc099/pprasad/results/mmpose_hrnet_w48_video_1_15_frame0.json

mkdir -p "$MODELS" "$(dirname "$OUT")"
if [[ ! -s "$WEIGHTS" ]]; then
  curl --fail --location --retry 3 \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth \
    --output "$WEIGHTS"
fi

cd "$CODE"
python tools/mmpose_crossarch_probe.py \
  --video "$VIDEO" --dwpose_npz "$DWPOSE" \
  --config "$CONFIG" --checkpoint "$WEIGHTS" \
  --frame_index 0 --device cpu --output_json "$OUT"
echo "MMPOSE_HRNET_FRAME_PROBE_DONE"
