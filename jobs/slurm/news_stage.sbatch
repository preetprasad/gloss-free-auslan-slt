#!/bin/bash -l
# DATA-SCALING SWING, step 1/4 — stage the News split (fetch + unzip Signer videos, get labels).
# CPU long pole. Next: DWPose-extract (step 2), build combined Comm+News manifests (3), retrain (4).
#SBATCH --job-name=news-stage
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
RC=/group/pmc099/pprasad/code/rclone.conf
N=/scratch/pmc099/pprasad/news
mkdir -p "$N/rgb_clips" "$N/dwpose_clips" "$N/manifests"

echo "=== verify DWPose runtime (rtmlib + onnxruntime-gpu) before staging 20GB ==="
python -c "import rtmlib, onnxruntime as o; print('rtmlib OK; ORT', o.__version__, o.get_available_providers()[:2])"

echo "=== fetch News Signer.zip + translation xlsx from Drive ==="
rclone --config "$RC" copyto "auslan_drive:Honours Datasets/Auslan-Daily/News/Signer Only Video Clip/Signer.zip" "$N/Signer.zip" -P --multi-thread-streams 4
# translation spreadsheet (News BLEU references) — try the known split-table locations
for P in \
  "Honours Datasets/Auslan-Daily/Dataset Split Table/Sign Language Translation/News/Auslan-Daily_News.xlsx" \
  "Honours Datasets/Auslan-Daily/Dataset Split Tables/Sign Language Translation/News/Auslan-Daily_News.xlsx" ; do
  rclone --config "$RC" copyto "auslan_drive:$P" "$N/Auslan-Daily_News.xlsx" 2>/dev/null && break || true
done
ls -lh "$N/Auslan-Daily_News.xlsx" 2>/dev/null || echo "[warn] News xlsx not at expected path — will locate separately"

echo "=== unzip Signer videos ==="
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -oq "$N/Signer.zip" -d "$N/rgb_clips" || unzip -o "$N/Signer.zip" -d "$N/rgb_clips" | tail -3
echo "=== structure + counts ==="
find "$N/rgb_clips" -maxdepth 3 -type d | head -10
echo "video files: $(find "$N/rgb_clips" -iname '*.mp4' | wc -l)"
echo "sample names:"; find "$N/rgb_clips" -iname '*.mp4' | head -5
echo "NEWS_STAGE_DONE"
