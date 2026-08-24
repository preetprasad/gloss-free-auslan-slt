#!/bin/bash -l
# PHOENIX-2014T staging (Phase A prerequisite for the GFSLT-VLP reproduction).
# CPU-only long pole: pull the 41.7 GB tar from Drive to scratch and extract it. No GPU.
#SBATCH --job-name=phoenix-stage
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
RC=/group/pmc099/pprasad/code/rclone.conf
DEST=/scratch/pmc099/pprasad/phoenix
mkdir -p "$DEST"
cd "$DEST"

SRC="auslan_drive:Honours Datasets/PHOENIX-2014T/phoenix-2014-T.v3.tar.gz"
if [[ ! -s phoenix-2014-T.v3.tar.gz ]]; then
  echo "=== fetching PHOENIX-2014T tar from Drive (41.7 GB) ==="
  rclone --config "$RC" copyto "$SRC" "$DEST/phoenix-2014-T.v3.tar.gz" -P --multi-thread-streams 4
fi
echo "=== fetched; size: ==="; ls -lh phoenix-2014-T.v3.tar.gz

echo "=== extracting ==="
mkdir -p extracted
tar xzf phoenix-2014-T.v3.tar.gz -C extracted
echo "=== extracted layout (top 2 levels) ==="
find extracted -maxdepth 2 -type d | head -40
echo "PHOENIX_STAGE_DONE"
