#!/bin/bash -l
# Stage Communication Non_Signer.zip (14.4 GB, approved by Preet 2026-09-22) from Drive to /group, and verify every
# one of the 14,803 Communication non-signer crops in Signer_Detection.xlsx is present (list built locally).
#SBATCH --job-name=stage-nonsigner
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
R="/usr/bin/rclone --config /group/pmc099/pprasad/code/rclone.conf"
DST=/group/pmc099/pprasad/scratch_backup/dwpose/rgb_nonsigner; mkdir -p "$DST"
SRC="auslan_drive:Honours Datasets/Auslan-Daily/Communication/Signer Only Video Clip/Non_Signer.zip"
echo "[1/3] pulling Non_Signer.zip"; $R copyto "$SRC" "$DST/Non_Signer.zip" --transfers 4 --checkers 8 --stats 120s --stats-one-line
ls -l "$DST/Non_Signer.zip"
echo "[2/3] extracting"; export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
cd "$DST" && unzip -q -o Non_Signer.zip && rm -f Non_Signer.zip
echo "[3/3] verifying"
python3 - <<'PY'
import os, glob
dst = "/group/pmc099/pprasad/scratch_backup/dwpose/rgb_nonsigner"
have = {os.path.basename(p): p for p in glob.glob(dst + "/**/*_nonsigner.mp4", recursive=True)}
need = [l.split("\t")[0] for l in open("/group/pmc099/pprasad/scratch_backup/dwpose/nonsigner_expected.tsv")]
miss = [n for n in need if n not in have]
print(f"nonsigner mp4 found {len(have)} | expected {len(need)} | missing {len(miss)} | extra {len(set(have) - set(need))}")
print("first missing:", miss[:10])
assert not miss, "expected non-signer crops are missing"
print("STAGE_NONSIGNER_OK")
PY
