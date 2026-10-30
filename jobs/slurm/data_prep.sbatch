#!/bin/bash -l
#SBATCH --job-name=data-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
DE=/scratch/pmc099/pprasad/dwpose
mkdir -p "$DE" && cd "$DE"
RC=/usr/bin/rclone
CFG=/group/pmc099/pprasad/code/rclone.conf
echo "fetching Communication Signer.zip from Drive..."
$RC --config "$CFG" --transfers 8 --drive-chunk-size 128M copy \
  "auslan_drive:Honours Datasets/Auslan-Daily/Communication/Signer Only Video Clip/Signer.zip" .
ls -la Signer.zip
echo "unzipping..."
unzip -q Signer.zip -d rgb_clips
echo "extracted $(find rgb_clips -iname '*.mp4' | wc -l) mp4s"
rm -f Signer.zip
echo "DATA_PREP_OK"
