#!/bin/bash -l
# =============================================================================
# Mirror datasets from your Google Drive to Kaya storage (run on a LOGIN/DATA node
# — they have network; compute nodes may not). Reuses your rclone remote.
#
# Storage policy (Kaya):
#   /group/pmc099/pprasad   1 TB, DURABLE  -> pose, annotations, code, results
#   /scratch/pmc099/pprasad large, PURGED after 21 days -> bulky RGB, active job data
# Mirroring the FULL ~510 GiB into /group would use half the project quota, so
# fetch per experiment. Everything below is resumable.
#
# Setup once (copy the rclone binary + config from your Mac):
#   scp ~/Honours/tools/rclone/rclone ~/Honours/data/rclone.conf \
#       <user>@kaya.hpc.uwa.edu.au:/group/pmc099/pprasad/code/
#
# Usage:  bash mirror_datasets_kaya.sh <preset>
#   dwpose   Communication Signer RGB (17.4 GB)  -> /scratch  (for DWPose extraction)
#   pose     Auslan-Daily + MM-WLAuslan pose      -> /group    (durable, reused)
#   phoenix  PHOENIX-2014T tarball                -> /scratch  (GFSLT-VLP repro)
#   full     everything (~510 GiB)                -> /group    (checks quota first)
# =============================================================================
set -uo pipefail
RC="/group/pmc099/pprasad/code/rclone"
CFG="/group/pmc099/pprasad/code/rclone.conf"
REMOTE="auslan_drive:Honours Datasets"
GROUP="/group/pmc099/pprasad/data"
SCRATCH="/scratch/pmc099/pprasad/data"
rc(){ "$RC" --config "$CFG" --progress --transfers 4 --checkers 8 "$@"; }
mkdir -p "$GROUP" "$SCRATCH"

case "${1:-}" in
  dwpose)
    mkdir -p "$SCRATCH/dwpose"
    rc copy "$REMOTE/Auslan-Daily/Communication/Signer Only Video Clip/Signer.zip" "$SCRATCH/dwpose/"
    echo "unzip:  cd $SCRATCH/dwpose && unzip -q Signer.zip -d rgb_clips && rm Signer.zip" ;;
  pose)
    mkdir -p "$GROUP/auslan_pose" "$GROUP/mmwl_pose"
    # Auslan-Daily HALPE pose annotations
    rc copy "$REMOTE/Auslan-Daily/Communication/Pose Annotation/SP_Annotation" "$GROUP/auslan_pose/Communication"
    rc copy "$REMOTE/Auslan-Daily/News/Pose Annotation/SP_Annotation" "$GROUP/auslan_pose/News"
    # MM-WLAuslan pose (all cameras, Train+Valid + test splits)
    rc copy "$REMOTE/MM-WLAuslan/Annotation/Pose" "$GROUP/mmwl_pose" ;;
  phoenix)
    mkdir -p "$SCRATCH/phoenix"
    rc copy "auslan_drive:PHOENIX-2014T/phoenix-2014-T.v3.tar.gz" "$SCRATCH/phoenix/" \
      || rc copy "$REMOTE/../PHOENIX-2014T/phoenix-2014-T.v3.tar.gz" "$SCRATCH/phoenix/"
    echo "extract on scratch:  tar -xzf $SCRATCH/phoenix/phoenix-2014-T.v3.tar.gz -C $SCRATCH/phoenix/" ;;
  full)
    echo "Full mirror ~510 GiB. /group quota is 1 TB shared for pmc099."
    echo "Current /group usage:"; du -sh "$GROUP" 2>/dev/null
    read -p "Proceed mirroring the full official set into /group? [y/N] " ok
    [ "$ok" = "y" ] || { echo "aborted"; exit 0; }
    rc copy "auslan_drive:Honours_Datasets_Official_Mirror_2026" "$GROUP/official_mirror" ;;
  *)
    echo "usage: bash mirror_datasets_kaya.sh {dwpose|pose|phoenix|full}"; exit 1 ;;
esac
echo "done: $1"
