#!/bin/bash -l
# Stage the shipped AlphaPose HALPE-136 pose on Kaya (never uploaded before — only the DWPose
# re-extraction was). Needed for the estimator-vs-norm disambiguation cell (AlphaPose + frame-norm).
# CPU-only: fetch the 90 per-video .pt from Drive, extract per-clip npz, build MATCHED manifests
# (identical clips/text/split as the DWPose manifests, just pointing at the AlphaPose npz).
#SBATCH --job-name=ap-stage
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
RC=/group/pmc099/pprasad/code/rclone.conf
AP=/scratch/pmc099/pprasad/alphapose
mkdir -p "$AP/pt" "$AP/clips" "$AP/manifests"
SRC="auslan_drive:Honours Datasets/Auslan-Daily/Communication/Pose Annotation/SP_Annotation/SP_Annotation_V1"

echo "=== fetch shipped AlphaPose HALPE .pt (90 videos) ==="
rclone --config "$RC" copy "$SRC" "$AP/pt" -P --transfers 8
echo "pt files: $(ls "$AP/pt" | wc -l)"

cd /group/pmc099/pprasad/code/phase2_baseline
echo "=== extract per-clip HALPE npz (KPS 136,3 + BBOX) ==="
python tools/extract_clips.py --pose_root "$AP/pt" --out_dir "$AP/clips" --num_kp 136

echo "=== build matched AlphaPose manifests from the DWPose split (same clips/text) ==="
python - <<'PY'
import os
DE="/scratch/pmc099/pprasad/dwpose/manifests"; AC="/scratch/pmc099/pprasad/alphapose/clips"
AM="/scratch/pmc099/pprasad/alphapose/manifests"; os.makedirs(AM,exist_ok=True)
for split in ("train","val","test"):
    kept=miss=0; out=[]
    for ln in open(f"{DE}/{split}.tsv"):
        p,_,t=ln.rstrip("\n").partition("\t")
        ap=os.path.join(AC, os.path.basename(p))
        if os.path.exists(ap): out.append(f"{ap}\t{t}"); kept+=1
        else: miss+=1
    open(f"{AM}/ap_{split}.tsv","w").write("\n".join(out)+"\n")
    print(f"{split}: kept {kept}  missing {miss}")
PY
echo "AP_STAGE_DONE"
