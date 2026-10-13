#!/bin/bash -l
# WAVE 12e, step 1 of 2: DINOv2 crop features that the three RGB-fusion members of the reported ensemble need.
# Pre-registration: results/PREREG_WAVE12E_NONSIGNER_ENSEMBLE_2026-10-13.md
#   neg  : the 1,678 dev+test Communication Non_Signer crops, pose = person-tracked dwpose_nonsigner -> rgb_feats_nonsigner
#   trk  : the 1,600 validation+test signer clips re-cropped on the person-tracked pose (arm ii)     -> rgb_feats_tracked
#   gate : 100 signer clips (50 test + 50 validation, seed 0) on the PRODUCTION pose, to be compared
#          with the stored August features by w12e_decode.sbatch (gate G-RGB)                         -> rgb_gate
# The extractor is the one that made the stored signer features, UNMODIFIED and md5-pinned (--frames 24, CPU, as
# job 1075349). It names a clip basename.replace("_signer.mp4", ""), so every input is a symlink
# <pose stem>_signer.mp4 in a node-local folder. Nothing is downloaded: DINOv2-small is read from the cache it was
# loaded from in August (revision ed25f3a3, HF_HUB_OFFLINE=1). No train crop is touched.
#   cd /group/pmc099/pprasad/code/phase2_baseline && sbatch kaya/w12e_extract.sbatch
#SBATCH --job-name=w12e-extract
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -euo pipefail
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; B=$G/scratch_backup/dwpose; M=$B/manifests; W12=$G/wave12; W=$W12/w12e
TOOL=$W12/code/tools/w12e_nonsigner_s6.py; TOOL_MD5=5e934edae55f20e0142016332cbc7b8b   # the file tools/w12e_dryrun.py tested
EXT=$G/code/results/dwpose_experiment/extract_rgb_feats.py
DEC=$G/ens_ablation/code            # frozen decode snapshot of job 120153; the tool imports slt/ from it
NSH=8
echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID of $NSH"

# ---- code gates
[ "$(md5sum "$EXT" | cut -d' ' -f1)" = e3797b9cdcf3665d3e6ee4effee9c910 ] || { echo "FAIL: extractor differs from the one verified locally"; exit 1; }
( cd "$DEC" && md5sum -c --quiet "$G/ens_ablation/code.md5" ) || { echo "FAIL: decode snapshot differs from its md5 list"; exit 1; }
[ "$(md5sum "$TOOL" | cut -d' ' -f1)" = "$TOOL_MD5" ] || { echo "FAIL: $TOOL is missing or is not the reviewed file"; exit 1; }
export W12E_CODE=$DEC OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SNAP=$HF_HOME/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056
[ -s "$SNAP/model.safetensors" ] || { echo "FAIL: DINOv2-small revision ed25f3a3 is not in the cache (no download is allowed)"; exit 1; }
echo "DINOv2 weights: $(readlink -f "$SNAP/model.safetensors")"
python -c "import cv2, torch, transformers; print('cv2', cv2.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

# ---- inputs: lists from the registered tables, then node-local symlinks (no /group inodes, no race between tasks)
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
python "$TOOL" manifest --lists_only --expected "$B/nonsigner_expected.tsv" --neg_dir "$B/dwpose_nonsigner" \
  --val_manifest "$M/val.tsv" --test_manifest "$M/test.tsv" \
  --trk_val "$B/manifests_tracked/val.tsv" --trk_test "$B/manifests_tracked/test.tsv" \
  --neg_mp4_dir "$B/rgb_nonsigner" --sig_mp4_dir "$B/rgb_signer" --out_dir "$STAGE"
for k in neg trk gate; do
  mkdir "$STAGE/$k"
  while IFS=$'\t' read -r src name; do ln -s "$src" "$STAGE/$k/$name"; done < "$STAGE/ext_$k.tsv"
done
[ "$(ls "$STAGE/neg" | wc -l)" -eq 1678 ] && [ "$(ls "$STAGE/trk" | wc -l)" -eq 1600 ] && [ "$(ls "$STAGE/gate" | wc -l)" -eq 100 ] \
  || { echo "FAIL: staged clip counts"; exit 1; }

run () {   # <set> <pose dir> <output dir>
  mkdir -p "$3"
  echo "=== $1: pose $2 -> $3"
  RGB_DIR="$STAGE/$1" DWPOSE_DIR="$2" OUT_DIR="$3" python "$EXT" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSH" --frames 24
}
run gate "$B/dwpose_clips"     "$W/rgb_gate"
run neg  "$B/dwpose_nonsigner" "$W/rgb_feats_nonsigner"
run trk  "$B/dwpose_tracked"   "$W/rgb_feats_tracked"
# the extractor skips a clip silently when its pose or video cannot be read; w12e_decode.sbatch checks every file
touch "$W/DONE_extract_$SLURM_ARRAY_TASK_ID"
echo "W12E_EXTRACT_DONE shard $SLURM_ARRAY_TASK_ID"
