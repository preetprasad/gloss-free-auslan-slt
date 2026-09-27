#!/bin/bash -l
# Tracker v2, step 1: re-extract all 14,041 Communication clips keeping EVERY detected person (KP_ALL / SC_ALL /
# DET_BOX), plus the v1 tracker's outputs (KPS / BBOX / CHOSEN / NDET) computed by the same imported functions.
# After this, any tracker can be built and compared offline without re-running DWPose. CPU partition: the gpu queue
# does not start before 28 Sep. The worker skips finished clips, so a timeout or resubmission loses nothing.
# Why: results/switch_video_89_193/README.md (v1 right in 91% of overrides; fails when the signer bends out of view).
#SBATCH --job-name=ext-alldet
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-47
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
B=/group/pmc099/pprasad/scratch_backup/dwpose
WK=/group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker_alldet.py
python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
export RGB_DIR=$B/rgb_signer OUT_DIR=$B/dwpose_alldet CLIP_SUFFIX=_signer.mp4 ORT_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$OUT_DIR"
echo "videos available: $(find "$RGB_DIR" -name '*_signer.mp4' | wc -l)"
# every clip in the train/val/test manifests must have its video (14,039; the spreadsheet's 14,041 rows include 2 clips
# no manifest uses -- job 31458 failed on a hard-coded 14,041)
python - "$B" <<'PY2'
import os, sys
B = sys.argv[1]; have = set()
for r, _, fs in os.walk(B + "/rgb_signer"):
    have.update(f[:-len("_signer.mp4")] for f in fs if f.endswith("_signer.mp4"))
need = set()
for s in ("train", "val", "test"):
    need |= {l.split("\t")[0].rsplit("/", 1)[1][:-4] for l in open(f"{B}/manifests/{s}.tsv") if "\t" in l}
miss = need - have
print(f"manifest clips {len(need)} | videos {len(have)} | missing {len(miss)}")
assert len(need) == 14039 and not miss, "manifest clips without a video"
PY2
md5sum "$WK" /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker_tracked.py
SECONDS=0
python "$WK" --shard "$SLURM_ARRAY_TASK_ID" --nshards 48 --device cpu --mode balanced
echo "shard $SLURM_ARRAY_TASK_ID done in ${SECONDS}s; total npz = $(find "$OUT_DIR" -name '*.npz' | wc -l)"
