#!/bin/bash -l
# Non-signer hard negatives (results/PREREG_NONSIGNER_GATE_2026-09-22.md): person-tracked DWPose on the 1,678 dev+test
# Communication Non_Signer crops. Submit with --dependency=afterany:<staging job>: staging asserts all 14,803 crops and
# may fail on a known release swap in TRAIN (B9), so this job re-checks only what it needs -- all 1,678 dev+test crops.
# Task 0 also runs the extractor gate G5 (B10): re-extract 20 test signer clips, compare with dwpose_tracked.
#SBATCH --job-name=ext-nonsigner
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
B=/group/pmc099/pprasad/scratch_backup/dwpose
WK=/group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_extract_worker_tracked.py
LIST=$B/nonsigner_devtest_ids.txt
awk -F'\t' '$3=="dev"||$3=="test"{sub(/\.mp4$/,"",$1); print $1}' $B/nonsigner_expected.tsv > $LIST.$SLURM_ARRAY_TASK_ID
python - "$B" "$LIST.$SLURM_ARRAY_TASK_ID" <<'PY'
import os, sys
b, lst = sys.argv[1:]
have = set()
for r, _, fs in os.walk(b + "/rgb_nonsigner"):
    have.update(f[:-4] for f in fs if f.endswith(".mp4"))
need = [l.strip() for l in open(lst)]
miss = [n for n in need if n not in have]
print(f"dev+test crops needed {len(need)} | present {len(need) - len(miss)} | missing {len(miss)} {miss[:10]}")
assert len(need) == 1678 and not miss, "dev/test non-signer crops missing"
PY
export RGB_DIR=$B/rgb_nonsigner OUT_DIR=$B/dwpose_nonsigner CLIP_SUFFIX=.mp4 CLIP_LIST=$LIST.$SLURM_ARRAY_TASK_ID DWPOSE_DEVICE=cuda
mkdir -p "$OUT_DIR"
python $WK --shard "$SLURM_ARRAY_TASK_ID" --nshards 4 --device cuda --mode balanced
echo "shard $SLURM_ARRAY_TASK_ID done; nonsigner npz = $(find "$OUT_DIR" -name '*.npz' | wc -l)"

[ "$SLURM_ARRAY_TASK_ID" = 0 ] || exit 0
# ---- G5: same extractor, same settings, 20 test signer clips (seed 0), against the production dwpose_tracked files
G5=$B/nonsigner_g5; mkdir -p $G5
test $(wc -l < $G5/ids.txt) -eq 20   # 20 test signer ids, random.Random(0).sample of the official test manifest (copied)
RGB_DIR=$B/rgb_signer OUT_DIR=$G5/npz CLIP_SUFFIX=_signer.mp4 CLIP_LIST=$G5/ids.txt \
    python $WK --shard 0 --nshards 1 --device cuda --mode balanced
python - "$B" <<'PY'
import numpy as np, sys
b = sys.argv[1]
ch_eq = ch_n = 0; dxy = []
for cid in open(b + "/nonsigner_g5/ids.txt").read().split():
    a, r = np.load(f"{b}/nonsigner_g5/npz/{cid}.npz"), np.load(f"{b}/dwpose_tracked/{cid}.npz")
    assert a["KPS"].shape == r["KPS"].shape, cid
    ch_eq += int((a["CHOSEN"] == r["CHOSEN"]).sum()); ch_n += a["CHOSEN"].size
    m = (a["KPS"][..., 2] >= 0.3) & (r["KPS"][..., 2] >= 0.3)
    dxy.extend(np.linalg.norm(a["KPS"][..., :2] - r["KPS"][..., :2], axis=-1)[m].tolist())
frac, med = ch_eq / ch_n, float(np.median(dxy))
print(f"G5: CHOSEN equal {ch_eq}/{ch_n} = {100 * frac:.2f}% | median |dxy| {med:.3f} px | max {max(dxy):.1f} px")
assert frac >= 0.99 and med < 1.0, "G5 FAILED"
print("G5 PASSED")
PY
