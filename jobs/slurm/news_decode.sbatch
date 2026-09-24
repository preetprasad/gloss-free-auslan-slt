#!/bin/bash -l
# NEWS decoding test (length penalty). Rules: results/PREREG_WAVE4_2026-09-22.md, "Addendum, 24 Sep 16:10 AWST",
# written before any tuned-decode test output existed. Decode only (CPU). Task = (arm, seed):
#   0-4 nA s42/101/202/303/404 (val + test)  |  5-9 nSA s42/101/202/303/404 (test; val sweeps exist in news_ens/)
# Each model alone, beam 10, LP sweep; the analysis reads test at the model's own VALIDATION-argmax LP.
#SBATCH --job-name=news-decode
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/group/pmc099/pprasad; M=$G/scratch_backup/dwpose/manifests; RN=$G/wave4/runs_news; OUT=$G/news_decode; mkdir -p "$OUT"
cd "$G/seedens/code" || { echo "FAIL: snapshot missing"; exit 1; }
echo "node $(hostname)"; md5sum tools/ensemble_mbr_wer.py slt/preproc_guard.py ensemble_decode.py
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202 303 404); ARMS=(nA nSA)
arm=${ARMS[$((T / 5))]}; s=${SEEDS[$((T % 5))]}; CK=$RN/${arm}_s$s/best.pt
[ -s "$CK" ] || { echo "FAIL: missing $CK"; exit 1; }
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --max_src_frames 512"
LPS=0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,2.0
dec () {   # dec <manifest> <split>
  local d="$OUT/${arm}_s${s}_$2"
  python tools/ensemble_mbr_wer.py --ckpts "$CK" --eval_manifest "$1" $POSE --beam 10 --lps "$LPS" --device cpu \
    --dump_dir "$d" > "$d.log" 2>&1 || { echo "FAIL: decode $arm $s $2"; tail -20 "$d.log"; exit 1; }
  echo "done $arm s$s $2"
}
if [ "$arm" = nA ]; then dec "$M/news_val.tsv" val; fi
dec "$M/news_test.tsv" test
touch "$OUT/${arm}_s${s}.DONE"; echo "NEWS_DECODE_DONE $arm $s"
