#!/bin/bash -l
# W11-NEWS: input frame limit for the News subset, chosen on VALIDATION only (512 frames was taken from the test
# scores of single runs, while validation preferred 256). One arm per array task: nF256 / nF512, News real clips
# only, 150 epochs, seeds 42/101/202, all in this one job, deterministic kernels.
# Pre-registration: results/PREREG_WAVE11_VALSELECT_2026-10-09.md
#SBATCH --job-name=w11-news
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-5
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh
F=(256 512); S=(42 101 202)
f=${F[$((SLURM_ARRAY_TASK_ID % 2))]}; s=${S[$((SLURM_ARRAY_TASK_ID / 2))]}
RUNS=$W11/runs_news; mkdir -p "$RUNS"
assert_group "$M/news_train.tsv" "$M/news_val.tsv" "$M/news_test.tsv"
[ "$(wc -l < "$M/news_train.tsv")" = 9649 ] && [ "$(wc -l < "$M/news_val.tsv")" = 700 ] && [ "$(wc -l < "$M/news_test.tsv")" = 699 ] \
  || { echo "FAIL: manifest sizes"; exit 1; }
echo "=== task $SLURM_ARRAY_TASK_ID: nF${f}, seed $s ==="
w4run "nF${f}" "$RUNS" "$M/news_train.tsv" "$M/news_val.tsv" "$M/news_test.tsv" "$s" 150 -- --max_src_frames "$f"
[ -s "$RUNS/nF${f}_s${s}/beam5_test_best.json" ] && touch "$RUNS/DONE_nF${f}_s${s}"
echo "W11_NEWS_DONE nF${f} s${s}"
