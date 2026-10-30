#!/bin/bash -l
# W4-NEWS: does the adopted recipe (1.1:1 stitch + pose-aug) lift News above the published 2.11-2.81 band?
# STAGE 2: seeds 303 / 404 only, after Stage 1 passed its futility screen (nSA - nA 3-seed mean beam-5 val +0.068).
#   even task: nA  = News real only            odd task: nSA = News + first 10,614 synthetic rows + pose-aug
# The synthetic pool is conversational-register by construction (only 283/9,669 News sentences clear
# cov>=0.9); this is NOT a News-conditioned corpus. Futility: 3-seed mean beam-5 VAL delta nSA-nA <= 0.
# Pre-registration: results/PREREG_WAVE4_2026-09-22.md
#SBATCH --job-name=w4-news-s2
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-3
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w4lib.sh
pin_pose   # Stage-1 omitted this (audit 22 Sep); the pinned clip is unread by News manifests, but the gate is required
S=(303 303 404 404); s=${S[$SLURM_ARRAY_TASK_ID]}
MF=$W4/manifests; RUNS=$W4/runs_news; mkdir -p "$RUNS"
assert_group "$M/news_train.tsv" "$MF/news_sa_train.tsv" "$M/news_val.tsv" "$M/news_test.tsv"
[ "$(wc -l < "$MF/news_sa_train.tsv")" = 20263 ] || { echo "FAIL: news_sa_train is not 9,649 + 10,614"; exit 1; }
if [ $((SLURM_ARRAY_TASK_ID % 2)) = 0 ]; then
  w4run nA "$RUNS" "$M/news_train.tsv" "$M/news_val.tsv" "$M/news_test.tsv" "$s" 150 -- --max_src_frames 512
else
  w4run nSA "$RUNS" "$MF/news_sa_train.tsv" "$M/news_val.tsv" "$M/news_test.tsv" "$s" 150 -- --max_src_frames 512 $AUG_DEF
fi
echo "W4_NEWS_DONE"
