#!/bin/bash -l
# W12A-NEWS read-out (CPU). It decides nothing by itself: every reading is fixed in
# results/PREREG_WAVE12A_NEWSDECOMP_2026-10-13.md. Submit after the array, with its job id:
#   sbatch --dependency=afterany:<array job id> --export=ALL,JOB=<array job id> kaya/w12a_news_decomp_summary.sbatch
# (add W12A_SMOKE=1 to --export to read a smoke run). The read-out needs tools that the wave-11 snapshot does not
# hold (tools/news_signable_recall.py, tools/w12a_news_decomp_readout.py), so it runs from the wave-12 snapshot.
#SBATCH --job-name=w12a-news-sum
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # module and venv set-up are not nounset-safe
set -u
: "${JOB:?submit with --export=ALL,JOB=<array job id of w12a-news>}"
KIND=runs; if [ "${W12A_SMOKE:-0}" = 1 ]; then KIND=smoke; fi
RUNS=$G/wave12/a_news/${KIND}_$JOB; LAB=$G/scratch_backup/mmwl/labels
[ -d "$RUNS" ] || { echo "FAIL: $RUNS does not exist"; exit 1; }

# --- gate: the read-out code is the code that was tested on 13 Oct 2026 (cwd is $G/wave12/code, set by the library)
md5sum -c --quiet - <<'MD5' || { echo "FAIL: a read-out tool differs from the file that was tested locally"; exit 1; }
bacef0b54c52b1e43259ae90a1b180f9  tools/w12a_news_decomp_readout.py
9f844c2bb0b7bc27c594ace469857f6c  tools/news_signable_recall.py
6479eff0871ffead572b5d6d7541ec10  tools/arm_readouts.py
e0a95aa351a0950d9b2af13e8bea4240  tools/seed_decision.py
4928ae2af004c2fadb9c6cfff38545ae  tools/w11_readout.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
MD5
# --- gate: labels and manifests are the files the pinned type counts were measured on
md5sum -c --quiet - <<MD5 || { echo "FAIL: labels or News manifests changed"; exit 1; }
915b3ce577d28b8c6898f940160df922  $LAB/Train.json
a14087ba93a9b511bfac8712085ae074  $LAB/Valid.json
48c2b247159ed5eb9c3a595f6b8fedcb  $M/news_train.tsv
fd31d79eef3a5d6dde5b9f0839d05dbd  $M/news_val.tsv
c11da128f6c46844ff6ce4f3352d894f  $M/news_test.tsv
MD5
# tools/stitch_lexicon.py is shared with other wave-12 work, so it is not md5-pinned: the read-out stops unless it
# reproduces the pinned type counts (934 / 1,821 / 2,379 / 878 / 1,501) and, where nA and nSA reproduce job 7950, the
# recorded signable and unsignable recall.
md5sum tools/stitch_lexicon.py
python tools/w12a_news_decomp_readout.py --self_test

for arm in nA nS nSA; do for s in 42 101 202; do
  [ -e "$RUNS/DONE_${arm}_s$s" ] || echo "MISSING: ${arm}_s$s has no DONE marker"
done; done
echo "===== existing tool tools/w11_readout.py: arm means and paired differences from nA and from nS ====="
echo "      (its SELECTED line is not used: this pre-registration selects nothing)"
python tools/w11_readout.py "$RUNS" nA 0.000001 nS nSA || echo "[WARN] w11_readout.py failed"   # its tie margin must be > 0
echo "===== pre-registered decomposition, R2 rule-7 read-outs and the mandatory News secondary ====="
python -u tools/w12a_news_decomp_readout.py --runs "$RUNS" --labels "$LAB" --news_train "$M/news_train.tsv" \
  --news_val "$M/news_val.tsv" --news_test "$M/news_test.tsv" --boot 1000 --out "$RUNS/w12a_readout.json"
echo "W12A_SUMMARY_DONE"
