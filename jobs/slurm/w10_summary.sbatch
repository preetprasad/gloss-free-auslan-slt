#!/bin/bash -l
# WAVE 10 readout (CPU): per-seed summaries (w4summary) and the seed-level contrasts the pre-registration reads
# (tools/seed_decision.py): S - A, SC - A, R - A over seeds 42/101/202. Decides nothing; the readings are in
# results/PREREG_WAVE10_SIGNORDER_2026-10-01.md. Submit with --dependency=afterany:<w10-order array id>.
#SBATCH --job-name=w10-summary
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
RUNS=$G/wave10/runs
for s in 42 101 202; do
  for arm in A S SC R; do [ -f "$RUNS/${arm}_s${s}/DONE" ] || echo "MISSING: ${arm}_s${s} (no DONE marker)"; done
  w4summary "$RUNS" "$s" A S SC R
done
F=("$RUNS"/summary_A_S_SC_R_s{42,101,202}.json)
for arm in S SC R; do
  echo "===== $arm - A ====="
  python tools/seed_decision.py --control A --arm "$arm" "${F[@]}"
done
echo "W10_SUMMARY_DONE"
