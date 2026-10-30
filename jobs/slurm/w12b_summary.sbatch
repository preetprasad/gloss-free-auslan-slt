#!/bin/bash -l
# WAVE 12b read-out (CPU): per-seed summaries (w4summary), the seed-level contrast W - A (tools/seed_decision.py) and
# the pre-registered secondaries and verdict (tools/w12b_readout.py, which reuses tools/arm_readouts.py). It decides
# nothing beyond applying the rule fixed in results/PREREG_WAVE12B_WRONGSENSE_2026-10-13.md, and it needs all six runs.
# Submit with --dependency=afterany:<w12b-wrongsense array id>.
#SBATCH --job-name=w12b-summary
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh   # env, G, M, w4summary; cd $G/wave11/code
set -u                                                             # after the library: module/activate are not -u clean
RUNS=$G/wave12b/runs; K=$G/wave12/code/tools
md5sum -c - <<PINS || { echo "FAIL: a read-out tool is not the pinned version"; exit 1; }
e0a95aa351a0950d9b2af13e8bea4240  tools/seed_decision.py
e0a95aa351a0950d9b2af13e8bea4240  $K/seed_decision.py
6479eff0871ffead572b5d6d7541ec10  $K/arm_readouts.py
1a2ff663b374e0e4ce5c9dd3badf3b02  $K/../slt/metrics.py
288864b002af02da1bbc8239f05243c9  $K/stitch_lexicon.py
9428a3f542522ffe7a1de2ae91f41f5d  $K/w12b_readout.py
PINS
python "$K/w12b_readout.py" selftest
miss=0
for s in 42 101 202; do
  for arm in A W; do [ -f "$RUNS/${arm}_s${s}/DONE" ] || { echo "MISSING: ${arm}_s${s} (no DONE marker)"; miss=1; }; done
  w4summary "$RUNS" "$s" A W
done
[ "$miss" = 0 ] || { echo "FAIL: not all six runs finished; no read-out (the rule needs every seed)"; exit 1; }
echo "===== W - A (seed_decision.py) ====="
python tools/seed_decision.py --control A --arm W "$RUNS"/summary_A_W_s{42,101,202}.json
echo "===== W - A, secondaries and verdict (w12b_readout.py) ====="
python "$K/w12b_readout.py" "$RUNS" --train_manifest "$M/train.tsv" | tee "$G/wave12b/readout_W_minus_A.txt"
echo "W12B_SUMMARY_DONE"
