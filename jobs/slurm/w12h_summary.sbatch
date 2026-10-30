#!/bin/bash -l
# WAVE 12h read-out (CPU; submit with --dependency=afterany:<w12h-pre array id>). Decides nothing by itself:
# the rule is in results/PREREG_WAVE12H_EXTRAPRETRAIN_2026-10-13.md, section 6.
#   Part A (frozen wave-11 snapshot only, as kaya/w10_summary.sbatch): per-seed summaries (w4summary) and the
#           seed-level contrasts PJ - C, PS - C, PS - PJ on beam-5 validation and test (tools/seed_decision.py).
#   Part B (wave-12 snapshot = the local tree): tools/w12h_readout.py -- the same contrasts recomputed from the saved
#           hypotheses, the secondary read-outs (unseen-target BLEU, content precision and recall, length ratio,
#           WER, soup), the pre-registered rule, and family_PJ.json for tools/holm_decision.py --family (run that
#           locally: it needs SciPy, which Kaya's environment does not have).
# Only arms carrying the DONE marker are used in Part B; Part A reports a missing marker.
#SBATCH --job-name=w12h-sum
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail   # -u only after the library: `module` and the venv activation read unset variables
LIB=/group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh; MD5_LIB=1463d0caae3a152d8ef4af914d8a5b8f
[ "$(md5sum < "$LIB" | cut -d' ' -f1)" = "$MD5_LIB" ] || { echo "FAIL: $LIB is not the version that ran wave 11"; exit 1; }
source "$LIB"   # sets G, M; cd wave11/code
set -u
RUNS=$G/wave12/runs_h; W12CODE=$G/wave12/code
[ -d "$RUNS" ] || { echo "FAIL: $RUNS does not exist (no arm has run)"; exit 1; }

echo "##### Part A: per-seed summaries and seed-level contrasts (wave-11 snapshot) #####"
F=()
for s in 505 606 707 808 909; do
  [ -d "$RUNS/C_s${s}" ] || [ -d "$RUNS/PJ_s${s}" ] || [ -d "$RUNS/PS_s${s}" ] || continue
  for arm in C PJ PS; do [ -f "$RUNS/${arm}_s${s}/DONE" ] || echo "MISSING: ${arm}_s${s} (no DONE marker)"; done
  w4summary "$RUNS" "$s" C PJ PS
  F+=("$RUNS/summary_C_PJ_PS_s${s}.json")
done
[ "${#F[@]}" -gt 0 ] || { echo "FAIL: no seed has a run folder"; exit 1; }
for pair in "C PJ" "C PS" "PJ PS"; do
  set -- $pair
  echo "===== $2 - $1 ====="
  # seed_decision.py raises when a contrast has no complete pair (e.g. every run of one arm failed): Part B must still run
  python tools/seed_decision.py --control "$1" --arm "$2" "${F[@]}" || echo "[WARN] $2 - $1: no seed-level read-out (no complete pair?)"
done

echo "##### Part B: pre-registered read-out (wave-12 snapshot) #####"
cd "$W12CODE" || { echo "FAIL: wave-12 code snapshot missing; run tools/w12h_readout.py locally on a copy of $RUNS"; exit 1; }
# tools/stitch_lexicon.py is not pinned by md5 (another wave-12 direction adds a constant to it): the read-out
# itself aborts unless its stop list is the 71-word list every earlier read-out used.
md5sum -c - <<'MD5' || { echo "FAIL: the read-out code is not the reviewed version"; exit 1; }
688f2c6aab2705d2d1e6e3d1b51bd84b  tools/w12h_readout.py
6479eff0871ffead572b5d6d7541ec10  tools/arm_readouts.py
e0a95aa351a0950d9b2af13e8bea4240  tools/seed_decision.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
MD5
python tools/w12h_readout.py selftest | tail -n 1
python tools/w12h_readout.py "$RUNS" --train_manifest "$M/train.tsv" --expect_unseen 549 570
echo "W12H_SUMMARY_DONE"
