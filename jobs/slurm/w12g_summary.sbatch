#!/bin/bash -l
# WAVE 12g read-out (CPU): BLEU-4 on the held-out signers (SD), the same clips with the signer in training (EXP =
# EXPa on fold B + EXPb on fold A), per signer and by seen / unseen target, with intervals over seeds and bootstraps.
# Decides nothing; the readings are fixed in results/PREREG_WAVE12G_SIGNERSPLIT_2026-10-13.md.
# Submit with --dependency=afterany:<w12g-train array id> --export=ALL,PREP=<prep job id>.
#SBATCH --job-name=w12g-summary
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh
set -u
W=$G/wave12g; TOOLS=${TOOLS:-$G/wave12/code/tools}; RUNS=$W/runs
MAN=$W/manifests_${PREP:?submit with --export=ALL,PREP=<prep job id>}
[ -f "$MAN/BUILD_OK" ] || { echo "FAIL: gate $MAN/BUILD_OK missing"; exit 1; }
# the read-out and what it imports are the files tested locally (metrics identical to the training snapshot's).
# stitch_lexicon.py is edited by other experiments, so the read-out pins the stop list it uses (71 words) by content.
md5sum -c - <<MD5 || { echo "FAIL: a read-out file is not the verified version"; exit 1; }
7b6d6e0c4dce482a54d1ec13346870c9  $TOOLS/w12g_readout.py
c33b312359b5dac0f28305865f41dd39  $TOOLS/w12g_signer_split.py
6479eff0871ffead572b5d6d7541ec10  $TOOLS/arm_readouts.py
e0a95aa351a0950d9b2af13e8bea4240  $TOOLS/seed_decision.py
1a2ff663b374e0e4ce5c9dd3badf3b02  $TOOLS/../slt/metrics.py
MD5
# A seed is complete only if all three arms have their DONE marker (written after the vocabulary gate and both
# decodes). Decodes of an arm without one are not read; with fewer than three complete seeds the read-out prints
# its tables and "NO READING".
OK=()
for s in 42 101 202; do n=0
  for arm in SD EXPa EXPb; do
    if [ -f "$RUNS/DONE_${arm}_s${s}" ]; then n=$((n + 1)); else echo "MISSING: ${arm}_s${s} (no DONE marker)"; fi
  done
  if [ "$n" = 3 ]; then OK+=("$s"); fi
done
[ "${#OK[@]}" -gt 0 ] || { echo "FAIL: no seed has all three DONE markers"; exit 1; }
echo "complete seeds: ${OK[*]}"
python "$TOOLS/w12g_readout.py" selftest | tail -1
python -u "$TOOLS/w12g_readout.py" readout --runs "$RUNS" --manifests "$MAN" --seeds "${OK[@]}" --boot 1000
echo "===== primary contrast through tools/seed_decision.py (must equal the 'all held-out signers' lines above) ====="
F=(); for s in "${OK[@]}"; do F+=("$RUNS/summary_w12g_s$s.json"); done   # written by the read-out just above
python "$TOOLS/seed_decision.py" --control SD --arm EXP "${F[@]}"
echo "W12G_SUMMARY_DONE"
