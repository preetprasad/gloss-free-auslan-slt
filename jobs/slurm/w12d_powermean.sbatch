#!/bin/bash -l
# WAVE 12d: the ensemble's averaging rule. The reported six-member ensemble averages its members' log-probabilities
# (geometric mean of their probabilities); does a power mean (p = -1, 0.5, 1, 2) decode VALIDATION better?
# One array task per rule (0: p = 0 control, 1: p = -1, 2: p = 0.5, 3: p = 1, 4: p = 2), then one read-out task.
# CPU only, no training, NO TEST DECODE. Pre-registration: results/PREREG_WAVE12D_POWERMEAN_2026-10-13.md
#   decode:    J=$(sbatch --parsable --export=ALL,MODE=decode kaya/w12d_powermean.sbatch)
#              (MODE is set explicitly: sbatch passes the submitting shell's environment on, and an exported
#               MODE=report would send all five tasks into the read-out branch)
#   read-out:  sbatch --array=0 --dependency=afterok:$J --export=ALL,MODE=report kaya/w12d_powermean.sbatch
# Code: the wave-12 snapshot ($G/wave12/code); the three files of this experiment are md5-pinned below.
#SBATCH --job-name=w12d-powermean
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # modules, environment, cd into the snapshot
set -u                                                              # only now: module and activate read unset variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK   # 16, as in job 120153 (gate 2)
OUT=$W12/w12d/out; PRIOR=$G/ens_ablation/out
MODE=${MODE:-decode}; T=$SLURM_ARRAY_TASK_ID
POWERS=(0 -1 0.5 1 2)
echo "mode $MODE  task $T  snapshot $(pwd)"

# ---- code gates ----
md5sum -c - <<'EOF' || { echo "FAIL: the experiment's code differs from the files that were verified locally"; exit 1; }
eb92e7a581d6c3acb51eb01746c16a48  tools/ensemble_mbr_wer.py
40168aac538c679fbfd1fb17fefde51e  tools/w12d_powermean.py
1016182f87254a707a3eeb275ea62975  ensemble_decode.py
EOF
if [ -f "$W12/code.md5" ]; then
  md5sum --quiet -c "$W12/code.md5" || { echo "FAIL: snapshot differs from $W12/code.md5"; exit 1; }
  echo "snapshot matches $W12/code.md5"
fi
echo "for the record, files that differ from the 10 October snapshot which produced the recorded hypotheses"
echo "(tools/ensemble_mbr_wer.py is expected; the read-out's gates show whether any difference changes the control):"
md5sum -c "$G/ens_ablation/code.md5" 2>/dev/null | grep -v ': OK$' || true
python tools/w12d_powermean.py self_test

if [ "$MODE" = report ]; then
  for p in "${POWERS[@]}"; do
    [ -f "$OUT/DONE_p$p" ] || { echo "FAIL: the decode of p = $p has not finished"; exit 1; }
  done
  [ "$(grep -c . "$M/train.tsv")" = 12439 ] || { echo "FAIL: training manifest row count"; exit 1; }
  python -u tools/w12d_powermean.py report --out "$OUT" --prior_dir "$PRIOR" --train_manifest "$M/train.tsv"
  touch "$OUT/DONE_report"
  echo "W12D_REPORT_DONE"; exit 0
fi
[ "$MODE" = decode ] || { echo "FAIL: MODE must be decode or report"; exit 1; }

# ---- data gates (validation only) ----
p=${POWERS[$T]}
VAL=$M/val.tsv; TEST=$M/test.tsv
[ "$(grep -c . "$VAL")" = 800 ] || { echo "FAIL: validation manifest row count"; exit 1; }
pin_pose
assert_group "$VAL" || { echo "FAIL: a validation clip is missing or not on /group"; exit 1; }
python - "$VAL" "$TEST" <<'PY'
import sys
v = [l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l]; t = {l.split("\t")[0] for l in open(sys.argv[2]) if "\t" in l}
assert len(v) == len(set(v)) == 800 and not set(v) & t, f"{len(set(v) & t)} clips in both val and test"
print(f"val {len(v)} clips, overlap with test 0")
PY
RGB=$G/scratch_backup/dwpose/rgb_feats
[ "$(find "$RGB" -maxdepth 1 -name '*.npz' | wc -l)" -eq 14041 ] || { echo "FAIL: RGB feature count"; exit 1; }
[ -s "$PRIOR/shard2of4.json" ] || { echo "FAIL: job 120153's six-member hypotheses (gate 2) are missing"; exit 1; }
CK="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
# the six reported members in their recorded order (same pins as kaya/ens_subsets.sbatch)
md5sum ${CK//,/ } | awk '{print $1}' | tr '\n' ' ' | grep -q "^3ea55e3474eaa9d1e093d6da300ad157 9b068067a1c4b90db5dcfd5b46d102b5 634d760e6e2eb09823992a03fa3b9c10 7146145c437014895a548e018888805c ccb09833497b81d541147103c2c382fd 2b3644b1b2c26b06641f0e6dbce1660b $" \
  || { echo "FAIL: checkpoint md5 mismatch"; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT/DONE_p$p"
echo "=== decode, p = $p ==="
python -u tools/w12d_powermean.py decode --ckpts "$CK" --manifest "$VAL" --rgb_feat_dir "$RGB" --out "$OUT" \
    --power="$p" --beam 10
[ -s "$OUT/nbest_p$p.json" ] || { echo "FAIL: no decode file for p = $p"; exit 1; }
touch "$OUT/DONE_p$p"
echo "W12D_DECODE_DONE p = $p"
