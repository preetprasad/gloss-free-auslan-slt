#!/bin/bash -l
# WAVE 12i read-out (CPU). Submit with --dependency=afterany:<w12i-heavy array id>. It reads nothing unless all ten
# runs carry their DONE marker. Order: tool self-test -> the tool must reproduce the 18 recorded wave-9 DiD values from
# the wave-9 dumps on /group -> per-seed summaries (w4summary) -> whole-split V2 - C0 (tools/seed_decision.py) ->
# protocol read-outs (tools/arm_readouts.py) -> the pre-registered heavy-bin difference-in-differences and its verdict
# (tools/w12i_heavy_did.py). Pre-registration: results/PREREG_WAVE12I_TRACKER_HEAVY_2026-10-13.md.
# The read-out code runs from the wave-12 snapshot; the training ran from the wave-5 snapshot (as wave 9 did).
# If tools/w12i_heavy_did.py is edited after review, update its md5 below before the snapshot is made.
#SBATCH --job-name=w12i-summary
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
set +u; source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh; set -u
W9=$G/wave9; MW=$W9/manifests; RUNS=$W12/w12i/runs; SEEDS=(303 404 505 606 707)
md5sum -c --quiet - <<'EOF' || { echo "FAIL: the read-out tool or the frozen bins are not the reviewed files"; exit 1; }
902bb342204954eacc725a14a52b3787  tools/w12i_heavy_did.py
201ea0c732d2f61c36fa0a248cc32ec2  tools/w12i_bins.json
EOF
md5sum -c --quiet - <<EOF || { echo "FAIL: a wave-9 validation or test manifest changed"; exit 1; }
df6cf8e8693825a09fa2e7aff8744e65  $MW/c0_val.tsv
f7af237641d9bcf2226789ce170af5ae  $MW/c0_test.tsv
493f59e68a93899728c8278f175e6946  $MW/v2_val.tsv
dd6040a6f9652f4d8ea300f75e5c2491  $MW/v2_test.tsv
d35c17eb6cf0c66819c9e7b50a43bfda  $MW/c0_train.tsv
EOF
python tools/w12i_heavy_did.py selftest | tail -1
python tools/w12i_heavy_did.py w9gate --runs "$W9/runs" --manifests "$MW"

miss=0
for s in "${SEEDS[@]}"; do for arm in c0 v2; do
  [ -f "$RUNS/${arm}_s${s}/DONE" ] || { echo "MISSING: ${arm}_s${s} (no DONE marker)"; miss=1; }
done; done
[ "$miss" -eq 0 ] || { echo "W12I_SUMMARY_INCOMPLETE: no score was read"; exit 1; }

F=(); D=()
for s in "${SEEDS[@]}"; do
  w4summary "$RUNS" "$s" c0 v2
  F+=("$RUNS/summary_c0_v2_s${s}.json"); D+=("$RUNS/c0_s${s}" "$RUNS/v2_s${s}")
done
echo "===== whole split, v2 - c0 (not the criterion) ====="
python tools/seed_decision.py --control c0 --arm v2 "${F[@]}"
for split in val test; do
  python tools/arm_readouts.py "${D[@]}" --split "$split" --train_manifest "$MW/c0_train.tsv" > "$RUNS/readouts_${split}.txt"
done
echo "===== heavy-switch bin (the pre-registered read-out) ====="
python -u tools/w12i_heavy_did.py report --runs "$RUNS" --manifests "$MW" --readouts_dir "$RUNS" | tee "$RUNS/w12i_report.txt"
echo "W12I_SUMMARY_DONE"
