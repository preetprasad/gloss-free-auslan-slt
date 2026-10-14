#!/bin/bash -l
# WAVE 12f, step 3 of 3 (CPU): the two measurements. No model, no checkpoint, no validation or test clip.
#   measurement 1  Auslan-Daily's own isolated-sign clips as the reference set for the conversational spans
#                  (gates H, E, T first; then the arms; MM-WLAuslan citation recordings as a second reference set)
#   measurement 2  estimated depth: x/y, x/y/z and z alone (only if step 2 wrote DONE_pose3d)
# Rules and readings: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md
#   sbatch --dependency=afterany:<extract job> kaya/w12f_probe.sbatch
# afterany, not afterok: a failure of the 3-D task (measurement 2) must not block measurement 1. This script itself
# refuses to run without task 0's marker (DONE_dwpose) and runs measurement 2 only with task 1's marker (DONE_pose3d).
#SBATCH --job-name=w12f-probe
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # modules, environment, cd into the snapshot
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12flib.sh      # paths, code pins, gates (the snapshot holds no job scripts)
set -u
w12f_code_gate
[ -f "$OUT/DONE_stage" ] || fail "step 1 (kaya/w12f_stage.sbatch) has not finished"
[ -f "$OUT/DONE_dwpose" ] || fail "step 2, task 0 (DWPose of the isolated clips) has not finished"
rm -f "$OUT/DONE_iso" "$OUT/DONE_depth"
pin_pose                                                            # production DWPose is the untracked original
[ "$(find "$B/dwpose_clips" -maxdepth 1 -name '*.npz' | wc -l)" = 14041 ] || fail "production DWPose file count"
[ "$(find "$DW_ISO" -maxdepth 1 -name '*.npz' | wc -l)" = "$N_ISO" ] || fail "DWPose files of the isolated clips"
for d in "$MM/dwpose_clips/train" "$MM/dwpose_clips/valid"; do [ -d "$d" ] || fail "missing $d"; done
for f in "$MM/labels/Train.json" "$MM/labels/Valid.json"; do [ -s "$f" ] || fail "missing $f"; done

# every span clip is a Communication TRAIN clip with a production pose file
python - "$OUT/spans_safe.csv" "$M" "$B/dwpose_clips" <<'PY' || fail "span clips"
import csv, os, re, sys
spans, m, prod = sys.argv[1:]
ids = {re.sub(r"_signer\.mp4$", "", r["Video_Name_Clip"]) for r in csv.DictReader(open(spans))}
split = {s: {os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(f"{m}/{s}.tsv") if "\t" in l} for s in ("train", "val", "test")}
assert ids <= split["train"] and not ids & (split["val"] | split["test"]), "a span clip is not a TRAIN clip"
miss = [c for c in ids if not os.path.exists(f"{prod}/{c}.npz")]
assert not miss, f"{len(miss)} span clips without a production pose"
print(f"span clips {len(ids)}: all in TRAIN, none in validation or test, all with a production pose")
PY

ARGS=(--spans "$OUT/spans_safe.csv" --prod_dir "$B/dwpose_clips" --bank_dir "$DW_ISO" --located "$OUT/located.csv")
echo "=== measurement 1, gates only ==="
python -u tools/w12f_probe.py iso "${ARGS[@]}" --gate_only | tee "$OUT/iso_gates.txt"
echo "=== measurement 1 ==="
python -u tools/w12f_probe.py iso "${ARGS[@]}" --out "$OUT/iso.json" \
    --mmwl_labels "$MM/labels/Train.json" "$MM/labels/Valid.json" \
    --mmwl_dirs "$MM/dwpose_clips/train" "$MM/dwpose_clips/valid" | tee "$OUT/iso.txt"
[ -s "$OUT/iso.json" ] || fail "measurement 1 wrote no result"
touch "$OUT/DONE_iso"
echo "W12F_ISO_DONE"

if [ -f "$OUT/DONE_pose3d" ]; then
  [ "$(find "$P3D" -maxdepth 1 -name '*.npz' | wc -l)" = "$N_SPAN_CLIPS" ] || fail "3-D pose file count"
  echo "=== measurement 2 ==="
  python -u tools/w12f_probe.py depth --spans "$OUT/spans_safe.csv" --prod_dir "$B/dwpose_clips" --d3_dir "$P3D" \
      --out "$OUT/depth.json" | tee "$OUT/depth.txt"
  [ -s "$OUT/depth.json" ] || fail "measurement 2 wrote no result"
  touch "$OUT/DONE_depth"
  echo "W12F_DEPTH_DONE"
else
  echo "W12F_DEPTH_NOT_RUN: step 2, task 1 left no DONE_pose3d marker (the weights were absent and their download was"
  echo "  not approved, or the task failed: see its log w12f-extract-<job>_1.out). Measurement 2 was not computed."
fi
