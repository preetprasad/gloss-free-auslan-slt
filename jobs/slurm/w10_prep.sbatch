#!/bin/bash -l
# WAVE 10 prep (CPU): does the ORDER of the stitched signs matter? Builds, from the SAME 13,760-sentence prefix wave 7
# used, corpus A (English sign order, the adopted build) and corpus S (--shuffle_order: identical exemplars, durations,
# sign count and per-position transition budgets, only the order changes), then gates them:
#   * A reproduces wave 7's corpus A array-for-array (the new builder is inert with the flag off, at full scale)
#   * S differs from A only in order (tools/shuffle_order_check.py order)
# The GPU array kaya/w10_order.sbatch runs afterok on this job (it checks $W10/PREP_OK_<this job id>).
# Code runs from a snapshot ($W10/code/tools), md5-pinned to the files verified locally, so the shared tree the pending
# waves import is untouched. Pre-registration: results/PREREG_WAVE10_SIGNORDER_2026-10-01.md
#SBATCH --job-name=w10-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
B=$G/scratch_backup; MM=$B/mmwl; W10=$G/wave10; K=$W10/code; C=$W10/corpora
POOL=$B/stitch_meta/stitch_v3/sentences_constrained.txt
W7A=/scratch/pmc099/pprasad/wave7/corpora/A
echo "383c6424feca8073a49375df472ded1b  $POOL" | md5sum -c -
echo "c6d81cc6f39f88f8d0163a67d9226311  $K/tools/build_posestitch_corpus_v2.py" | md5sum -c -
echo "fef11937580691709dc9239e8f99eab6  $K/tools/stitch_lexicon.py" | md5sum -c -
echo "83493b02b383b8272f2d8da4c3e826ec  $K/tools/shuffle_order_check.py" | md5sum -c -
echo "c10d8e2666d18262aa35953aaae87cad  $K/tools/w10_ctc_map.py" | md5sum -c -
[ "$(grep -c . "$W7A/posestitch_v2_train.tsv")" = 13760 ] || { echo "FAIL: wave-7 corpus A is not 13,760 rows"; exit 1; }
[ ! -e "$C" ] || { echo "FAIL: $C already exists; refusing to mix builds"; exit 1; }
mkdir -p "$C/A" "$C/S" "$W10/runs"

COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
--min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1 --seed 42"
head -n 13760 "$POOL" > "$C/sent.txt"
[ "$(grep -c . "$C/sent.txt")" = 13760 ] || { echo "FAIL: pool prefix is not 13,760 sentences"; exit 1; }
cd "$K"
python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C/sent.txt" --out_dir "$C/A" \
  --audit_jsonl "$C/A/audit.jsonl" > "$C/A.log" 2>&1 &
python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C/sent.txt" --out_dir "$C/S" \
  --audit_jsonl "$C/S/audit.jsonl" --shuffle_order > "$C/S.log" 2>&1 &
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
tail -n 2 "$C/A.log" "$C/S.log"
[ "$fail" = 0 ] || { echo "FAIL: a corpus build exited non-zero"; exit 1; }

python tools/shuffle_order_check.py inert "$W7A" "$C/A"
python tools/shuffle_order_check.py order "$C/A" "$C/S"
# arm SC: S's CTC targets re-ordered to follow its signs (checks: == to_gloss in English order, same words, no collisions)
python tools/w10_ctc_map.py --audit "$C/S/audit.jsonl" --labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid --trainer_dir "$W5/code" \
  --real "$M/train.tsv" "$M/val.tsv" --out "$W10/ctc_map_S.tsv"
python - "$C/A/posestitch_v2_train.tsv" "$C/S/posestitch_v2_train.tsv" "$M/train.tsv" "$W10" <<'PY'
import os, sys
a, s, real, w10 = sys.argv[1:5]
A = [l.rstrip("\n").split("\t", 1) for l in open(a) if "\t" in l]
S = [l.rstrip("\n").split("\t", 1) for l in open(s) if "\t" in l]
assert len(A) == len(S) == 13760, f"row counts {len(A)} / {len(S)}, expected 13760 each"
assert [t for _, t in A] == [t for _, t in S], "the two corpora do not match row-for-row on English text"
assert not {p for p, _ in A} & {p for p, _ in S}, "the two corpora share clip files"
R = [l.rstrip("\n") for l in open(real) if "\t" in l]
assert len(R) == 12439, f"real train has {len(R)} rows"
for arm, rows in (("A", A), ("S", S)):
    with open(os.path.join(w10, f"joint_{arm}.tsv"), "w") as f:
        f.write("\n".join(R) + "\n" + "\n".join(f"{p}\t{t}" for p, t in rows) + "\n")
print("GATE OK: 13,760 rows each, English identical row-for-row, clip sets disjoint, real 12,439 -> joint 26,199")
PY
touch "$W10/PREP_OK_$SLURM_JOB_ID"; echo "W10_PREP_DONE gate $W10/PREP_OK_$SLURM_JOB_ID"
