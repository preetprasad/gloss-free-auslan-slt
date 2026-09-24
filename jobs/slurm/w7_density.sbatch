#!/bin/bash -l
# WAVE 7 -- sign density: stop rendering the signs Auslan drops. Task = (arm, seed):  0-2 A | 3-5 D  (42/101/202)
#   A = adopted corpus rebuilt in batch          D = the SAME sentences with --drop_function_signs
# Real Auslan carries 0.467 signs per English word; the stitched corpus carries 0.78 (09-05 audit, intervention (iv),
# never run until now). The English target, the vocabulary and the real half are identical between arms.
# Task 0 builds both corpora and gates them; the other tasks wait for the gate file, so the array needs no prep job.
# Pre-registration: results/PREREG_WAVE7_DENSITY_2026-09-23.md
#SBATCH --job-name=w7-density
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-5
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 101 202); ARMS=(A D)
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
# clips go to /scratch (regenerable); manifests, gate file and runs stay on /group
W7=$G/wave7; C=/scratch/pmc099/pprasad/wave7/corpora; MM=$B/mmwl
POOL=$B/stitch_meta/stitch_v3/sentences_constrained.txt
RUNS=$W7/runs; mkdir -p "$W7" "$RUNS"
# gate is scoped to THIS array job: a stale BUILD_OK from an earlier submission must never be honoured
GATE=$W7/BUILD_OK_$SLURM_ARRAY_JOB_ID
echo "task $T -> arm $arm seed $s"

# ---- task 0 builds both corpora once and writes the gate file; every other task waits for it ----
COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1 --seed 42"
if [ "$T" = 0 ]; then
  mkdir -p "$C"; head -n 13760 "$POOL" > "$C/sent.txt"
  [ "$(grep -c . "$C/sent.txt")" = 13760 ] || { echo "FAIL: pool prefix is not 13,760 sentences"; exit 1; }
  python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C/sent.txt" --out_dir "$C/A" | tail -3
  python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C/sent.txt" --out_dir "$C/D" \
    --drop_function_signs | tail -4
  python - "$C/A/posestitch_v2_train.tsv" "$C/D/posestitch_v2_train.tsv" "$M/train.tsv" "$W7" <<'PY'
import os, sys
a, d, real, w7 = sys.argv[1:5]
A = [l.rstrip("\n").split("\t", 1) for l in open(a) if "\t" in l]
D = [l.rstrip("\n").split("\t", 1) for l in open(d) if "\t" in l]
assert len(A) == len(D) == 13760, f"row counts {len(A)} / {len(D)}, expected 13760 each"
assert [t for _, t in A] == [t for _, t in D], "the two corpora do not match row-for-row on English text"
assert len({p for p, _ in A} & {p for p, _ in D}) == 0, "the two corpora share clip files"
R = [l.rstrip("\n") for l in open(real) if "\t" in l]
assert len(R) == 12439, f"real train has {len(R)} rows"
for arm, rows in (("A", A), ("D", D)):
    with open(os.path.join(w7, f"joint_{arm}.tsv"), "w") as f:
        f.write("\n".join(R) + "\n" + "\n".join(f"{p}\t{t}" for p, t in rows) + "\n")
print(f"GATE OK: 13,760 rows each, English text identical row-for-row, clip sets disjoint, real 12,439")
PY
  touch "$GATE"
else
  for i in $(seq 1 240); do [ -f "$GATE" ] && break; sleep 30; done
  [ -f "$GATE" ] || { echo "FAIL: corpus build gate never appeared"; exit 1; }
fi

JOINT=$W7/joint_${arm}.tsv
[ "$(grep -c . "$JOINT")" = 26199 ] || { echo "FAIL: $JOINT is not 26,199 rows"; exit 1; }
ALLOW_PREFIX="/scratch/pmc099/pprasad/wave7/corpora/" assert_group "$JOINT" "$M/val.tsv" "$M/test.tsv"
w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF \
  || { echo "[FAIL] train ${arm}"; exit 1; }
DIR=$RUNS/${arm}_s${s}
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$DIR/DONE"
echo "W7_DONE ${arm} seed ${s}"
