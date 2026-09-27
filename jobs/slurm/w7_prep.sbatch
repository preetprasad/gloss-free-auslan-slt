#!/bin/bash -l
# WAVE 7 prep (CPU): build corpora A and D from the SAME 13,760-sentence prefix and gate them; the GPU array
# kaya/w7_density.sbatch runs afterok on this job (it checks $W7/BUILD_OK_<this job id>).
# Split out of the array's task 0 after job 20075 failed: $B was never defined (so the pool path was /stitch_meta/...),
# and the frozen wave-5 snapshot's builder predates --drop_function_signs. The builder used here is the shared-tree
# copy, md5-pinned to the file the pre-registration verified locally (flag off inert, 593/1,947 spans dropped).
# Pre-registration: results/PREREG_WAVE7_DENSITY_2026-09-23.md
#SBATCH --job-name=w7-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
B=$G/scratch_backup; W7=$G/wave7; C=/scratch/pmc099/pprasad/wave7/corpora; MM=$B/mmwl
POOL=$B/stitch_meta/stitch_v3/sentences_constrained.txt
BUILDER=$G/code/phase2_baseline/tools/build_posestitch_corpus_v2.py
mkdir -p "$W7" "$W7/runs"
[ -s "$POOL" ] || { echo "FAIL: pool $POOL missing"; exit 1; }
echo "d8ca137f600d21f2a41fe91d3b9483bd  $BUILDER" | md5sum -c - || { echo "FAIL: builder is not the verified version"; exit 1; }
echo "fef11937580691709dc9239e8f99eab6  $G/code/phase2_baseline/tools/stitch_lexicon.py" | md5sum -c -
[ ! -e "$C/A" ] && [ ! -e "$C/D" ] || { echo "FAIL: $C/A or $C/D already exists; refusing to mix builds"; exit 1; }

COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
--min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1 --seed 42"
mkdir -p "$C"; head -n 13760 "$POOL" > "$C/sent.txt"
[ "$(grep -c . "$C/sent.txt")" = 13760 ] || { echo "FAIL: pool prefix is not 13,760 sentences"; exit 1; }
python "$BUILDER" $COMMON --sentences "$C/sent.txt" --out_dir "$C/A" | tail -3
python "$BUILDER" $COMMON --sentences "$C/sent.txt" --out_dir "$C/D" \
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
touch "$W7/BUILD_OK_$SLURM_JOB_ID"; echo "W7_PREP_DONE gate $W7/BUILD_OK_$SLURM_JOB_ID"
