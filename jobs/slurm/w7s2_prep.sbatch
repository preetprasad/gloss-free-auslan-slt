#!/bin/bash -l
# WAVE 7 STAGE 2 prep (CPU). Stage 1 (job 31220) cleared futility (D - A beam-5 validation +0.099), so the pre-registration
# sends D to Stage 2 with a third arm, DL = D with each clip's per-sign lengths rescaled so its TOTAL frame count equals its
# A counterpart (results/PREREG_WAVE7_DENSITY_2026-09-23.md, sections 4 and 10). A and D are reused from Stage 1 unchanged.
# This job builds DL and gates it:
#   * the new builder, flag off, reproduces Stage-1 corpus D array-for-array (inert at full scale)
#   * DL has D's signs, exemplars and transition budgets, A's clip length, and sign lengths that are the exact proportional
#     rescale of D's sampled lengths (tools/shuffle_order_check.py match)
# Code runs from a snapshot ($S2/code/tools) md5-pinned to the files verified locally. The GPU array
# kaya/w7s2_train.sbatch runs afterok on this job and checks $S2/PREP_OK_<this job id>.
#SBATCH --job-name=w7s2-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
B=$G/scratch_backup; MM=$B/mmwl; W7=$G/wave7; S2=$G/wave7s2; K=$S2/code
C1=/scratch/pmc099/pprasad/wave7/corpora; C2=/scratch/pmc099/pprasad/wave7s2/corpora
echo "40ec10a752f0e0ae0c81996ddb45a220  $K/tools/build_posestitch_corpus_v2.py" | md5sum -c -
echo "fef11937580691709dc9239e8f99eab6  $K/tools/stitch_lexicon.py" | md5sum -c -
echo "a3634876c20f1e36da06e16529fe15e6  $K/tools/shuffle_order_check.py" | md5sum -c -
echo "f79c97bd7edb3589a447e7ded6463f22  $C1/sent.txt" | md5sum -c -
echo "55c3e9673d24621885a1eadbd629099e  $C1/A/posestitch_v2_train.tsv" | md5sum -c -
echo "22c79b7f0399075a940a676e4f712883  $C1/D/posestitch_v2_train.tsv" | md5sum -c -
echo "d87466e755f80981ee73b0f334994161  $W7/joint_A.tsv" | md5sum -c -
[ ! -e "$C2" ] || { echo "FAIL: $C2 already exists; refusing to mix builds"; exit 1; }
mkdir -p "$C2/D" "$C2/DL"

# exactly Stage 1's flags (kaya/w7_prep.sbatch)
COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
--min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1 --seed 42"
cd "$K"
python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C1/sent.txt" --out_dir "$C2/D" \
  --audit_jsonl "$C2/D/audit.jsonl" --drop_function_signs > "$C2/D.log" 2>&1 &
python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C1/sent.txt" --out_dir "$C2/DL" \
  --audit_jsonl "$C2/DL/audit.jsonl" --drop_function_signs --match_frames "$C1/A/posestitch_v2_train.tsv" \
  > "$C2/DL.log" 2>&1 &
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
tail -n 3 "$C2/D.log" "$C2/DL.log"
[ "$fail" = 0 ] || { echo "FAIL: a corpus build exited non-zero"; exit 1; }

python tools/shuffle_order_check.py inert "$C1/D" "$C2/D"
python tools/shuffle_order_check.py match "$C1/A" "$C2/D" "$C2/DL"
python - "$C2/DL/posestitch_v2_train.tsv" "$W7/joint_A.tsv" "$W7/joint_D.tsv" "$M/train.tsv" "$S2" <<'PY'
import os, sys
dl, ja, jd, real, s2 = sys.argv[1:6]
rd = lambda p: [l.rstrip("\n").split("\t", 1) for l in open(p) if "\t" in l]
DL, JA, JD, R = rd(dl), rd(ja), rd(jd), [l.rstrip("\n") for l in open(real) if "\t" in l]
assert len(DL) == 13760 and len(R) == 12439 and len(JA) == len(JD) == 26199, (len(DL), len(R), len(JA), len(JD))
joint = R + [f"{p}\t{t}" for p, t in DL]
J = [l.split("\t", 1) for l in joint]
assert [t for _, t in J] == [t for _, t in JA] == [t for _, t in JD], "English text differs from joint_A / joint_D"
assert [l for l in joint[:12439]] == ["\t".join(x) for x in JA[:12439]], "real half differs from joint_A"
syn = lambda rows: {p for p, _ in rows[12439:]}
assert not syn(J) & syn(JA) and not syn(J) & syn(JD), "DL shares clip files with A or D"
with open(os.path.join(s2, "joint_DL.tsv"), "w") as f:
    f.write("\n".join(joint) + "\n")
print("GATE OK: joint_DL 26,199 rows; English text identical to joint_A and joint_D row for row; real half identical; "
      "DL clip set disjoint from A and D")
PY
touch "$S2/PREP_OK_$SLURM_JOB_ID"; echo "W7S2_PREP_DONE gate $S2/PREP_OK_$SLURM_JOB_ID"
