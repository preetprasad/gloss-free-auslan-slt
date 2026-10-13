#!/bin/bash -l
# WAVE 12g prep (CPU): build and check the signer-disjoint split and its exposure control.
#   SD    train = signers 1 and 2 outside six validation videos (12,024 clips)   test = all 1,087 clips of the 47 other signers
#   EXPa  train = SD train with 544 random clips replaced by fold A              test = fold B (543)
#   EXPb  train = SD train with 543 random clips replaced by fold B              test = fold A (544)
#   validation (all arms) = signers 1 and 2 in six whole videos that hold no other signer (928 clips)
# The split is deterministic; this job fails unless every manifest is byte-identical to the one built and checked on
# the Mac on 13 Oct 2026 from byte-identical copies of the official manifests. The GPU array kaya/w12g_train.sbatch
# runs afterok on this job and reads $W/manifests_<this job id>.
# Needs (copied by hand before submission): $W/Auslan-Daily_Communication.xlsx and $TOOLS/w12g_signer_split.py.
# Pre-registration: results/PREREG_WAVE12G_SIGNERSPLIT_2026-10-13.md
#SBATCH --job-name=w12g-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh
set -u                                   # after the library: `module` and the venv activation are not nounset-clean
W=$G/wave12g; TOOLS=${TOOLS:-$G/wave12/code/tools}; XLSX=$W/Auslan-Daily_Communication.xlsx
MAN=$W/manifests_$SLURM_JOB_ID
SPLIT="$TOOLS/w12g_signer_split.py"
IN=(--xlsx "$XLSX" --train "$M/train.tsv" --val "$M/val.tsv" --test "$M/test.tsv")
pin_pose

# ---- gate 1: the tool and its four inputs are the files the split was built from locally ----
md5sum -c - <<MD5 || { echo "FAIL: the split tool or one of its inputs is not the verified file"; exit 1; }
c33b312359b5dac0f28305865f41dd39  $SPLIT
ddee0900da09d416365e96056e0e533f  $XLSX
e67f2eb08b62fe62124a35afc9d4eeab  $M/train.tsv
b612245b71a57e4b9abdc2abc47172f8  $M/val.tsv
f73404a27780df006107a1d9e55db263  $M/test.tsv
MD5
[ ! -e "$MAN" ] || { echo "FAIL: $MAN already exists"; exit 1; }

# ---- build, then re-read and check the written files (signers, videos, folds, sizes, verbatim rows, /group paths) ----
python "$SPLIT" selftest
python "$SPLIT" build "${IN[@]}" --out_dir "$MAN"
python "$SPLIT" check "${IN[@]}" --out_dir "$MAN"

# ---- gate 2: byte-identical to the local build ----
( cd "$MAN" && md5sum -c - <<MD5 ) || { echo "FAIL: a manifest differs from the local build"; exit 1; }
a22c62a9e74c3ce2a756080f03a1b204  w12g_SD_train.tsv
c8fb5f854a277ed6760f1b6d65acb4a2  w12g_SD_test.tsv
09ee6f3ca47e72fc3a9c715569802b6d  w12g_EXPa_train.tsv
7a287f49776d69e0e66a11b3e5a44af7  w12g_EXPa_test.tsv
a0abd96db602e3df40a380bc1a2d7b72  w12g_EXPb_train.tsv
82133f7f1fd92d2809c41e66c84c6f85  w12g_EXPb_test.tsv
000a54bc2dab309e84da16bf73ca43f1  w12g_val.tsv
dec86a56dfbbb49d8b0c04209686a8a3  w12g_clips.tsv
MD5
wc -l "$MAN"/w12g_*.tsv

# ---- gate 3: every clip of every manifest exists on /group ----
assert_group "$MAN"/w12g_{SD,EXPa,EXPb}_{train,test}.tsv "$MAN/w12g_val.tsv"
touch "$MAN/BUILD_OK"
echo "W12G_PREP_DONE gate $MAN/BUILD_OK"
