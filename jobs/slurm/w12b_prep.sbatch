#!/bin/bash -l
# WAVE 12b prep (CPU): removing wrong-sense signs from the synthetic clips.
# Pre-registration (written before any corpus or run existed): results/PREREG_WAVE12B_WRONGSENSE_2026-10-13.md
#
# 14 English words or phrases are rendered with a sign of the wrong sense ("be" -> BE (NOT BE), "can" -> CAN (TIN), ...):
# 3,257 of the 94,428 signs of the adopted 13,760-clip corpus. From the SAME 13,760-sentence prefix of the pool, the
# same seed and the same flags, this job builds
#   A  the adopted corpus again with the edited builder, flag off   (gate only: must equal the adopted clips array for array)
#   W  the same build with --drop_wrong_sense                      (trained on by arm W)
# and gates them with tools/w12b_prep_check.py: A == adopted corpus 13,760/13,760; W == A minus exactly those signs
# (same English row for row, same exemplar/length/frames for every sign that stays); the pre-registered counts; no
# validation leak and only the one disclosed test match. Arm A then trains on the literal adopted manifest.
# The GPU array kaya/w12b_wrongsense.sbatch runs afterok on this job and checks $W12B/PREP_OK_<this job id>.
# Code: the wave-12 snapshot ($G/wave12/code), md5-pinned below to the files verified locally on 13 Oct 2026.
# If this job fails after building, move $W12B/corpora aside before resubmitting (it refuses to mix builds).
#SBATCH --job-name=w12b-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # env, G, M, assert_group; cd $G/wave12/code
set -u                                                             # after the library: module/activate are not -u clean
B=$G/scratch_backup; MM=$B/mmwl; ADOPTED=$G/stitch_scale2; W12B=$G/wave12b
POOL=$B/stitch_meta/stitch_v3/sentences_constrained.txt
CA=/scratch/pmc099/pprasad/wave12b/A_$SLURM_JOB_ID    # control rebuild: read only by the gate below, so /scratch is safe
CW=$W12B/corpora/W                                    # trained on: must be on /group

md5sum -c - <<EOF || { echo "FAIL: an input or a code file is not the pinned version"; exit 1; }
aabdeca5ec80cb8c0c4b180e0bd9e286  $W12/code/tools/build_posestitch_corpus_v2.py
288864b002af02da1bbc8239f05243c9  $W12/code/tools/stitch_lexicon.py
a7f67890a77652ad256cb275cb13d183  $W12/code/tools/w12b_prep_check.py
383c6424feca8073a49375df472ded1b  $POOL
915b3ce577d28b8c6898f940160df922  $MM/labels/Train.json
a14087ba93a9b511bfac8712085ae074  $MM/labels/Valid.json
500f9ce402e48e7285029bc19e20e5be  $ADOPTED/syn_13760.tsv
f81defc699396b920ac69e16a125af92  $ADOPTED/joint_13760.tsv
e67f2eb08b62fe62124a35afc9d4eeab  $M/train.tsv
b612245b71a57e4b9abdc2abc47172f8  $M/val.tsv
f73404a27780df006107a1d9e55db263  $M/test.tsv
EOF
[ "$(ls "$MM/dwpose_clips/train" | wc -l)" -eq 38580 ] && [ "$(ls "$MM/dwpose_clips/valid" | wc -l)" -eq 6430 ] \
  || { echo "FAIL: MM-WLAuslan DWPose clips are not 38,580 + 6,430"; exit 1; }
[ ! -e "$CA" ] && [ ! -e "$W12B/corpora" ] || { echo "FAIL: $CA or $W12B/corpora already exists; refusing to mix builds"; exit 1; }
assert_group "$ADOPTED/joint_13760.tsv" "$M/val.tsv" "$M/test.tsv"   # the adopted clips the gate and arm A read

python tools/w12b_prep_check.py selftest | tail -n 1
python tools/w12b_prep_check.py lexicon "$MM/labels/Train.json" "$MM/labels/Valid.json"

mkdir -p "$CA" "$CW" "$W12B/runs"
head -n 13760 "$POOL" > "$W12B/sent.txt"
[ "$(grep -c . "$W12B/sent.txt")" = 13760 ] || { echo "FAIL: pool prefix is not 13,760 sentences"; exit 1; }
COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
--min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1 --seed 42 --sentences $W12B/sent.txt"
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$CA" --audit_jsonl "$CA/audit.jsonl" \
  > "$W12B/build_A.log" 2>&1 &
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$CW" --audit_jsonl "$CW/audit.jsonl" --drop_wrong_sense \
  > "$W12B/build_W.log" 2>&1 &
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
tail -n 4 "$W12B/build_A.log" "$W12B/build_W.log"
[ "$fail" = 0 ] || { echo "FAIL: a corpus build exited non-zero"; exit 1; }
grep -q "drop_wrong_sense: removed 3254 of 94428 signs (3.45%); 1 clips left as built" "$W12B/build_W.log" \
  || { echo "FAIL: the builder did not remove the pre-registered 3,254 of 94,428 signs"; exit 1; }

python tools/w12b_prep_check.py gate --real "$M/train.tsv" --val "$M/val.tsv" --test "$M/test.tsv" \
  --control "$ADOPTED/joint_13760.tsv" --orig "$ADOPTED/syn_13760.tsv" --A "$CA" --W "$CW" --out "$W12B"

[ "$(grep -c . "$W12B/joint_W.tsv")" = 26199 ] || { echo "FAIL: joint_W.tsv is not 26,199 rows"; exit 1; }
eng () { cut -f2- "$1" | md5sum | cut -d' ' -f1; }
[ "$(eng "$W12B/joint_W.tsv")" = "$(eng "$ADOPTED/joint_13760.tsv")" ] \
  || { echo "FAIL: joint_W.tsv and the adopted manifest differ in English text"; exit 1; }
assert_group "$W12B/joint_W.tsv"
cp "$CA/audit.jsonl" "$W12B/audit_A.jsonl"            # A's clips are on /scratch and may be purged; its audit is kept
(cd "$W12B" && md5sum joint_W.tsv > joint_W.md5 && cat joint_W.md5)
touch "$W12B/PREP_OK_$SLURM_JOB_ID"
echo "W12B_PREP_DONE gate $W12B/PREP_OK_$SLURM_JOB_ID"
