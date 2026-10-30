#!/bin/bash -l
# WAVE 4 corpus prep (CPU). Builds three stitch corpora from ONE sentence pool with ONE seed, then gates
# them with tools/w4_prep_check.py; the training array (w4_train.sbatch) is afterok on this job.
#   A  adopted build (trim 0.15)        -- must reproduce stitch_scale2/syn_13760 array-for-array
#   T  --trim_frac 0.0                  -- w4-trim: same exemplars/durations/transitions as A
#   Q  --realizations 2                 -- w5-div: first 6,880 of A's sentences x 2 (P == A)
# Pre-registration: results/PREREG_WAVE4_2026-09-22.md
#SBATCH --job-name=w4-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; W4=$G/wave4; B=$G/scratch_backup; MM=$B/mmwl
cd "$W4/code"; md5sum tools/build_posestitch_corpus_v2.py tools/stitch_lexicon.py tools/w4_prep_check.py
python tools/w4_prep_check.py --selftest
SENTS=$B/stitch_meta/stitch_v3/sentences_constrained.txt
COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --sentences $SENTS --min_cov 0.9 --min_signs 3 --seed 42 \
  --morphology --no_paren_alias --sign_dur --max_clips 13760"
C=$W4/corpora; rm -rf "$C"; mkdir -p "$C"
python tools/build_posestitch_corpus_v2.py $COMMON --realizations 1 --out_dir "$C/A" --audit_jsonl "$C/audit_A.jsonl" > "$C/A.log" 2>&1 &
python tools/build_posestitch_corpus_v2.py $COMMON --realizations 1 --trim_frac 0.0 --out_dir "$C/T" --audit_jsonl "$C/audit_T.jsonl" > "$C/T.log" 2>&1 &
python tools/build_posestitch_corpus_v2.py $COMMON --realizations 2 --out_dir "$C/Q" --audit_jsonl "$C/audit_Q.jsonl" > "$C/Q.log" 2>&1 &
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
tail -n 2 "$C"/A.log "$C"/T.log "$C"/Q.log
[ "$fail" = 0 ] || { echo "FAIL: a corpus build exited non-zero"; exit 1; }
python tools/w4_prep_check.py --real "$B/dwpose/manifests/train.tsv" \
  --control "$G/stitch_scale2/joint_13760.tsv" --orig "$G/stitch_scale2/syn_13760.tsv" \
  --corpora "$C" --out "$W4/manifests"
