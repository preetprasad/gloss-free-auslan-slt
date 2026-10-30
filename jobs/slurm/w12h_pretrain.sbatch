#!/bin/bash -l
# WAVE 12h -- extra pre-training before the main training: a confirmatory test with NEW seeds.
#   C  = the adopted recipe from scratch (control, same job, same seed)
#   PJ = pre-train 92 epochs on the adopted joint corpus (12,439 real + 13,760 synthetic clips = 26,199;
#        2,410,308 clip-passes, 37,720 steps), then the adopted recipe from the FINAL pre-training checkpoint
#   PS = pre-train 175 epochs on the 13,760 synthetic clips of that corpus only (2,408,000 clip-passes,
#        37,625 steps: the whole number of epochs closest to PJ's steps and clip-passes), then the same
# Task = (arm, seed).  Stage 1 (default):                     seeds 505/606/707 -> 0-2 C | 3-5 PJ | 6-8 PS
#                      Stage 2 (W12H_STAGE=2, --array=0-5):   seeds 808/909     -> 0-1 C | 2-3 PJ | 4-5 PS
# A task that failed for a technical reason is resubmitted once, alone (--array=<task>); a finished arm (DONE
# marker) is refused, so no arm can be chosen between two runs.
# Everything else is wave 5's C and PJ: the fixed wave-5 vocabulary, --deterministic True, pose augmentation,
# 40-epoch main training, best.pt by greedy validation, beam-5 validation and test of best.pt and the soup.
# Code: the frozen wave-11 snapshot through _w11lib.sh; the files this job runs are md5-pinned below to the
# values of the wave-5 snapshot that ran C and PJ in jobs 12629 and 16397.
# Pre-registration: results/PREREG_WAVE12H_EXTRAPRETRAIN_2026-10-13.md
#SBATCH --job-name=w12h-pre
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-8
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail   # -u only after the library: `module` and the venv activation read unset variables
LIB=/group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh; MD5_LIB=1463d0caae3a152d8ef4af914d8a5b8f
[ "$(md5sum < "$LIB" | cut -d' ' -f1)" = "$MD5_LIB" ] || { echo "FAIL: $LIB is not the version that ran wave 11"; exit 1; }
source "$LIB"   # sets G, M, POSE, REC, DET, AUG_DEF; cd wave11/code
set -u

# ---- pinned inputs (read on Kaya on 13 Oct 2026; the files wave 5's C and PJ used)
N_TRAIN=12439; N_SYN=13760; N_VAL=800; N_TEST=800
EP_PJ=92; EP_PS=175; EP_FT=40
MD5_JOINT=f81defc699396b920ac69e16a125af92
MD5_SYN=500f9ce402e48e7285029bc19e20e5be
MD5_TRAIN=e67f2eb08b62fe62124a35afc9d4eeab
MD5_VAL=b612245b71a57e4b9abdc2abc47172f8
MD5_TEST=f73404a27780df006107a1d9e55db263
MD5_VOCAB=9c042dcc10e8763b14af9b06cb423727
LEAK_VAL=0; LEAK_TEST=1   # synthetic sentences equal to a validation / test reference (counted 13 Oct 2026)
LOGDIR=/group/pmc099/pprasad/code/logs

# ---- task -> (arm, seed)
T=$SLURM_ARRAY_TASK_ID
STAGE=${W12H_STAGE:-1}
case "$STAGE" in
  1) SEEDS=(505 606 707) ;;
  2) SEEDS=(808 909) ;;
  *) echo "FAIL: W12H_STAGE must be 1 or 2 (got '$STAGE')"; exit 1 ;;
esac
ARMS=(C PJ PS); NS=${#SEEDS[@]}
[ "$T" -lt $(( ${#ARMS[@]} * NS )) ] || { echo "FAIL: task $T is outside the ${#ARMS[@]} x $NS grid of stage $STAGE"; exit 1; }
arm=${ARMS[$((T / NS))]}; s=${SEEDS[$((T % NS))]}
RUNS=$G/wave12/runs_h; D=$RUNS/${arm}_s${s}
VOCAB=$G/wave5/manifests/vocab.json; V="--vocab_file $VOCAB"
JOINT=$G/stitch_scale2/joint_13760.tsv; SYN=$G/stitch_scale2/syn_13760.tsv
LOG=$LOGDIR/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${T}.out
echo "stage $STAGE task $T -> arm $arm seed $s | out $D | log $LOG"
[ "${DET}" = "--deterministic True" ] || { echo "FAIL: this wave runs every arm with deterministic kernels (DET='${DET}')"; exit 1; }
# the library takes POSE from the submitting shell when it is set there (sbatch exports the environment): refuse that
[ "${POSE}" = "--num_keypoints 133 --keypoint_subset all --coord_norm frame" ] || { echo "FAIL: POSE is not the adopted pose setting (POSE='${POSE}'); unset it in the submitting shell"; exit 1; }
[ ! -e "$D/DONE" ] || { echo "FAIL: $D is already DONE; a finished arm is never run again"; exit 1; }
[ -f "$LOG" ] || { echo "FAIL: this job's log is not at $LOG (the transfer gate reads it)"; exit 1; }

# ---- G-code: the snapshot files this job runs are the ones that ran wave 5's C and PJ
md5sum -c - <<'MD5' || { echo "FAIL: the code snapshot differs from the pinned wave-5 code"; exit 1; }
75783cd2aaddb8d5822f0c5d1a4f10e5  train_ctc.py
2e54dd6d2a5e3f99572e74bb086c749e  config.py
1187c9ff21c8de97b6ba98dfedf3897f  slt/data.py
d41145cd1acabb5a89349e6b6a5c51ef  slt/reference_model.py
076d9b1ac5e0008a895b3a5bcfe4f470  slt/model.py
8399ac01b7f856e774b3f31d87dcf385  slt/engine.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
6553f52a3f0fbeb57feef6510289d66c  tools/eval_beam.py
2fd20e3989b8e20a9857f6d90106da42  weight_average.py
MD5

# ---- G-data: exact files, exact row counts, every clip on /group, no validation or test clip in training
for f in "$JOINT" "$SYN" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" "$VOCAB"; do
  [ -s "$f" ] || { echo "FAIL: $f missing or empty"; exit 1; }
done
md5sum -c - <<MD5 || { echo "FAIL: an input file is not the pinned one"; exit 1; }
$MD5_JOINT  $JOINT
$MD5_SYN  $SYN
$MD5_TRAIN  $M/train.tsv
$MD5_VAL  $M/val.tsv
$MD5_TEST  $M/test.tsv
$MD5_VOCAB  $VOCAB
MD5
pin_pose
assert_group "$JOINT" "$SYN" "$M/val.tsv" "$M/test.tsv"
python - "$JOINT" "$SYN" "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" \
         "$N_TRAIN" "$N_SYN" "$N_VAL" "$N_TEST" "$LEAK_VAL" "$LEAK_TEST" <<'PYGATE'
import os, sys
joint, syn, train, val, test = ([l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8") if "\t" in l]
                                for p in sys.argv[1:6])
n_train, n_syn, n_val, n_test, leak_val, leak_test = map(int, sys.argv[6:12])
assert (len(train), len(syn), len(val), len(test)) == (n_train, n_syn, n_val, n_test), \
    f"row counts {len(train)}/{len(syn)}/{len(val)}/{len(test)}"
assert len(joint) == n_train + n_syn, f"joint has {len(joint)} rows"
assert joint[:n_train] == train, "the first rows of the joint corpus are not the real training manifest"
assert joint[n_train:] == syn, "the last rows of the joint corpus are not the synthetic manifest"
name = lambda rows: {os.path.basename(p) for p, _ in rows}
assert len(name(joint)) == len(joint), "duplicate clip names in the joint corpus"
held = name(val) | name(test)
assert len(held) == n_val + n_test, "validation and test share a clip"
assert not (name(joint) & held), f"{len(name(joint) & held)} validation/test clips are in the training corpus"
norm = lambda t: " ".join(t.lower().split())
syn_text = {norm(t) for _, t in syn}
lv = len(syn_text & {norm(t) for _, t in val}); lt = len(syn_text & {norm(t) for _, t in test})
assert (lv, lt) == (leak_val, leak_test), f"synthetic sentences equal to a validation/test reference: {lv}/{lt}"
print(f"G-data OK: real {len(train)} + synthetic {len(syn)} = joint {len(joint)}; val {len(val)}, test {len(test)}; "
      f"0 held-out clips in training; synthetic sentences equal to a val/test reference {lv}/{lt} (as recorded)")
PYGATE

# ---- a task that failed earlier (no DONE marker) is resubmitted whole (pre-registration section 6, rule 4): its partial
# folders are moved aside, so that no file of the failed attempt can pass a gate below or reach the read-out
mkdir -p "$RUNS"
for x in "$D" "$RUNS/pre_${arm}_s${s}"; do
  if [ -e "$x" ]; then mv "$x" "$x.partial_${SLURM_ARRAY_JOB_ID}_${T}"; echo "moved the partial folder of an earlier attempt to $x.partial_${SLURM_ARRAY_JOB_ID}_${T}"; fi
done

# ---- pre-training (PJ, PS): the FINAL checkpoint is carried forward, never one chosen on validation
EXTRA=""
if [ "$arm" != C ]; then
  if [ "$arm" = PJ ]; then PRE_M=$JOINT; EP=$EP_PJ; else PRE_M=$SYN; EP=$EP_PS; fi
  n=$(wc -l < "$PRE_M"); n=$((n))
  echo "=== PRE-TRAIN ${arm} (seed $s): $n clips of $(basename "$PRE_M") x $EP epochs = $((n * EP)) clip-passes," \
       "$(( (n + 63) / 64 * EP )) steps at batch 64 ==="
  PRE=$RUNS/pre_${arm}_s${s}
  python train_ctc.py --train_manifest "$PRE_M" --val_manifest "$M/val.tsv" $POSE $REC $DET $AUG_DEF $V \
    --epochs "$EP" --eval_every "$EP" --save_last True --seed "$s" --out_dir "$PRE" \
    || { echo "[FAIL] pre-train ${arm}"; exit 1; }
  python - "$PRE/last.pt" "$VOCAB" "$EP" "$PRE_M" "$s" <<'PY'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ck["vocab"] == json.load(open(sys.argv[2])), "pre-train vocabulary != the fixed vocabulary"
assert ck["epoch"] == int(sys.argv[3]), (ck["epoch"], sys.argv[3])
c = ck["cfg"]
assert c["train_manifest"] == sys.argv[4] and c["seed"] == int(sys.argv[5]) and c["deterministic"] is True, \
    (c["train_manifest"], c["seed"], c["deterministic"])
assert not c.get("test_manifest"), "the pre-training stage must not read the test manifest"
print(f"last.pt OK: epoch {ck['epoch']}, seed {c['seed']}, {c['train_manifest']}, deterministic, "
      f"vocabulary identical to vocab.json ({len(ck['vocab'])})")
PY
  EXTRA="--resume_checkpoint $PRE/last.pt"
fi

# ---- main training: identical in every arm; only the initialisation differs
w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" "$EP_FT" -- $AUG_DEF $V $EXTRA \
  || { echo "[FAIL] main training ${arm}"; exit 1; }
# w4run only warns when a beam decode fails; every decode the read-out uses is required here
for f in beam5_val_best beam5_test_best beam5_val_soup_last5 beam5_test_soup_last5; do
  [ -s "$D/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }
done

# ---- G-transfer: PJ / PS start from the whole pre-trained model with the same vocabulary; C starts from scratch
python - "$LOG" "$arm" <<'PY'
import re, sys
log, arm = open(sys.argv[1], errors="replace").read(), sys.argv[2]
m = re.findall(r"loaded (\d+)/(\d+) shape-matched tensors", log)
if arm == "C":
    assert "[resume]" not in log and not m, "the control was warm-started"
    print("G-transfer OK: the control trained from scratch (no [resume] line in its log)")
else:
    assert "[resume] vocabulary identical" in log, "the main training did not report an identical vocabulary"
    assert len(m) == 1 and m[0][0] == m[0][1], f"not every tensor transferred exactly once: {m}"
    print(f"G-transfer OK: {m[0][0]}/{m[0][1]} tensors, vocabulary identical (the CTC head is re-initialised by design)")
PY
echo "${SLURM_ARRAY_JOB_ID}_${T} $(hostname) stage $STAGE" > "$D/JOB"
# the script exactly as SLURM ran it; a record only, so a failed copy must not cost a finished arm its DONE marker
cp "$0" "$D/w12h_pretrain.sbatch.ran" || echo "[WARN] could not copy the job script ($0) into $D"
touch "$D/DONE"   # the read-out uses only arms carrying this marker
echo "W12H_DONE ${arm} seed ${s}"
