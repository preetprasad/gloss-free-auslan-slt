#!/bin/bash -l
# WAVE 5 step 3 -- synthetic PRE-training at fixed compute (2,400,000 clip-passes) on the first 25k / 100k / 200k /
# 300k clips, then fine-tuning with the ADOPTED recipe; C = the adopted recipe from scratch; PJ = the COMPUTE control
# (the same 2.4M clip-passes of pre-training, on the adopted joint corpus itself, so no new sentences). Task = (arm, seed):
#   STAGE 2 (pre-registered): only the arm that passed Stage-1 futility (P200), with a FRESH control C and the
#   compute control PJ, at seeds 303 / 404.  0-1 C | 2-3 P200 | 4-5 PJ
# Every arm: the same fixed vocabulary (--vocab_file), --deterministic True, the frozen wave-5 snapshot.
# Pre-registration: results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md
#SBATCH --job-name=w5-train-s2
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
T=$SLURM_ARRAY_TASK_ID; SEEDS=(303 404); ARMS=(C P200 PJ)
arm=${ARMS[$((T / 2))]}; s=${SEEDS[$((T % 2))]}
MW=$W5/manifests; RUNS=$W5/runs; mkdir -p "$RUNS"
V="--vocab_file $MW/vocab.json"; JOINT=$G/stitch_scale2/joint_13760.tsv
LOG=/group/pmc099/pprasad/code/logs/w5-train-s2-${SLURM_ARRAY_JOB_ID}_${T}.out
echo "task $T -> arm $arm seed $s"
[ -f "$MW/vocab.json" ] || { echo "FAIL: fixed vocabulary missing"; exit 1; }
assert_group "$JOINT" "$M/val.tsv" "$M/test.tsv"

EXTRA=""
if [ "$arm" != C ]; then
  if [ "$arm" = PJ ]; then SYN=$JOINT; else SYN=$MW/syn_$(( ${arm#P} * 1000 )).tsv; fi
  [ -f "$SYN" ] || { echo "FAIL: $SYN missing"; exit 1; }
  ALLOW_PREFIX="$W5S/corpora/" assert_group "$SYN"
  n=$(grep -c . "$SYN"); EP=$(( (2400000 + n / 2) / n ))
  echo "=== PRE-TRAIN ${arm} (seed $s): $n clips of $(basename "$SYN") x $EP epochs = $((n * EP)) clip-passes ==="
  PRE=$RUNS/pre_${arm}_s${s}
  python train_ctc.py --train_manifest "$SYN" --val_manifest "$M/val.tsv" $POSE $REC $DET $AUG_DEF $V \
    --epochs $EP --eval_every $EP --save_last True --seed $s --out_dir "$PRE" \
    || { echo "[FAIL] pre-train ${arm}"; exit 1; }
  python - "$PRE/last.pt" "$MW/vocab.json" "$EP" <<'PY'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ck["vocab"] == json.load(open(sys.argv[2])), "pre-train vocabulary != the fixed vocabulary"
assert ck["epoch"] == int(sys.argv[3]), (ck["epoch"], sys.argv[3])
print(f"last.pt OK: epoch {ck['epoch']}, vocabulary identical to vocab.json ({len(ck['vocab'])})")
PY
  EXTRA="--resume_checkpoint $PRE/last.pt"
fi

w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF $V $EXTRA \
  || { echo "[FAIL] fine-tune ${arm}"; exit 1; }
D=$RUNS/${arm}_s${s}   # w4run only warns when a beam decode fails; the criterion decodes are required here
for f in beam5_val_best beam5_test_best; do [ -s "$D/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done

if [ "$arm" != C ]; then   # G-transfer: identical vocabulary, and every tensor of the model loaded
  python - "$LOG" <<'PY'
import re, sys
log = open(sys.argv[1]).read()
assert "[resume] vocabulary identical" in log, "fine-tune did not report an identical vocabulary"
m = re.findall(r"loaded (\d+)/(\d+) shape-matched tensors", log)
assert m and m[-1][0] == m[-1][1], f"not every tensor transferred: {m}"
print(f"G-transfer OK: {m[-1][0]}/{m[-1][1]} tensors, vocabulary identical (the CTC head is re-initialised by design)")
PY
fi
touch "$D/DONE"   # the summary reads only arms carrying this marker
echo "W5_TRAIN_DONE ${arm} seed ${s}"
