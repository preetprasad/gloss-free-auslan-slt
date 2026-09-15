#!/bin/bash -l
# WAVE 1 OF RESEARCH ROUND 2 — hand masking (R2-HM) and a 60-epoch budget (R2-L60) against the adopted recipe.
# Design: results/BLEU_PUSH_ROUND2_LEARN_FROM_FAILURES_2026-09-16.md §4. Pre-registration, written before any number:
# results/PREREG_WAVE1_R2_2026-09-16.md.
#
# ARMS (all on stitch_scale2/joint_13760.tsv = 12,439 real + 13,760 synthetic, all with pose augmentation on):
#   w1A    adopted recipe, 40 epochs — the in-batch control
#   w1H    w1A + --aug_hand_drop 0.08 --aug_hand_span 0.15 (post-normalisation hand masking; expected keep ~0.85).
#          Its mask draws come from a separate generator, so its pose-augmentation draws equal w1A's.
#   w1L60  w1A with --epochs 60 (budget + cosine schedule; warmup stays 800)
# Array task = seed (0/1/2 -> 42/101/202). Order A, H, L60: a timeout loses the lowest-value arm, never the control.
#
# READOUTS per arm: greedy val_best and greedy test (train_ctc), then beam-5 (LP 0.6) on VALIDATION and TEST for best.pt
# and for the last-5-epoch weight soup (secondary). Every checkpoint's saved cfg is asserted against the intended arm.
# STAGE-1 FUTILITY (pre-registered): an arm stops if its 3-seed mean paired delta (arm - w1A) in beam-5 VALIDATION
# BLEU-4 of best.pt is <= 0. No test number enters that decision.
# DETERMINISM: off in every arm (the A/A determinism smoke has not been run; the protocol forbids mixing modes).
#SBATCH --job-name=wave1-r2
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
SEEDS=(42 101 202); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SRC=$G/stitch_scale2; M="$SRC/manifests"; TRAIN="$SRC/joint_13760.tsv"
OUT=$G/wave1_r2; RUNS="$OUT/runs"; mkdir -p "$RUNS"
echo "array task $SLURM_ARRAY_TASK_ID -> seed $S on $(hostname)"
md5sum config.py slt/data.py train_ctc.py tools/eval_beam.py weight_average.py

# --- inputs: assert what we actually got (counts, composition, existence, no val/test overlap) ---
python - "$TRAIN" "$M/val.tsv" "$M/test.tsv" <<'PYCHK'
import os, sys
train, val, test = sys.argv[1:4]
rows = lambda p: [l.split("\t", 1)[0] for l in open(p) if "\t" in l]
tr, va, te = rows(train), rows(val), rows(test)
syn = sum("/corpus/clips/" in p for p in tr)
missing = sum(not os.path.exists(p) for p in tr + va + te)
ov = len(set(map(os.path.basename, va)) & set(map(os.path.basename, te)))
print(f"train {len(tr)} ({syn} synthetic) | val {len(va)} | test {len(te)} | missing {missing} | val/test overlap {ov}")
ok = len(tr) == 26199 and syn == 13760 and len(va) == 800 and len(te) == 800 and missing == 0 and ov == 0
sys.exit(0 if ok else 1)
PYCHK
python -c "
from config import Config
c = Config(pose_aug=True, aug_hand_drop=0.08, aug_hand_span=0.15)
assert (c.aug_speed, c.aug_rotate_deg, c.aug_scale, c.aug_hand_drop, c.aug_hand_span) == (0.2, 10.0, 0.1, 0.08, 0.15)
print('config fields OK')"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
AUG="--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1"
HM="--aug_hand_drop 0.08 --aug_hand_span 0.15"

# --- smoke gate: masked + augmented loader on REAL and SYNTHETIC clips with 8 workers on this node ---
SM="$OUT/_smoke_s${S}"; rm -rf "$SM"; mkdir -p "$SM"
python - "$TRAIN" "$SM/train.tsv" <<'PYSMOKE'
import sys
rows = [l for l in open(sys.argv[1]) if "\t" in l]
real = [r for r in rows if "/corpus/clips/" not in r][:32]
syn = [r for r in rows if "/corpus/clips/" in r][:32]
assert len(real) == 32 and len(syn) == 32
open(sys.argv[2], "w").writelines(real + syn)
PYSMOKE
head -32 "$M/val.tsv" > "$SM/val.tsv"
python train_ctc.py --train_manifest "$SM/train.tsv" --val_manifest "$SM/val.tsv" $POSE \
  --batch_size 16 --epochs 1 --device cuda --num_workers 8 --bleu_tokenize 13a --detokenize True $AUG $HM \
  --seed "$S" --out_dir "$SM/run" > "$SM/log.txt" 2>&1
grep -E "^\[(ctc|aug)" "$SM/log.txt"
grep -q "realised hand keep fraction" "$SM/log.txt" || { echo "FAIL: smoke did not log a hand keep fraction"; exit 1; }
rm -rf "$SM"

run () {   # $1 arm  $2 epochs  $3 aug_hand_drop  $4 aug_hand_span  $5.. extra train flags
  local arm=$1 ep=$2 hd=$3 hs=$4; shift 4
  local d="$RUNS/${arm}_s${S}"
  echo "=== ARM ${arm} (seed $S, epochs $ep, hand drop $hd, hand span $hs) ==="
  python train_ctc.py --train_manifest "$TRAIN" $POSE $REC $AUG --epochs "$ep" --seed "$S" --swa_last_k 5 \
    --out_dir "$d" "$@" || { echo "[FAIL] train ${arm}"; return 1; }
  python - "$d/best.pt" "$S" "$TRAIN" "$ep" "$hd" "$hs" <<'PYCFG' || { echo "[FAIL] cfg ${arm}"; return 1; }
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
want = dict(seed=int(sys.argv[2]), train_manifest=sys.argv[3], epochs=int(sys.argv[4]),
            aug_hand_drop=float(sys.argv[5]), aug_hand_span=float(sys.argv[6]),
            pose_aug=True, aug_speed=0.2, aug_rotate_deg=10.0, aug_scale=0.1,
            num_keypoints=133, keypoint_subset="all", coord_norm="frame",
            batch_size=64, lr=5e-4, lr_schedule="cosine", warmup_steps=800, rgb_feat_dir="")
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
print("checkpoint cfg", "OK" if not bad else f"MISMATCH (saved, wanted): {bad}")
sys.exit(1 if bad else 0)
PYCFG
  local swa; swa=$(ls "$d"/swa_ep*.pt | paste -sd, -)
  [ "$(echo "$swa" | tr ',' '\n' | wc -l)" -eq 5 ] || { echo "[FAIL] ${arm}: expected 5 swa checkpoints"; return 1; }
  python weight_average.py --ckpts "$swa" --out "$d/soup_last5.pt" || { echo "[FAIL] soup ${arm}"; return 1; }
  for ck in best soup_last5; do
    for split in val test; do
      python tools/eval_beam.py --ckpt "$d/$ck.pt" --manifest "$M/$split.tsv" --beam 5 --device cuda \
        --output_json "$d/beam5_${split}_${ck}.json" > "$d/beam5_${split}_${ck}.log" 2>&1 \
        || echo "[FAIL] beam5 ${split} ${ck} ${arm}"
    done
  done
}
run w1A   40 0.0  0.0        || echo "[FAIL] arm w1A"
run w1H   40 0.08 0.15 $HM   || echo "[FAIL] arm w1H"
run w1L60 60 0.0  0.0        || echo "[FAIL] arm w1L60"

echo "=== TASK SUMMARY seed $S (readouts only; the futility decision needs all three seeds) ==="
python - "$RUNS" "$S" <<'PYSUM'
import json, math, os, sys
runs, s = sys.argv[1], sys.argv[2]
KEY = "tokenize=13a;detok=True"
def b5(d, split, ck):
    try:
        return json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))["beam_result"]["metrics"][KEY]
    except Exception:
        return float("nan")
rows = {}
for arm in ("w1A", "w1H", "w1L60"):
    d = os.path.join(runs, f"{arm}_s{s}")
    try:
        p = json.load(open(os.path.join(d, "progress.json")))
    except Exception as e:
        print(f"{arm}: NO RESULT ({e})"); continue
    keep = [h["hand_keep"] for h in p["history"] if "hand_keep" in h]
    rows[arm] = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                     greedy_test=p.get("test", {}).get("bleu4", float("nan")),
                     beam5_val=b5(d, "val", "best"), beam5_test=b5(d, "test", "best"),
                     soup_beam5_val=b5(d, "val", "soup_last5"), soup_beam5_test=b5(d, "test", "soup_last5"),
                     hand_keep_mean=(sum(keep) / len(keep)) if keep else None)
    print(arm, json.dumps(rows[arm]))
for arm in ("w1H", "w1L60"):
    if arm in rows and "w1A" in rows:
        dv = rows[arm]["beam5_val"] - rows["w1A"]["beam5_val"]; dt = rows[arm]["beam5_test"] - rows["w1A"]["beam5_test"]
        print(f"{arm} - w1A (seed {s}): beam-5 val {dv:+.3f} | beam-5 test {dt:+.3f}")
json.dump(rows, open(os.path.join(runs, f"summary_s{s}.json"), "w"), indent=1)
PYSUM
echo "WAVE1_TASK_DONE seed $S"
