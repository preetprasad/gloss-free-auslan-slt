#!/bin/bash -l
# ROUND 2 — WAVE 2 (fingerspelled-letter fix) + WAVE 1 STAGE 2 (60-epoch budget), one batch.
# Pre-registration, written before any arm was trained: results/PREREG_WAVE2_R2_2026-09-22.md
# Protocol: the project log §9 "ROUND-2 DECISION PROTOCOL".
#
# ARRAY TASK -> (question, seed):
#   0,1,2  WAVE 2, seeds 42/101/202
#            w2A   12,439 real + 13,760 synthetic built with the CURRENT lexicon   (in-batch control, 40 ep)
#            w2LF  the SAME 13,760 sentences rebuilt with --letter_fix             (40 ep)
#          The two corpora are paired row-for-row by wave2_prep.sbatch: same sentences, same order, same budget,
#          only WHICH SIGN IS SHOWN differs. 8.67% of segments in the control are fingerspelled single letters
#          ("I" as the LETTER I, "a" as the LETTER A); in w2LF that is 0%, with "i" rendered as ME.
#   3,4    WAVE 1 STAGE 2, seeds 303/404, on the adopted recipe's manifest
#            A2    40 epochs (fresh control -- Wave-1 controls are a different batch and cross-batch
#                  same-config gaps reach 1.18 BLEU-4)
#            L60   60 epochs
# Control first in every task: a timeout then loses the treatment, never the comparator.
#
# STAGE-1 FUTILITY (pre-registered, Wave 2 only): stop w2LF if its 3-seed mean paired delta (w2LF - w2A) in
# beam-5 VALIDATION BLEU-4 of best.pt is <= 0. No test number enters that decision.
# DETERMINISM: off in every arm (the R2-P0 A/A smoke has not been run; the protocol forbids mixing modes).
#SBATCH --job-name=wave2-r2
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
SRC=$G/stitch_scale2; M="$SRC/manifests"; LF=$G/wave2_lf
OUT=$G/wave2_r2; RUNS="$OUT/runs"; mkdir -p "$RUNS"

case $SLURM_ARRAY_TASK_ID in
  0) S=42;  MODE=wave2 ;;
  1) S=101; MODE=wave2 ;;
  2) S=202; MODE=wave2 ;;
  3) S=303; MODE=stage2 ;;
  4) S=404; MODE=stage2 ;;
  *) echo "bad array id"; exit 1 ;;
esac
echo "array task $SLURM_ARRAY_TASK_ID -> $MODE seed $S on $(hostname)"
md5sum config.py slt/data.py train_ctc.py tools/stitch_lexicon.py tools/eval_beam.py weight_average.py

# --- inputs: assert what we actually got, never what the filename implies ---
python - "$MODE" "$LF/joint_a.tsv" "$LF/joint_lf.tsv" "$SRC/joint_13760.tsv" "$M/val.tsv" "$M/test.tsv" <<'PYCHK'
import os, sys
mode, ja, jlf, jadopted, val, test = sys.argv[1:7]
rows = lambda p: [l.split("\t", 1)[0] for l in open(p) if "\t" in l]
need = [ja, jlf] if mode == "wave2" else [jadopted]
va, te = rows(val), rows(test)
ok = len(va) == 800 and len(te) == 800
ok &= len(set(map(os.path.basename, va)) & set(map(os.path.basename, te))) == 0
for p in need:
    tr = rows(p)
    syn = sum("/corpus" in c for c in tr)
    miss = sum(not os.path.exists(c) for c in tr)
    print(f"{os.path.basename(p)}: {len(tr)} rows ({syn} synthetic), {miss} missing")
    ok &= len(tr) == 26199 and syn == 13760 and miss == 0
print(f"val {len(va)} | test {len(te)} | val/test overlap 0 | inputs {'OK' if ok else 'BAD'}")
sys.exit(0 if ok else 1)
PYCHK

# Wave 2 only: re-assert the pairing here, on the node that will train on it.
if [ "$MODE" = wave2 ]; then
python - "$LF/joint_a.tsv" "$LF/joint_lf.tsv" <<'PYPAIR'
import sys
rd = lambda p: [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8")][-13760:]
a, b = rd(sys.argv[1]), rd(sys.argv[2])
same = sum(x[1] == y[1] for x, y in zip(a, b))
print(f"paired synthetic rows {len(a)} | identical English text {same}/{len(a)}")
sys.exit(0 if same == len(a) == 13760 else 1)
PYPAIR
fi

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
AUG="--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1"

run () {   # $1 arm  $2 train manifest  $3 epochs
  local arm=$1 tr=$2 ep=$3
  local d="$RUNS/${arm}_s${S}"
  echo "=== ARM ${arm} (seed $S, epochs $ep, manifest $(basename "$tr")) ==="
  python train_ctc.py --train_manifest "$tr" $POSE $REC $AUG --epochs "$ep" --seed "$S" --swa_last_k 5 \
    --out_dir "$d" || { echo "[FAIL] train ${arm}"; return 1; }
  python - "$d/best.pt" "$S" "$tr" "$ep" <<'PYCFG' || { echo "[FAIL] cfg ${arm}"; return 1; }
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
want = dict(seed=int(sys.argv[2]), train_manifest=sys.argv[3], epochs=int(sys.argv[4]),
            pose_aug=True, aug_speed=0.2, aug_rotate_deg=10.0, aug_scale=0.1,
            aug_hand_drop=0.0, aug_hand_span=0.0,
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

if [ "$MODE" = wave2 ]; then
  ARMS="w2A w2LF"
  run w2A  "$LF/joint_a.tsv"  40 || echo "[FAIL] arm w2A"
  run w2LF "$LF/joint_lf.tsv" 40 || echo "[FAIL] arm w2LF"
else
  ARMS="A2 L60"
  run A2  "$SRC/joint_13760.tsv" 40 || echo "[FAIL] arm A2"
  run L60 "$SRC/joint_13760.tsv" 60 || echo "[FAIL] arm L60"
fi

echo "=== TASK SUMMARY $MODE seed $S (readouts only; no decision is made from one seed) ==="
python - "$RUNS" "$S" $ARMS <<'PYSUM'
import json, os, sys
runs, s = sys.argv[1], sys.argv[2]; arms = sys.argv[3:]
KEY = "tokenize=13a;detok=True"
def b5(d, split, ck):
    try:
        return json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))["beam_result"]["metrics"][KEY]
    except Exception:
        return float("nan")
rows = {}
for arm in arms:
    d = os.path.join(runs, f"{arm}_s{s}")
    try:
        p = json.load(open(os.path.join(d, "progress.json")))
    except Exception as e:
        print(f"{arm}: NO RESULT ({e})"); continue
    rows[arm] = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                     greedy_test=p.get("test", {}).get("bleu4", float("nan")),
                     beam5_val=b5(d, "val", "best"), beam5_test=b5(d, "test", "best"),
                     soup_beam5_val=b5(d, "val", "soup_last5"), soup_beam5_test=b5(d, "test", "soup_last5"))
    print(arm, json.dumps(rows[arm]))
ctrl, treat = arms[0], arms[1]
if ctrl in rows and treat in rows:
    print(f"{treat} - {ctrl} (seed {s}): beam-5 val {rows[treat]['beam5_val'] - rows[ctrl]['beam5_val']:+.3f} | "
          f"beam-5 test {rows[treat]['beam5_test'] - rows[ctrl]['beam5_test']:+.3f}")
json.dump(rows, open(os.path.join(runs, f"summary_s{s}.json"), "w"), indent=1)
PYSUM
echo "WAVE2_TASK_DONE $MODE seed $S"
