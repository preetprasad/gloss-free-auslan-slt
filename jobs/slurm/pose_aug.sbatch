#!/bin/bash -l
# POSE AUGMENTATION on the best stitch recipe (duration-stitch at 1.1:1).
#
# WHY. The binding constraint is visual sign IDENTIFICATION under conversational production variability:
# pose carries sign identity in citation form (40% top-1, 45-way) and conversational signing destroys 55%
# of it, a loss that is articulation/variability rather than image quality (2026-09-09 probes). The
# synthetic half of the training data is measured ~7x LESS variable than real signing (E0c), and this
# pipeline has never augmented pose at all -- the only augmentation ever tried was TEXT paraphrasing.
#
# WHAT MOVES. --pose_aug True. Per clip, per epoch, from a (seed, epoch, clip) generator:
#   speed ±20% (nearest-frame temporal resampling), in-plane rotation ±10 deg, independent x/y scale ±10%.
# Train loader only; val/test and every eval tool never augment. Verified locally 2026-09-15: flag-off and
# eval paths bit-identical to the pre-patch loader on 200/200 real clips; flag-on finite, frame ratio
# 0.79-1.20, deterministic within an epoch and re-drawn across epochs; zero keypoints and confidence untouched;
# 2-epoch CPU training smoke with 2 workers passed.
#
# DESIGN. Paired seeds 42/101/202 x {ctrl, aug}, in-batch, on the SAME 1.1:1 corpus that won on test 3/3
# (stitch_scale2/joint_13760.tsv = 12,439 real + 13,760 synthetic), identical recipe otherwise. Controls are
# re-run in-batch: same-config reruns differ 0.3-0.5 beam-5 (09-10ab), so the historical r111 numbers are
# NOT valid controls.
#
# PRE-REGISTERED RULE: aug must beat its OWN same-seed control on validation AND beam-5 test, 3/3 seeds.
# Anything less is recorded as neutral/negative. Risk stated in advance: augmentation often needs a longer
# schedule and the r111 controls already peak late (ep 30-40), so a null at 40 epochs does not rule out
# augmentation + longer training -- that would be a separate, later question.
#SBATCH --job-name=pose-aug
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
SRC=$G/stitch_scale2; M="$SRC/manifests"; TRAIN="$SRC/joint_13760.tsv"
OUT=$G/pose_aug; RUNS="$OUT/runs"; mkdir -p "$RUNS"

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
python -c "from config import Config; c = Config(pose_aug=True); print('config pose_aug fields:', c.pose_aug, c.aug_speed, c.aug_rotate_deg, c.aug_scale)"

POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
     --val_manifest $M/val.tsv --test_manifest $M/test.tsv"
AUG="--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1"

# --- smoke gate: the augmented loader must survive REAL and SYNTHETIC clips with 8 workers on this node ---
echo "=== SMOKE (32 real + 32 synthetic clips, 1 epoch, aug on) ==="
python - "$TRAIN" "$OUT/smoke_train.tsv" <<'PYSMOKE'
import sys
rows = [l for l in open(sys.argv[1]) if "\t" in l]
real = [r for r in rows if "/corpus/clips/" not in r][:32]
syn = [r for r in rows if "/corpus/clips/" in r][:32]
assert len(real) == 32 and len(syn) == 32
open(sys.argv[2], "w").writelines(real + syn)
PYSMOKE
head -32 "$M/val.tsv" > "$OUT/smoke_val.tsv"
python train_ctc.py --train_manifest "$OUT/smoke_train.tsv" --val_manifest "$OUT/smoke_val.tsv" $POSE \
  --batch_size 16 --epochs 1 --device cuda --num_workers 8 --bleu_tokenize 13a --detokenize True $AUG \
  --out_dir "$OUT/_smoke" | grep -E "^\[(ctc|aug)"
rm -rf "$OUT/_smoke"

run () {   # $1 arm  $2 seed  $3.. extra flags
  local arm=$1 seed=$2; shift 2
  echo "=== ARM $arm (seed $seed) ==="
  python train_ctc.py --train_manifest "$TRAIN" $POSE $REC --seed "$seed" --out_dir "$RUNS/$arm" "$@"
  python tools/eval_beam.py --ckpt "$RUNS/$arm/best.pt" --manifest "$M/test.tsv" --beam 5 --device cuda \
    --output_json "$RUNS/$arm/beam5_test.json" || true
}
for S in 42 101 202; do   # interleaved, so a timeout costs both arms equally
  run "pa_ctrl_s$S" "$S"
  run "pa_aug_s$S"  "$S" $AUG
done

echo "=== SUMMARY (pre-registered: aug beats its own same-seed ctrl on validation AND beam-5 test, 3/3) ==="
python - "$RUNS" <<'PYSUM'
import json, os, sys
runs = sys.argv[1]
def get(arm):
    d = json.load(open(os.path.join(runs, arm, "progress.json")))
    b = json.load(open(os.path.join(runs, arm, "beam5_test.json")))
    return d["best"]["bleu4"], d.get("test", {}).get("bleu4", float("nan")), b["beam_result"]["metrics"]["tokenize=13a;detok=True"]
wins = 0
for s in (42, 101, 202):
    try:
        cv, cg, cb = get(f"pa_ctrl_s{s}"); av, ag, ab = get(f"pa_aug_s{s}")
    except Exception as e:
        print(f"s{s}: NO RESULT ({e})"); continue
    win = av > cv and ab > cb; wins += win
    print(f"s{s}: val {cv:6.3f} -> {av:6.3f} ({av-cv:+.3f}) | greedy {cg:6.3f} -> {ag:6.3f} ({ag-cg:+.3f}) | "
          f"beam5 {cb:5.2f} -> {ab:5.2f} ({ab-cb:+.2f}) | {'WIN' if win else 'no'}")
print(f"pre-registered rule met in {wins}/3 seeds -> {'ADOPT' if wins == 3 else 'do NOT adopt'}")
PYSUM
echo "POSE_AUG_DONE"
