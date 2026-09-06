#!/bin/bash -l
# LOW-RESOURCE REGULARISATION SWEEP — the cheapest untested lever on the board.
#
# WHY: the project log has long asserted "remaining levers are noise-level knobs (LR/depth/dropout)".
# That looks wrong. Araabi & Monz (COLING 2020) sweep IWSLT14 De-En subsampled to EXACTLY our
# scale; at 10k pairs **dropout 0.1 -> 0.3 alone = +4.3 BLEU** (11.9 -> 17.0), the full recipe
# gives 11.9 -> 18.7, and the capacity trim is worth only +0.8 of that while the REGULARISATION
# is worth +5.6. Escolano et al. (SIGUL 2024) report +4 BLEU at 10k sentences from residual
# dropout alone. Our config.py sits at their *baseline* column: dropout 0.1, label_smoothing 0.1,
# no activation dropout / LayerDrop / word dropout / residual dropout.
#
# Four independent corroborations that we are under-regularised:
#   * our own GASLT reference recipe uses dropout 0.5 / label smoothing 0.4
#   * SWA improved a member (10.71 -> 10.885) = noisy-late-training signature
#   * VLP-80 > VLP-40, i.e. more pretraining was not saturated
#   * validation peaks at epoch 8-24 of 40 across the whole P-series, then overfits
#   * R-Drop being NEUTRAL at alpha 1.0 is CONSISTENT with this: at dropout 0.1 the two
#     forward passes barely differ, so there is nothing for the consistency loss to act on.
#
# PROTOCOL: ordered greedy (the paper's own method) — one knob at a time, each fixed once
# chosen, chosen on VALIDATION only. Everything else is the established DWPose+CTC recipe with
# ctc_weight held at 0.3 (already swept and confirmed optimal; re-tuning it here would confound).
#
# EPOCHS 60, NOT 40, for every arm including the control: regularised models need more updates
# and their validation peak moves right, so a 40-epoch budget would under-train exactly the arms
# under test and manufacture a false negative. Matched across arms, so the comparison stays fair.
# Selection is on best-validation checkpoint, so extra epochs cannot flatter a late-overfitting arm.
#
# On `gpu` (3-day cap). First routed to ondemand-gpu, but that partition scheduled it a day
# out despite having ZERO jobs running -- it is gated by something other than availability.
# `run` skips any arm whose progress.json already exists, so a resubmit resumes, never restarts.
#
# ZERO CODE CHANGES: train_ctc.py auto-exposes every Config dataclass field as a CLI flag, so
# dropout / embedding_dropout / label_smoothing are already sweepable. Nothing in slt/ is touched
# — which also means this cannot disturb the three jobs currently queued that import those modules.
#SBATCH --job-name=reg-sweep
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
RS=$DE/runs/regsweep
cd "$G/code/phase2_baseline"
mkdir -p "$RS"

COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
BASE="--batch_size 64 --epochs 60 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
      --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

run () {   # $1 = arm name, rest = extra flags
  local name="$1"; shift
  if [ -f "$RS/$name/progress.json" ]; then echo "=== ARM $name (cached) ==="; return; fi
  echo "=== ARM $name : $* ==="
  python train_ctc.py $COM $POSE $BASE "$@" --out_dir "$RS/$name"
  python tools/eval_beam.py --ckpt "$RS/$name/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
}

# pick the arm with the highest VALIDATION BLEU-4 (never test)
pick () {
  python - "$@" <<'PY'
import sys, os, json
best = None
for spec in sys.argv[1:]:
    tag, d = spec.split("=", 1)
    try:
        v = json.load(open(os.path.join(d, "progress.json")))["best"]["bleu4"]
    except Exception:
        v = -1.0
    print("    val_best %-28s %.3f" % (tag, v), file=sys.stderr)
    if best is None or v > best[1]:
        best = (tag, v)
print(best[0])
PY
}

echo "########## ROUND A — dropout (the paper's single biggest step: +4.3 at 10k) ##########"
run d01_control --dropout 0.1
run d02         --dropout 0.2
run d03         --dropout 0.3
A=$(pick "0.1=$RS/d01_control" "0.2=$RS/d02" "0.3=$RS/d03")
echo ">>> ROUND A winner: dropout=$A"

echo "########## ROUND B — embedding_dropout, holding dropout=$A ##########"
run e02 --dropout "$A" --embedding_dropout 0.2
run e03 --dropout "$A" --embedding_dropout 0.3
B=$(pick "0.1=$RS/d01_control" "0.2=$RS/e02" "0.3=$RS/e03")
# the control column only stands in for "keep 0.1" when dropout is unchanged; re-run if it moved
if [ "$B" = "0.1" ] && [ "$A" != "0.1" ]; then
  run e01 --dropout "$A" --embedding_dropout 0.1
  B=$(pick "0.1=$RS/e01" "0.2=$RS/e02" "0.3=$RS/e03")
fi
echo ">>> ROUND B winner: embedding_dropout=$B"

echo "########## ROUND C — label_smoothing, holding dropout=$A embedding_dropout=$B ##########"
# NOTE: label smoothing also acts as an implicit LENGTH penalty (arXiv 2205.00659), and our
# decode-time LP is already tuned, so this knob and LP act on the same axis. Report the
# jointly-selected point; do not stack the two corrections blindly.
run l02 --dropout "$A" --embedding_dropout "$B" --label_smoothing 0.2
run l03 --dropout "$A" --embedding_dropout "$B" --label_smoothing 0.3
C=$(pick "0.2=$RS/l02" "0.3=$RS/l03")
echo ">>> ROUND C winner: label_smoothing=$C"

echo "########## SUMMARY (validation-selected; test shown for the ladder only) ##########"
for R in d01_control d02 d03 e01 e02 e03 l02 l03; do
  [ -f "$RS/$R/progress.json" ] || continue
  python -c "
import json
d=json.load(open('$RS/$R/progress.json')); t=d.get('test',{})
print('%-12s val_best=%6.3f @ep%-3d   test_greedy=%6.3f  wer=%.3f' % (
    '$R', d['best']['bleu4'], d['best']['epoch'],
    t.get('bleu4', float('nan')), t.get('wer', float('nan'))))
"
done
echo "in-batch control is d01_control. Historical references: DWPose+CTC ~9.6-9.9 beam-5, best single 10.71 (VLP-80)."
echo "REG_SWEEP_DONE"
