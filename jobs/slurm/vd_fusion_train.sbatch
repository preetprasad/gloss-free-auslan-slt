#!/bin/bash -l
# G1 -- two RGB-fusion members trained on vd_train only, warm-started from vdB (results/PREREG_VD_XMODAL_2026-09-25.md).
# S6 fusion recipe (gpu_batch_10h.sbatch): batch 32, 40 ep, ctc 0.3, lr 5e-4 cosine, warmup 800. Task 0 = seed 303
# (init vdB_s303), task 1 = seed 404 (init vdB_s404). Frozen wave-5 code snapshot. Determinism is deliberately OFF
# (these are ensemble members, not a paired comparison; the S6 fusion members were trained the same way).
#SBATCH --job-name=vd-fusion
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
DET=""; source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(303 404); s=${SEEDS[$T]}
INIT=$G/wave3_vd/runs/vdB_s$s/best.pt; RGB=$G/scratch_backup/dwpose/rgb_feats; OUT=$G/vd_xmodal/runs/fus_s$s
[ -s "$INIT" ] || { echo "FAIL: missing $INIT"; exit 1; }
N=$(find "$RGB" -name "rgb_*.npz" | wc -l); echo "rgb feature files: $N"; [ "$N" -ge 14041 ] || { echo "FAIL: rgb feats $N < 14041"; exit 1; }
for f in vd_train vd_val vd_test; do [ -s "$M/$f.tsv" ] || { echo "FAIL: missing $M/$f.tsv"; exit 1; }; done
assert_group "$M/vd_train.tsv" "$M/vd_val.tsv" "$M/vd_test.tsv"
python - "$M/vd_train.tsv" "$RGB" <<'PY'
import os, sys
rows = [l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l]
miss = [r for r in rows if not os.path.exists(os.path.join(sys.argv[2], "rgb_" + os.path.basename(r)))]
print(f"vd_train {len(rows)} clips; RGB features missing for {len(miss)}"); sys.exit(1 if len(miss) > len(rows) * 0.01 else 0)
PY
mkdir -p "$(dirname "$OUT")"
python train_ctc.py --train_manifest "$M/vd_train.tsv" --val_manifest "$M/vd_val.tsv" --test_manifest "$M/vd_test.tsv" \
  $POSE --rgb_feat_dir "$RGB" --rgb_feature_dim 1152 --init_encoder "$INIT" --seed $s --batch_size 32 --epochs 40 \
  --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --num_beams 5 --length_penalty 0.6 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$OUT" 2>&1 | tee "$OUT.train.log"
# warm-start gate: every base tensor transferred except the vocabulary-sized ones
python - "$OUT.train.log" "$INIT" "$OUT/best.pt" <<'PY'
import re, sys, torch
line = next(l for l in open(sys.argv[1]) if "[init] warm-started base with" in l)
n, m = map(int, re.search(r"with (\d+)/(\d+)", line).groups())
init = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["model"]
new = torch.load(sys.argv[3], map_location="cpu", weights_only=False)["model"]
base = {k[5:]: v for k, v in new.items() if k.startswith("base.")} or new
diff = [k for k in base if k in init and tuple(init[k].shape) != tuple(base[k].shape)]
vocab_like = [k for k in diff if 2867 in tuple(init[k].shape)]
print(f"[gate] warm-started {n}/{m}; shape-differing base tensors {len(diff)}, of which vocab-sized {len(vocab_like)}: {diff}")
assert len(diff) == len(vocab_like) and m - n == len(diff), "warm start did not transfer every non-vocabulary tensor"
print("[gate] OK: every non-vocabulary base tensor was transferred")
PY
touch "$OUT/DONE"; echo "VD_FUSION_DONE s$s"
