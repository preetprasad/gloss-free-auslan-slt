#!/bin/bash -l
# Retrain the translator on DWPose pose (133 kpts, frame-norm) and beam-eval.
# Builds manifests inline (repointed to the DWPose npz). Submit with a dependency on
# the extraction array so it auto-runs when extraction finishes:
#   sbatch --dependency=afterany:<extract_array_jobid> kaya/train_dwpose.sbatch
#SBATCH --job-name=dwpose-train
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true   # WER metric dep (missing from env_build)
DE=/scratch/pmc099/pprasad/dwpose
mkdir -p "$DE/manifests" "$DE/runs"

python - <<'PY'
import os
CL="/scratch/pmc099/pprasad/dwpose/dwpose_clips"
SRC="/group/pmc099/pprasad/code/comm_run/manifests"
OUT="/scratch/pmc099/pprasad/dwpose/manifests"; os.makedirs(OUT, exist_ok=True)
for s in ["train","val","test"]:
    n=t=0
    with open(f"{OUT}/{s}.tsv","w") as w:
        for ln in open(f"{SRC}/{s}.tsv"):
            ln=ln.rstrip("\n")
            if not ln: continue
            t+=1; p,txt=ln.split("\t"); dp=os.path.join(CL, os.path.basename(p))
            if os.path.exists(dp): w.write(f"{dp}\t{txt}\n"); n+=1
    print(f"{s}: {n}/{t} clips have DWPose pose", flush=True)
PY

cd /group/pmc099/pprasad/code/phase2_baseline
python train.py \
  --train_manifest "$DE/manifests/train.tsv" --val_manifest "$DE/manifests/val.tsv" \
  --test_manifest  "$DE/manifests/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --epochs 40 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --out_dir "$DE/runs/comm_dwpose_full"
python tools/eval_beam.py --ckpt "$DE/runs/comm_dwpose_full/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda
cp -r "$DE/runs/comm_dwpose_full" /group/pmc099/pprasad/   # durable (scratch purges)
echo "DWPOSE_TRAIN_DONE"
