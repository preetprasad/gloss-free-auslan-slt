#!/bin/bash -l
# Learned-graph (CTR-GCN-style) keypoint front-end — the last untouched ARCHITECTURAL wall.
# The encoder currently treats the 133 COCO-WholeBody keypoints as a flat 399-D vector, throwing
# away skeletal structure; every SOTA pose-SLT system uses graph convolutions. This adds a gated,
# dimension-preserving learned-adjacency graph block over keypoints BEFORE the encoder (input_proj
# unchanged). Matched DWPose+CTC recipe; control = 9.63 beam-5 (pose_graph off).
#SBATCH --job-name=graph-enc
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
cd "$G/code/phase2_baseline"

echo "=== SMOKE GATE: build model with pose_graph=True and run one forward (shapes/CUDA) ==="
python - <<'PY'
import torch
from config import Config
from slt.reference_model import build_model
cfg = Config(pose_graph=True, pose_graph_layers=2, num_keypoints=133,
             keypoint_subset="all", coord_norm="frame")
F = cfg.feature_dim(); print("feature_dim =", F)
m = build_model(F, 100, cfg)
assert m.pose_graph is not None, "pose_graph front-end not attached!"
src = torch.randn(2, 20, F); pad = torch.zeros(2, 20, dtype=torch.bool)
mem = m.encode(src, pad)
assert mem.shape == (2, 20, cfg.d_model), mem.shape
print("SMOKE_OK encode ->", tuple(mem.shape),
      "| graph params:", sum(p.numel() for p in m.pose_graph.parameters()))
PY

COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
for L in 2 3; do
  echo "=== DWPose+CTC + learned graph front-end, layers=$L (control off = 9.63) ==="
  python train_ctc.py $COM $POSE $REC --pose_graph True --pose_graph_layers $L \
    --out_dir "$DE/runs/ctc_graph_$L"
  python tools/eval_beam.py --ckpt "$DE/runs/ctc_graph_$L/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$DE/runs/ctc_graph_$L" "$G/comm_dwpose_ctc_graph_$L" 2>/dev/null || true
done
echo "GRAPH_ENCODER_DONE"
