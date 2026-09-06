#!/bin/bash -l
# Learned-graph keypoint front-end — FAIR RE-RUN after fixing the init confound.
#
# The 2026-09-04 run (job 1171052) scored layers2=7.53 / layers3=7.82 vs an in-batch control
# of ~9.2, but that result was confounded: PoseSLT.reset_parameters() ran a blanket xavier
# over every dim>1 parameter, which included the graph's (K,K) adjacency A — wiping its
# near-identity init so the block started as a DESTRUCTIVE random keypoint-mixer instead of
# a residual no-op. That tested "graph front-end from random init", not "skeleton structure".
#
# Fixed in slt/model.py: A is now initialised from the real COCO-WholeBody-133 anatomical
# skeleton and excluded from the xavier re-init. This re-run is the fair test.
# Arms: in-batch control (graph off) + layers 2 + layers 3. Compare ONLY to the in-batch
# control — batch-to-batch seed/env variance was 0.4-0.8 BLEU last round.
#SBATCH --job-name=graph-fair
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

echo "=== SMOKE GATE: adjacency must SURVIVE reset_parameters (this is the whole fix) ==="
python - <<'PY'
import torch
from config import Config
from slt.reference_model import build_model
from slt.model import _coco_wholebody_133_adjacency

A0 = _coco_wholebody_133_adjacency()
assert A0.shape == (133, 133) and torch.equal(A0, A0.T), "adjacency malformed"
cfg = Config(pose_graph=True, pose_graph_layers=2, num_keypoints=133,
             keypoint_subset="all", coord_norm="frame")
F = cfg.feature_dim(); print("feature_dim =", F)
m = build_model(F, 100, cfg)
assert m.pose_graph is not None, "pose_graph front-end not attached!"
A = m.pose_graph.A[0].detach()
# self-loops and anatomical edges must still be ~1, non-edges ~0 (xavier would flatten all)
assert torch.allclose(A.diag(), torch.ones(133), atol=0.05), "A diagonal wiped -> confound is BACK"
assert A[5, 7] > 0.8, "shoulder-elbow edge wiped -> confound is BACK"
assert abs(A[5, 100]) < 0.1, "non-edge is not ~0 -> A was re-initialised"
src = torch.randn(2, 20, F); pad = torch.zeros(2, 20, dtype=torch.bool)
mem = m.encode(src, pad)
assert mem.shape == (2, 20, cfg.d_model), mem.shape
print("SMOKE_OK skeleton init survives reset | encode ->", tuple(mem.shape),
      "| graph params:", sum(p.numel() for p in m.pose_graph.parameters()))
PY

COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== ARM 0: in-batch CONTROL (pose_graph off) ==="
python train_ctc.py $COM $POSE $REC --pose_graph False --out_dir "$DE/runs/graphfair_control"
python tools/eval_beam.py --ckpt "$DE/runs/graphfair_control/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
cp -r "$DE/runs/graphfair_control" "$G/comm_dwpose_graphfair_control" 2>/dev/null || true

for L in 2 3; do
  echo "=== ARM: skeleton-init graph front-end, layers=$L ==="
  python train_ctc.py $COM $POSE $REC --pose_graph True --pose_graph_layers $L \
    --out_dir "$DE/runs/graphfair_$L"
  python tools/eval_beam.py --ckpt "$DE/runs/graphfair_$L/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
  cp -r "$DE/runs/graphfair_$L" "$G/comm_dwpose_graphfair_$L" 2>/dev/null || true
done

echo "=== SUMMARY (compare arms to the in-batch control above, not to 9.63) ==="
for R in graphfair_control graphfair_2 graphfair_3; do
  python -c "
import json,sys
p='$DE/runs/$R/progress.json'
try:
    d=json.load(open(p)); t=d.get('test',{})
    print('$R', 'val_best=%.3f@ep%d' % (d['best']['bleu4'], d['best']['epoch']),
          'test_bleu4=%.3f' % t.get('bleu4', float('nan')), 'wer=%.3f' % t.get('wer', float('nan')))
except Exception as e:
    print('$R', 'NO RESULT', e)
"
done
echo "GRAPH_FAIR_DONE"
