#!/bin/bash -l
# Select decoding parameters on validation, then evaluate exactly once on test.
# This does not retrain or use test to choose beam width/length penalty.
#SBATCH --job-name=dwpose-beam
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate

CODE=/group/pmc099/pprasad/code/phase2_baseline
RUN=/group/pmc099/pprasad/comm_dwpose_full
MANIFESTS=/scratch/pmc099/pprasad/dwpose/manifests
SWEEP="$RUN/beam_sweep_validation.json"
TEST_RESULT="$RUN/beam_selected_test.json"
cd "$CODE"

python tools/tune_beam.py \
  --ckpt "$RUN/best.pt" --manifest "$MANIFESTS/val.tsv" \
  --beams 3,5,8,10,12 --length_penalties 0,0.2,0.4,0.6,0.8,1.0,1.2 \
  --batch_size 16 --device cuda --output_json "$SWEEP"

read -r beam penalty < <(python - "$SWEEP" <<'PY'
import json, sys
best = json.load(open(sys.argv[1]))["best"]
print(best["beam"], best["length_penalty"])
PY
)
python tools/eval_beam.py \
  --ckpt "$RUN/best.pt" --manifest "$MANIFESTS/test.tsv" --device cuda \
  --beam "$beam" --length_penalty "$penalty" --output_json "$TEST_RESULT"
echo "DWPOSE_BEAM_DONE beam=$beam length_penalty=$penalty"
