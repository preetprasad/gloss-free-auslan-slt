#!/bin/bash -l
# Build a frozen, versioned text-semantic target cache on a CPU worker.
#SBATCH --job-name=auslan-sentence-targets
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
python build_sentence_embeddings.py \
  --manifest "$DE/manifests/train.tsv" \
  --model sentence-transformers/all-MiniLM-L12-v2 \
  --cache "$DE/semantic_targets/all_minilm_l12_train.pt"
echo "SENTENCE_TARGETS_DONE"
