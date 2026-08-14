#!/bin/bash -l
# Semantic-adequacy grading (0-4) of the three ladder models' beam-5 outputs, using a local
# HF instruct model as judge on a CPU work node (no GPU contention, no external Ollama).
#SBATCH --job-name=grade-sem
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=8 HF_HOME=/scratch/pmc099/pprasad/hf_cache
S=/group/pmc099/pprasad/code/results/semantic_grading
H=/scratch/pmc099/pprasad/dwpose/semantic
MODEL=Qwen/Qwen2.5-3B-Instruct
for f in hyps_dwpose_ctc hyps_vlp hyps_rgbfusion; do
  echo "===== grading $f (beam-5) ====="
  python "$S/grade_semantic_hf.py" --in "$H/$f.jsonl" --field hyp_beam --model "$MODEL" --device cpu \
    --out "$H/$f.graded_hf.csv"
done
cp "$H"/*.graded_hf.csv /group/pmc099/pprasad/semantic_hyps/ 2>/dev/null || true
echo "GRADE_SEM_DONE"
