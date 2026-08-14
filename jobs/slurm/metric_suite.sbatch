#!/bin/bash -l
# Beyond-BLEU metric suite + seen/unseen (sentence-disjoint) breakdown on existing hyp dumps.
#SBATCH --job-name=metrics
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q sacrebleu 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; H=$DE/semantic
cd /group/pmc099/pprasad/code/phase2_baseline
for f in hyps_dwpose_ctc hyps_vlp hyps_rgbfusion; do
  python metric_suite.py --in "$H/$f.jsonl" --field hyp_beam --train_manifest "$DE/manifests/train.tsv"
done
echo "METRICS_DONE"
