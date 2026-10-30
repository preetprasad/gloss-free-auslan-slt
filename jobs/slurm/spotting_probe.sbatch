#!/bin/bash -l
# Feasibility probe: can the MM-WLAuslan ISLR encoder spot its known signs in continuous Auslan-Daily?
#SBATCH --job-name=spot-probe
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose; M=/scratch/pmc099/pprasad/mmwl; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
python tools/spotting_probe.py \
  --encoder "$G/islr_dwpose/islr_encoder_best.pt" \
  --mmwl_labels "$M/labels/Valid.json" --mmwl_npz_dir "$M/dwpose_clips/valid" \
  --ad_manifest "$DE/manifests/train.tsv" \
  --protos_per_gloss 3 --limit_ad 40 --win 16 --stride 8 --device cuda
echo "SPOT_PROBE_DONE"
