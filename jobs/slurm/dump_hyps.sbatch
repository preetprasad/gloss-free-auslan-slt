#!/bin/bash -l
# Dump test (ref, hyp) pairs for semantic grading — CPU work node (no GPU contention).
# Three checkpoints across the ladder: DWPose+CTC 9.63, VLP 10.27, RGB-fusion 10.335.
#SBATCH --job-name=dump-hyps
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
DE=/scratch/pmc099/pprasad/dwpose
OUT=$DE/semantic; mkdir -p "$OUT"
cd /group/pmc099/pprasad/code/phase2_baseline
export PYTHONPATH=.
DUMP=/group/pmc099/pprasad/code/results/semantic_grading/dump_hyps.py
COMMON="--test_manifest $DE/manifests/test.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame --beam 5 --device cpu"

echo "=== DWPose+CTC baseline ==="
python "$DUMP" --ckpt /group/pmc099/pprasad/comm_dwpose_ctc/best.pt $COMMON --out "$OUT/hyps_dwpose_ctc.jsonl"
echo "=== VLP best (10.27) ==="
python "$DUMP" --ckpt /group/pmc099/pprasad/comm_dwpose_vlp_ctc/best.pt $COMMON --out "$OUT/hyps_vlp.jsonl"
echo "=== RGB fusion best (10.335) ==="
python "$DUMP" --ckpt /group/pmc099/pprasad/comm_dwpose_rgb_fusion/best.pt --rgb_feat_dir "$DE/rgb_feats" $COMMON --out "$OUT/hyps_rgbfusion.jsonl"
cp -r "$OUT" /group/pmc099/pprasad/semantic_hyps
echo "DUMP_HYPS_DONE"
