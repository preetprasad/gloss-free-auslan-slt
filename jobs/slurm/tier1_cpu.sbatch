#!/bin/bash -l
# Tier-1 training-free BLEU boosters: (1) weight-average the 3 DWPose+CTC seeds; (2) beam/LP
# re-tune of the 10.76 ensemble, SELECTED ON VAL then confirmed on TEST. Reuses checkpoints.
#SBATCH --job-name=tier1
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer sacrebleu 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd /group/pmc099/pprasad/code/phase2_baseline
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --test_manifest $DE/manifests/test.tsv"
ENS="$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt"
RGB="--rgb_feat_dir $DE/rgb_feats --rgb_feature_dim 1152"

echo "###### (1) WEIGHT-AVERAGE the 3 DWPose+CTC seeds ######"
python weight_average.py --ckpts "$G/comm_dwpose_ctc_seed13/best.pt,$G/comm_dwpose_ctc_seed42/best.pt,$G/comm_dwpose_ctc_seed7/best.pt" --out "$DE/runs/soup_seeds.pt"
echo "--- souped seeds, TEST beam-5 ---"
python ensemble_decode.py --ckpts "$DE/runs/soup_seeds.pt" $POSE --beam 5 --device cpu || echo "SOUP_EVAL_FAILED (continuing)"

echo "###### (2) ENSEMBLE beam/LP re-tune (select on VAL) ######"
for BEAM in 5 10; do for LP in 0.6 1.0 1.2; do
  echo "--- VAL beam=$BEAM lp=$LP ---"
  python ensemble_decode.py --ckpts "$ENS" $POSE $RGB --beam $BEAM --length_penalty $LP \
    --eval_manifest "$DE/manifests/val.tsv" --device cpu
done; done
echo "###### best-looking configs on TEST (confirm) ######"
for CFG in "10 1.0" "10 1.2" "5 1.0"; do set -- $CFG
  echo "--- TEST beam=$1 lp=$2 ---"
  python ensemble_decode.py --ckpts "$ENS" $POSE $RGB --beam $1 --length_penalty $2 --device cpu
done
echo "TIER1_DONE"
