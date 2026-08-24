#!/bin/bash -l
# Locked evaluation of the validation-selected six-member S6 ensemble.
# No hyperparameters are searched here: beam-5 / LP 1.0 / official test split.
# ensemble_decode.py prints BLEU for protocol verification plus WER and S/D/I counts.
#SBATCH --job-name=wer-s6
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out

module purge
module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate

DE=/scratch/pmc099/pprasad/dwpose
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"

# S6 is the established validation-selected 11.50 BLEU-4 ensemble.
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"

python ensemble_decode.py \
  --ckpts "$S6" \
  --num_keypoints 133 \
  --keypoint_subset all \
  --coord_norm frame \
  --rgb_feat_dir "$DE/rgb_feats" \
  --rgb_feature_dim 1152 \
  --eval_manifest "$DE/manifests/test.tsv" \
  --beam 5 \
  --length_penalty 1.0 \
  --device cpu
