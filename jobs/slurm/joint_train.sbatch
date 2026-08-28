#!/bin/bash -l
# DATA-SCALING SWING, step 4/4 — joint VLP+CTC on Communication+News (22,088 clips), the exact
# VLP->CTC recipe that scored 10.27 single on Comm-only. Only the training data changes (Comm ->
# joint), so this cleanly isolates the DATA-SCALING effect. Selection on Comm val; evaluation on the
# Communication official TEST (unchanged) so the number is comparable to the 11.80 pose result.
#SBATCH --job-name=joint-train
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
COM="--train_manifest $DE/manifests/joint_train.tsv --val_manifest $DE/manifests/joint_val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"

echo "=== VLP pretrain on joint (40 ep) ==="
python pretrain_vlp.py $COM $POSE --seed 42 --batch_size 64 --epochs 40 --lr 5e-4 \
  --lr_schedule cosine --device cuda --num_workers 8 --out_dir "$DE/runs/vlp_joint"

echo "=== CTC on joint, warm-started from joint-VLP encoder (40 ep); eval on Comm test ==="
python train_ctc.py $COM $POSE --init_encoder "$DE/runs/vlp_joint/vlp_encoder_best.pt" --seed 42 \
  --batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 --out_dir "$DE/runs/ctc_joint"
python tools/eval_beam.py --ckpt "$DE/runs/ctc_joint/best.pt" --manifest "$DE/manifests/test.tsv" \
  --beam 5 --device cuda || true
cp -r "$DE/runs/ctc_joint" "$G/comm_dwpose_ctc_joint" 2>/dev/null || true
echo "JOINT_TRAIN_DONE"
