#!/bin/bash -l
# RGB-VLP smoke: synthetic rgb+text -> RGB-VLP pretrain -> RGB fusion warm-started from it.
# Validates the whole RGB-VLP chain on CPU before the GPU run.
#SBATCH --job-name=rgbvlp-smoke
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
cd /group/pmc099/pprasad/code/phase2_baseline
TMP=$(mktemp -d)
python - "$TMP" <<'PY'
import os, sys, numpy as np
tmp=sys.argv[1]; os.makedirs(f"{tmp}/pose",exist_ok=True); os.makedirs(f"{tmp}/rgb",exist_ok=True)
rng=np.random.default_rng(0); words=["hello","how","are","you","good","yes","no","thanks"]
def line(cid):
    T=int(rng.integers(12,30)); kps=rng.normal(0.5,0.2,(T,133,3)).astype(np.float32); kps[...,2]=rng.uniform(0.3,0.9,(T,133))
    np.savez(f"{tmp}/pose/{cid}.npz",KPS=kps,BBOX=np.tile(np.array([0,0,720,720],np.float32),(T,1)))
    Tr=int(rng.integers(8,24)); np.savez(f"{tmp}/rgb/rgb_{cid}.npz",FEATURES=rng.normal(0,1,(Tr,1152)).astype(np.float32))
    return f"{tmp}/pose/{cid}.npz\t"+" ".join(rng.choice(words,size=int(rng.integers(3,6))))
rows=[line(f"clip{i}") for i in range(24)]
for name,sl in [("train",slice(0,16)),("val",slice(16,20)),("test",slice(20,24))]:
    open(f"{tmp}/{name}.tsv","w").write("\n".join(rows[sl])+"\n")
print("READY",tmp)
PY
echo "=== RGB-VLP pretrain ==="
python pretrain_rgb_vlp.py --train_manifest "$TMP/train.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$TMP/rgb" --rgb_feature_dim 1152 \
  --d_model 128 --num_encoder_layers 2 --dim_feedforward 256 \
  --batch_size 4 --epochs 2 --lr 5e-4 --lr_schedule cosine --warmup_steps 3 \
  --device cpu --out_dir "$TMP/rgbvlp"
echo "=== RGB fusion warm-started from RGB-VLP ==="
python train_ctc.py --train_manifest "$TMP/train.tsv" --val_manifest "$TMP/val.tsv" --test_manifest "$TMP/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir "$TMP/rgb" --rgb_feature_dim 1152 --rgb_vlp_init "$TMP/rgbvlp/rgb_vlp_best.pt" \
  --d_model 128 --num_encoder_layers 2 --num_decoder_layers 2 --dim_feedforward 256 \
  --batch_size 4 --epochs 1 --lr 3e-4 --lr_schedule cosine --warmup_steps 3 --device cpu --out_dir "$TMP/fuse"
echo "RGBVLP_SMOKE_DONE"; rm -rf "$TMP"
