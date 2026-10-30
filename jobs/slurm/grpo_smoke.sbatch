#!/bin/bash -l
# GRPO smoke: synthetic pose -> tiny supervised ckpt -> GRPO loop runs (sample/reward/advantage/
# KL/update/eval) on CPU. Validates the RL machinery before the GPU run.
#SBATCH --job-name=grpo-smoke
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q sacrebleu 2>/dev/null || true
cd /group/pmc099/pprasad/code/phase2_baseline
TMP=$(mktemp -d)
python - "$TMP" <<'PY'
import os,sys,numpy as np,torch
tmp=sys.argv[1]; os.makedirs(f"{tmp}/pose",exist_ok=True)
rng=np.random.default_rng(0); words=["hello","how","are","you","good","yes","no","thanks","today","fine"]
def line(cid):
    T=int(rng.integers(12,30)); k=rng.normal(0.5,0.2,(T,133,3)).astype(np.float32); k[...,2]=rng.uniform(.3,.9,(T,133))
    np.savez(f"{tmp}/pose/{cid}.npz",KPS=k,BBOX=np.tile(np.array([0,0,720,720],np.float32),(T,1)))
    return f"{tmp}/pose/{cid}.npz\t"+" ".join(rng.choice(words,size=int(rng.integers(3,6))))
rows=[line(f"c{i}") for i in range(24)]
for nm,sl in [("train",slice(0,16)),("val",slice(16,20)),("test",slice(20,24))]:
    open(f"{tmp}/{nm}.tsv","w").write("\n".join(rows[sl])+"\n")
# tiny supervised ckpt to warm-start from
sys.path.insert(0,".")
from config import Config; from slt.data import Vocab, read_manifest; from slt.reference_model import build_model
cfg=Config(num_keypoints=133,keypoint_subset="all",coord_norm="frame",d_model=128,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=256)
vocab=Vocab.build([t for _,t in read_manifest(f"{tmp}/train.tsv")])
m=build_model(cfg.feature_dim(),len(vocab),cfg)
torch.save({"model":m.state_dict(),"vocab":vocab.itos,"cfg":cfg.to_dict()},f"{tmp}/init.pt")
print("READY",tmp)
PY
python train_grpo.py --init_ckpt "$TMP/init.pt" \
  --train_manifest "$TMP/train.tsv" --val_manifest "$TMP/val.tsv" --test_manifest "$TMP/test.tsv" \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --d_model 128 --num_encoder_layers 2 --num_decoder_layers 2 --dim_feedforward 256 \
  --grpo_group 4 --grpo_beta 0.04 --lr 1e-5 --epochs 2 --batch_size 4 --warmup_steps 2 --device cpu \
  --out_dir "$TMP/grpo"
echo "GRPO_SMOKE_DONE"; rm -rf "$TMP"
