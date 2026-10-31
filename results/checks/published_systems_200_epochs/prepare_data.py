"""Builds data/old/ from the saved outputs of the older single runs. Run once by hand, not by make.

  python3 prepare_data.py KAYA_DIR
KAYA_DIR holds d512.metrics.json and full.metrics.json, byte copies of metrics.json from the cluster run folders
comm_dwpose_d512 (job dwpose-d512) and comm_dwpose_full (job train_dwpose).
Everything else is read from this working tree. Files are copied byte for byte, except the checkpoint settings
(a small subset of the config stored inside each checkpoint; the checkpoints themselves are not copied).
"""
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "data" / "old"
OUT.mkdir(parents=True, exist_ok=True)
kaya = Path(sys.argv[1])

copies = {
    "dwpose_d512.metrics.json": kaya / "d512.metrics.json",
    "dwpose_full.metrics.json": kaya / "full.metrics.json",
    "exact_slt_run_summary.json": ROOT / "results/runs/communication_signjoey_direct_xywh_partial_2026-08-05.json",
    "exact_slt_train.log": ROOT / "comm_signjoey_direct_xywh_full/train.log",
    "exact_slt_progress.json": ROOT / "comm_signjoey_direct_xywh_full/out/progress.json",
    "signjoey_run_summary.json": ROOT / "results/runs/communication_signjoey_slt_partial_2026-08-05.json",
    "signjoey_train.log": ROOT / "comm_slt_signjoey_reference/train.log",
    "accum2_metrics.json": ROOT / "comm_slt_1024_512_3x3_accum2/out/metrics.json",
    "accum2_train.log": ROOT / "comm_slt_1024_512_3x3_accum2/train.log",
}
for name, src in copies.items():
    shutil.copyfile(src, OUT / name)

keys = ["model_arch", "lr", "epochs", "lr_schedule", "batch_size", "grad_accum_steps", "d_model",
        "num_encoder_layers", "num_decoder_layers", "num_keypoints", "coord_norm", "bbox_format"]
cfgs = {}
for name, ck in (("signjoey_reference", "comm_slt_signjoey_reference/out/best.pt"),
                 ("accum2", "comm_slt_1024_512_3x3_accum2/out/best.pt")):
    c = torch.load(ROOT / ck, map_location="cpu", weights_only=False)["cfg"]
    cfgs[name] = {"checkpoint": ck, **{k: c.get(k) for k in keys}}
(OUT / "checkpoint_cfgs.json").write_text(json.dumps(cfgs, indent=1))
print("wrote", sorted(p.name for p in OUT.iterdir()))
