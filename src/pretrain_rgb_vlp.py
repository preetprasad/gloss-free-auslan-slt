"""CLI for RGB Visual-Language Pretraining (see slt/rgb_vlp.py).

  python pretrain_rgb_vlp.py --train_manifest ... --rgb_feat_dir <dir> \
      --num_keypoints 133 --keypoint_subset all --coord_norm frame \
      --batch_size 64 --epochs 40 --lr 5e-4 --lr_schedule cosine --device cuda --out_dir runs/rgb_vlp
"""
from __future__ import annotations
import argparse
from config import Config
from slt.rgb_vlp import train_rgb_vlp


def main():
    cfg = Config()
    p = argparse.ArgumentParser(description="RGB-VLP (RGB<->text contrastive)")
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--vlp_proj", type=int, default=256)
    p.add_argument("--vlp_temp", type=float, default=0.07)
    a = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in ("vlp_proj", "vlp_temp")})
    if not cfg.rgb_feat_dir:
        raise SystemExit("RGB-VLP requires --rgb_feat_dir")
    train_rgb_vlp(cfg, a.vlp_proj, a.vlp_temp)


if __name__ == "__main__":
    main()
