"""Byte-copy the saved outputs this check needs that are not already published (run once by hand, not by make)."""
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
R = ROOT / "results"
files = [f"dumps_Faug_val/hyps_LP={lp}.jsonl" for lp in "0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0".split()]
files += ["dumps_Faug_test/hyps_LP=0.8.jsonl", "ensemble_six_duration_two_fusion_decodes/hyps_LP=0.7.jsonl", "log_Faug_test.log", "video_disjoint_cross_modal_ensemble/decode/VD5_test.log"]
files += ["video_disjoint_cross_modal_ensemble/decode/VD5_test/hyps_LP=0.9.jsonl", "video_disjoint_cross_modal_ensemble/prior/seedens_vdB/s101_test/hyps_LP=0.9.jsonl",
          "video_disjoint_cross_modal_ensemble/prior/seedens_vdB/ens5_test/hyps_LP=0.8.jsonl", "t2_vdens_10674/dumps_V6_test/hyps_LP=1.0.jsonl",
          "t2_vdens_10674/dumps_V3A_test/hyps_LP=0.9.jsonl", "t2_vdens_10674/dumps_V3B_test/hyps_LP=0.8.jsonl"]
for f in files:
    d = HERE / "data" / f.replace("/", "__")
    d.parent.mkdir(exist_ok=True)
    shutil.copyfile(R / f, d)

# --- second part: settings of the ensemble member checkpoints (programmatic subset of each checkpoint's saved cfg) and the VD read-out
import json, hashlib, torch
shutil.copyfile(R / "video_disjoint_cross_modal_ensemble/g1_readout.txt", HERE / "data" / "vd_xmodal_31205__g1_readout.txt")
members = {}
for tag, rel in [("pa_aug_s42", "pose_augmentation_decodes/ckpts/pa_aug_s42"), ("pa_aug_s101", "pose_augmentation_decodes/ckpts/pa_aug_s101"),
                 ("pa_aug_s202", "pose_augmentation_decodes/ckpts/pa_aug_s202"), ("rgb_fusion", "ckpts_fus/rgb_fusion"), ("fus_s101", "ckpts_fus/fus_s101"),
                 ("r111_s42", "ckpts_r111/s42"), ("r111_s101", "ckpts_r111/s101"), ("r111_s202", "ckpts_r111/s202")]:
    p = R / rel / "best.pt"
    c = torch.load(p, map_location="cpu", weights_only=False)["cfg"]
    members[tag] = {"train_manifest_name": Path(c["train_manifest"]).name, "pose_aug": bool(c.get("pose_aug", False)),
                    "pose_aug_field_present": "pose_aug" in c, "uses_rgb_features": bool(c.get("rgb_feat_dir")),
                    **{k: c[k] for k in ("num_keypoints", "coord_norm", "epochs", "lr", "lr_schedule", "batch_size", "seed")},
                    "checkpoint_md5": hashlib.md5(p.read_bytes()).hexdigest()}
json.dump(members, open(HERE / "data" / "member_cfgs.json", "w"), indent=1)
