"""One-off (not run by make): subset the saved training checkpoints into cfg_members.json.
Reads results/ckpts/*.pt (torch needed; the checkpoints are not published) and keeps only the saved
training configuration, the tensor shapes that show the architecture, and the parameter count."""
import json, re, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
CK = HERE.parents[1] / "ckpts"
KEEP = ["num_keypoints", "keypoint_subset", "use_confidence", "coord_norm", "max_src_frames", "max_tgt_tokens",
        "d_model", "nhead", "dim_feedforward", "num_encoder_layers", "num_decoder_layers", "dropout",
        "embedding_dropout", "label_smoothing", "lr", "weight_decay", "grad_clip", "lr_schedule",
        "warmup_steps", "batch_size", "epochs", "seed", "rgb_feature_dim", "num_beams", "length_penalty"]
out = {}
for f in sorted(CK.glob("*.pt")):
    c = torch.load(f, map_location="cpu", weights_only=False)
    m, cfg = c["model"], c["cfg"]
    enc = {int(x.group(1)) for k in m if (x := re.search(r"transformer\.encoder\.layers\.(\d+)\.", k))}
    dec = {int(x.group(1)) for k in m if (x := re.search(r"transformer\.decoder\.layers\.(\d+)\.", k))}
    ip = [tuple(v.shape) for k, v in m.items() if k.endswith("input_proj.0.weight")]
    out[f.stem] = {"cfg": {k: cfg[k] for k in KEEP if k in cfg},
                   "rgb_stream": any(k.startswith("rgb_") for k in m),
                   "encoder_layers_in_weights": len(enc), "decoder_layers_in_weights": len(dec),
                   "input_proj_shape": ip[0] if ip else None, "vocab": len(c["vocab"])}
json.dump(out, open(HERE / "cfg_members.json", "w"), indent=1)
print(len(out), "checkpoints written")

# decode settings and the recorded score of the gate run, taken from the saved job output (no hypotheses, no references)
src = HERE.parents[1] / "pose_augmentation_decodes" / "pa_aug_s42" / "beam5_test.json"
b = json.load(open(src))
json.dump({"beam": b["beam"], "length_penalty": b["length_penalty"], "n": b["beam_result"]["n"],
           "recorded_bleu4_13a": b["beam_result"]["metrics"]["tokenize=13a;detok=True"]},
          open(HERE / "run_meta.json", "w"), indent=1)
