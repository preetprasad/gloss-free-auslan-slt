"""Run once by hand (not by make): read the six S6 ensemble checkpoints and save a small text summary of what they say about themselves.

Output: members_config.json (training config fields stored inside each checkpoint + parameter counts).
No weights are copied. Needs torch (use the twin python). Checkpoints are local only (results/ckpts/, not published).
"""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAMES = ["comm_dwpose_ctc_vlplong80", "comm_dwpose_rgb_fusion", "comm_dwpose_vlp_ctc",
         "comm_dwpose_ctc_s202", "comm_dwpose_fus_s101", "comm_dwpose_fus_s202"]  # order of jobs/slurm/mbr_wer_test2.sbatch


def tail(p):
    # drop the cluster prefix, keep the part that identifies the run
    for marker in ("/dwpose/", "/pprasad/"):
        if marker in p:
            return p.split(marker, 1)[1]
    return p


out = []
for n in NAMES:
    ck = torch.load(ROOT / "results/ckpts" / f"{n}.pt", map_location="cpu", weights_only=False)
    cfg, sd = ck["cfg"], ck["model"]
    pe = {k: v.numel() for k, v in sd.items() if k.split(".")[-1] == "pe"}
    other_buffers = [k for k in sd if "running_" in k or "num_batches" in k]
    assert not other_buffers, other_buffers
    total = sum(v.numel() for v in sd.values())
    out.append({
        "member": n, "epochs": cfg["epochs"], "seed": cfg["seed"], "batch_size": cfg["batch_size"], "model_arch": cfg["model_arch"],
        "init_encoder_tail": tail(cfg.get("init_encoder", "")),
        "rgb_feat_dir_set": bool(cfg.get("rgb_feat_dir")),
        "n_tensors": len(sd), "state_dict_numel": total,
        "position_table_buffers": pe, "parameters_without_position_tables": total - sum(pe.values()),
    })
json.dump(out, open(HERE / "members_config.json", "w"), indent=1)
print("wrote members_config.json", [o["parameters_without_position_tables"] for o in out])
