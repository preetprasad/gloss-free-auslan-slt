"""Run once (not by make): makes the small data files of this check from saved outputs. See SOURCES.md.

Usage: python prepare_data.py CONTROL_BEAM5_JSON SIGNJOEY_JSON
  (the --output_json files written by tools/eval_beam.py for the two local checkpoints, see SOURCES.md)
"""
import csv, json, shutil, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]   # results/
H = Path(__file__).resolve().parent
man = [l.rstrip("\n").split("\t") for l in open(R.parent / "comm_run/manifests/test.tsv") if "\t" in l]
ids = [Path(a).name[:-4] for a, _ in man]
# 1. first-listed AlphaPose control: greedy and beam-5 test decodes, per clip (references only under "ref")
c = json.load(open(sys.argv[1]))
assert len(c["greedy_hyps"]) == len(c["beam_hyps"]) == len(c["refs"]) == len(ids) == 800
assert c["beam"] == 5 and c["length_penalty"] == 0.6
with open(H / "control_decodes.jsonl", "w") as f:
    for i, g, b, r in zip(ids, c["greedy_hyps"], c["beam_hyps"], c["refs"]):
        f.write(json.dumps({"id": i, "hyp_greedy": g, "hyp_beam5": b, "ref": r}) + "\n")
# 2. SignJoey-compatible reconstruction: greedy test decodes
s = json.load(open(sys.argv[2]))
assert len(s["greedy_hyps"]) == len(s["refs"]) == len(ids)
with open(H / "signjoey_decodes.jsonl", "w") as f:
    for i, g, r in zip(ids, s["greedy_hyps"], s["refs"]):
        f.write(json.dumps({"id": i, "hyp_greedy": g, "ref": r}) + "\n")
# 3. DWPose per-clip mean hand confidence (hand column only)
with open(H / "dwpose_hand_conf.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip", "hand_conf"])
    for r in csv.DictReader(open(R / "dwpose_experiment/dwpose_conf.csv")):
        w.writerow([r["clip"], r["dwpose_hand_conf"]])
# 4. byte copies
shutil.copy(R / "probe_signerfix_2026-10-25/pred_ap_clips.csv", H / "pred_firstlisted_alphapose.csv")
shutil.copy(R / "probe_signerfix_2026-10-25/log_ap_clips.txt", H / "probe_log_firstlisted_alphapose.txt")
shutil.copy(R.parent / "comm_bbox_xywh_baseline/test_rescore.txt", H / "bbox_xywh_test_rescore.txt")
shutil.copy(R / "legacy_runs/comm_slt_1024_512_3x3_accum2.metrics.json", H / "capacity_accum2_metrics.json")
# 5. parameter count of the 25.95 M model, from its saved checkpoint
import torch
ck = torch.load(R.parent / "comm_slt_1024_512_3x3_accum2/out/best.pt", map_location="cpu", weights_only=False)
sd = ck.get("model", ck.get("model_state", ck))
# pos_enc is a fixed sinusoid buffer, not a parameter
json.dump({"n_params": sum(v.numel() for k, v in sd.items() if not k.startswith("pos_enc"))}, open(H / "capacity_params.json", "w"))
