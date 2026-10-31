"""Run once (not by make): programmatic subsets of saved outputs. No reference sentences are copied."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
q = json.load(open(ROOT / "figures/qual_examples.json"))
with open("qual_subset.jsonl", "w") as f:
    for e in q:
        f.write(json.dumps({k: e[k] for k in ("i", "clip", "hyp", "ce")}) + "\n")
with open("stage_times.jsonl", "w") as f:
    for r in map(json.loads, open(ROOT / "results/checks/speech_chain/records.jsonl")):
        f.write(json.dumps({k: r[k] for k in ("id", "frames", "translate_ms", "synth_ms")}) + "\n")
t = json.load(open(ROOT / "results/checks/speech_chain/pose_throughput_subset.json"))
json.dump({"pose_only_cpu": t["pose_only_cpu"]}, open("pose_rate.json", "w"))
