"""Run once (not by make): writes the two small data files of this check from local saved outputs.
  pose_throughput_subset.json  <- results/pose_throughput_m3max_2026-09-15.json (tools/pose_throughput_bench.py)
  test_clip_frames.json        <- 'frames' of each record in results/speech_pipeline_bench.json (tools/speech_pipeline_bench.py)
Only numbers are kept; no subtitles, no local paths."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

t = json.load(open(ROOT / "results/pose_throughput_m3max_2026-09-15.json"))
sub = {"machine": {k: t["machine"][k] for k in ("machine", "cpu_count", "onnxruntime")},
       "video_fps": sorted({v["fps"] for v in t["videos"]}), "n_videos": len(t["videos"])}
for k in ("full_pipeline_cpu", "pose_only_cpu", "pose_only_coreml", "translator_cpu"):
    sub[k] = {a: b for a, b in t[k].items() if a != "agreement_with_cpu"} if k != "translator_cpu" else t[k]
json.dump(sub, open(HERE / "pose_throughput_subset.json", "w"), indent=1)

b = json.load(open(ROOT / "results/speech_pipeline_bench.json"))
json.dump({r["clip"]: r["frames"] for r in b["records"]}, open(HERE / "test_clip_frames.json", "w"))
