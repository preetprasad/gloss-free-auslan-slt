"""Ensemble / translator / pose-front-end cost and latency (Results 'latency', Discussion, appendix latency table, Method set-up).
Parses the saved timing outputs: ensemble_latency_log.txt (tools/ensemble_latency.py, M3 Max CPU, 800 validation clips),
pose_throughput_subset.json (tools/pose_throughput_bench.py), test_clip_frames.json (frames per official test clip),
and reads the GPU request of every training job script in the repository."""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
R = lambda x, d=0: str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP))

# ---- ensemble decoding cost -------------------------------------------------------------------------------------
rows = {}
for l in open(HERE / "ensemble_latency_log.txt"):
    m = re.match(r"beam\s+(\d+) \| members (\d+) \| BLEU-4\s+([\d.]+) WER ([\d.]+) \| per clip median\s+([\d.]+) ms\s+p95\s+([\d.]+) ms\s+mean\s+([\d.]+) ms", l)
    if m:
        rows[(int(m[2]), int(m[1]))] = dict(bleu=float(m[3]), med=float(m[5]), p95=float(m[6]))
assert len(rows) == 6
print(f"GATE six members, beam 10: validation BLEU-4 {rows[(6,10)]['bleu']:.3f} (recorded 12.470, the headline validation score)")
if round(rows[(6, 10)]["bleu"], 3) != 12.470:
    sys.exit("GATE FAILED")
print("read from the saved job output ensemble_latency_log.txt (M3 Max CPU, 12 threads, batch 1, 800 validation clips)")

ids = {(1, 5): "c16-050", (3, 5): "c16-051", (6, 5): "c16-052", (1, 10): "c16-053", (3, 10): "c16-054", (6, 10): "c16-055"}
print(f"{'members':>8}{'beam':>6}{'BLEU-4':>8}{'median ms':>11}{'p95 ms':>8}")
for (k, b), i in ids.items():
    r = rows[(k, b)]
    print(f"{k:8d}{b:6d}{R(r['bleu'],2):>8}{R(r['med']):>11}{R(r['p95']):>8}  [{i}]")
print("members 1, 3 and 6 on 800 validation clips at batch size 1, 12 threads (log header: clips 800, torch threads 12)  [c16-049]")
print(f"6 members (the six-member ensemble), beam 10: {R(rows[(6,10)]['bleu'],2)} validation BLEU-4 at a median {R(rows[(6,10)]['med'])} ms per clip  [c13-024 c13-026]")
print(f"3 members (three), beam 10: {R(rows[(3,10)]['bleu'],2)} validation BLEU-4 at a median {R(rows[(3,10)]['med'])} ms per clip  [c18-056 c18-057]")

# ---- laptop pose front end --------------------------------------------------------------------------------------
t = json.load(open(HERE / "pose_throughput_subset.json"))
full, pose, cml, tr = t["full_pipeline_cpu"], t["pose_only_cpu"], t["pose_only_coreml"], t["translator_cpu"]["beam5"]
# recomputed from the per-frame mean latency (the bench's own definition: 1000 / mean ms)
fps = lambda d: 1000.0 / d["mean_ms"]
for d in (full, pose, cml):
    assert abs(fps(d) - d["fps_from_mean"]) < 1e-9
print("\nread from the saved job output pose_throughput_subset.json (M3 Max CPU; frames per second recomputed as 1000 / mean ms per frame)")
f5, f25, fcm = R(fps(full), 1), R(fps(pose), 1), R(fps(cml), 1)
print(f"detector on every frame: {f5} frames per second ({full['n']} frames)  [c02-027 c13-021 c13-025 c14-086]")
print(f"detector once per clip, pose model alone: {f25} frames per second  [c13-022 c14-086]")
print(f"pose model alone on the Core ML provider: {fcm} frames per second  [c18-055]")
print(f"single pose-only translator, beam width 5: mean {R(tr['mean_ms'])} ms per clip ({tr['n']} test clips)  [c02-028 c13-023]")
assert len(t["video_fps"]) == 1
print(f"source video: {R(t['video_fps'][0])} frames per second (container rate of all {t['n_videos']} timed clips)  [c14-086]")

fr = np.array(list(json.load(open(HERE / "test_clip_frames.json")).values()))
med = float(np.median(fr))
print(f"\nmedian official test clip: {R(med)} frames ({len(fr)} clips, frames of each clip in the saved speech-chain benchmark)")
print(f"pose estimation alone for the median clip at {f5} frames/s: {R(med / float(f5), 1)} s  [c14-089]")

# ---- GPU request of the training jobs --------------------------------------------------------------------------------
excl = re.compile(r"^_?w(12n|12nb|12nc|12c|12l|1[3-9]|2[0-9])")
cnt = {}
for p in sorted((ROOT / "jobs/slurm").glob("*.sbatch")):
    if excl.match(p.name):
        continue
    s = p.read_text()
    if "train_ctc.py" not in s:
        continue
    g = re.findall(r"^#SBATCH\s+--gres=(\S+)", s, re.M)
    cnt[g[0] if g else "none (CPU, summary or smoke script)"] = cnt.get(g[0] if g else "none (CPU, summary or smoke script)", 0) + 1
n = sum(cnt.values())
print(f"\nread from the job scripts: {n} scripts that call train_ctc.py; their `#SBATCH --gres` request:")
for k, v in sorted(cnt.items()):
    print(f"  {v:3d} x {k}")
gpu = {k: v for k, v in cnt.items() if not k.startswith("none")}
one = sum(v for k, v in gpu.items() if k.endswith(":1") or k == "gpu:1")
v100 = sum(v for k, v in gpu.items() if "v100" in k)
assert one == sum(gpu.values())
print(f"every one of the {one} scripts that request a GPU asks for exactly one GPU; {v100} name the V100 type, "
      f"{one - v100} ask for any single GPU  [c07-069]")
assert not any(k.endswith(":2") for k in gpu)
