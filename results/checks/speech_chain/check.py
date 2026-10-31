"""Speech chain: latency by stage, voiced output by coverage (dissertation Section 'Speech Output').

Recomputes from the saved per-clip records of tools/speech_pipeline_bench.py (800 official test clips,
VLP-40 pose-only model, beam 5): stage medians/p95, shares, and BLEU-4 of the voiced subset at each
confidence quantile. Pose time is estimated from clip length (frames / 25.4 fps), as in the bench.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

recs = [json.loads(l) for l in open(HERE / "records.jsonl")]
meta = json.load(open(HERE / "meta.json"))
thr_json = json.load(open(HERE / "pose_throughput_subset.json"))
n = len(recs)
hyps = [r["hyp"] for r in recs]; refs = [r["ref"] for r in recs]
ces = np.array([r["ce"] for r in recs])
fr = np.array([r["frames"] for r in recs]); tr = np.array([r["translate_ms"] for r in recs])
sy = np.array([r["synth_ms"] for r in recs])

# references must be the official test references (when `make data` has written the manifest)
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
    assert len(m) == 800 and all(m[r["id"]] == r["ref"] for r in recs), "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")

# gate: the 100%-coverage score is the recorded VLP-40 beam-5 test score (10.27, ladder job 6806)
b_all = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
print(f"GATE all 800 voiced: BLEU-4 {b_all:.4f} (recorded 10.27 for the 40-epoch pre-trained model)")
if round(b_all, 2) != 10.27:
    sys.exit("GATE FAILED")

POSE_FPS = round(thr_json["pose_only_cpu"]["fps_from_mean"], 1)
print(f"\nSetting: {n} official test clips, beam {meta['beam']}, estimated pose time = frames / {POSE_FPS} fps")
print(f"pose-only CPU rate {POSE_FPS} frames/s (read from the saved job summary pose_throughput_subset.json)")
pose = fr / POSE_FPS * 1000.0
whole = pose + tr + sy
med_whole = np.median(whole)
print("\nLATENCY per utterance (ms)")
print(f"{'Stage':28s}{'Median':>9s}{'95th':>9s}{'Share':>8s}")
for name, v in [("Pose estimation (estimated)", pose), ("Translation (beam width 5)", tr),
                ("Speech synthesis", sy), ("Translation + synthesis", tr + sy), ("Whole chain", whole)]:
    print(f"{name:28s}{np.median(v):9.1f}{np.percentile(v, 95):9.1f}{np.median(v)/med_whole*100:7.1f}%")
print(f"\nmedian delay {med_whole/1000:.2f} s, 95th percentile {np.percentile(whole,95)/1000:.2f} s")
print(f"translation with this benchmark's own beam search: median {np.median(tr):.0f} ms")

fp = thr_json["full_pipeline_cpu"]; tc = thr_json["translator_cpu"]["beam5"]
fps_full = fp["fps_from_mean"]
med_frames = float(np.median(fr))
print(f"\nDetector on every frame: {fps_full:.1f} frames/s (read from the saved job summary); "
      f"median clip {med_frames:.0f} frames -> pose estimation alone {med_frames/round(fps_full,1):.1f} s")
print(f"Batched beam-5 search, single pose-only translator: mean {tc['mean_ms']:.0f} ms per clip "
      f"({tc['n']} test clips; read from the saved job summary)")

print("\nVOICED OUTPUT by coverage (threshold = quantile of the test-set mean per-token cross-entropy)")
print(f"{'Coverage':>9s}{'Voiced':>8s}{'Threshold':>11s}{'BLEU-4':>9s}")
for q in (1.00, 0.90, 0.75, 0.50, 0.25):
    thr = float(np.quantile(ces, q)) if q < 1.0 else float("inf")
    keep = [j for j in range(n) if ces[j] <= thr]
    b = corpus_bleu4([detokenize(hyps[j]) for j in keep], [detokenize(refs[j]) for j in keep])
    ts = "---" if q == 1.0 else f"{thr:.3f}"
    print(f"{q:8.0%}{len(keep):9d}{ts:>11s}{b:9.2f}")
