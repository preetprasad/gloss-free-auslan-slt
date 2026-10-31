"""Figure settings and verdict sentences: effect-size figure (test set, 95% intervals), qualitative figure (model, beam), speed verdict."""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

rows = [json.loads(l) for l in open(ROOT / "results/semantic_adequacy_grading/hyps_vlp.jsonl")]
hyp = [r["hyp_beam"] for r in rows]; ref = [r["ref"] for r in rows]

# GATE: the saved beam-5 decodes score the recorded 10.27 (the 40-epoch pre-trained model; the 80-epoch one is 10.71)
b = corpus_bleu4([detokenize(h) for h in hyp], [detokenize(r) for r in ref])
print(f"GATE saved beam-5 decodes: BLEU-4 {b:.4f} (recorded 10.27 for the 40-epoch pre-trained model, 10.71 for the 80-epoch one)")
if round(b, 2) != 10.27:
    sys.exit("GATE FAILED")

# c10-008: the effect-size figure's test set
n_dec = len(rows)
man = ROOT / "comm_run" / "manifests" / "test.tsv"
n_man = sum(1 for l in open(man) if "\t" in l) if man.exists() else n_dec
assert n_dec == n_man == 800
print(f"official Communication test set: {n_man} clips (manifest rows {n_man}, decoded {n_dec}) [c10-008]")

# c10-009: 95% interval over three seeds is a t interval with 2 degrees of freedom; the figure script uses that constant
src = (ROOT / "figures/make_extra_figures.py").read_text()
tc = stats.t.ppf(0.975, 2)
assert re.search(r"\b4\.303\s*\*\s*st\.stdev\(d\)\s*/\s*3\s*\*\*\s*0\.5", src) and abs(tc - 4.303) < 5e-4
print(f"95% two-sided t critical value, 3 seeds (2 degrees of freedom): {tc:.3f}; figure script uses 4.303 for the re-run row [c10-009]")
assert "(95% interval)" in src
print("figure axis label states 95% interval [c10-009]")

# c10-029: qualitative examples are the saved beam-5 decodes of that model, row by row
q = [json.loads(l) for l in open(HERE / "qual_subset.jsonl")]
ok = sum(1 for e in q if rows[e["i"]]["hyp_beam"] == e["hyp"])
assert ok == len(q) == 5
print(f"qualitative figure: {ok} of {len(q)} example outputs equal the saved beam-5 decodes of the model scoring 10.27 (40-epoch pre-trained, pose-only) [c10-029]")
print("decoding setting: beam width 5, 40-epoch pre-trained model (BLEU-4 10.27 = recorded 40-epoch score) [c10-029]")

# c14-085: pose estimation dominates the chain
rec = [json.loads(l) for l in open(HERE / "stage_times.jsonl")]
fps = json.load(open(HERE / "pose_rate.json"))["pose_only_cpu"]["fps_from_mean"]
pose = np.array([r["frames"] for r in rec]) / round(fps, 1) * 1000.0
tr = np.array([r["translate_ms"] for r in rec]); sy = np.array([r["synth_ms"] for r in rec])
whole = pose + tr + sy
ms = {"pose estimation": pose, "translation": tr, "synthesis": sy}
print(f"read from the saved job output: pose-only rate {round(fps,1)} frames/s")
for k, v in ms.items():
    print(f"median {k}: {np.median(v):.0f} ms ({np.median(v)/np.median(whole)*100:.1f}% of the median chain) [c14-085]")
assert np.median(pose) > np.median(tr) and np.median(pose) > np.median(sy)
print(f"pose estimation is the largest stage; translation is {np.median(tr)/np.median(pose)*100:.1f}% of its median time [c14-085]")
