"""One-off (not run by make): per-clip data behind tools/translation_predictors.py.

Applies the tool's own clip_features() to the 800 official-test DWPose clips (results/test_clips_dwpose) and stores,
per clip, the 13 descriptors, the video id, the reference ("ref") and the beam-5 hypothesis of the three single models
(results/semantic_adequacy_grading/hyps_*.jsonl). No pose arrays are saved.
Run: cd src && python3 ../results/checks/input_side_predictors/prepare_data.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from translation_predictors import clip_features

rows = [l.rstrip("\n").split("\t") for l in open(ROOT / "comm_run/manifests/test.tsv")]
names = [Path(r[0]).name for r in rows]; refs = [r[1] for r in rows]
models = {m: [json.loads(l) for l in open(ROOT / f"results/semantic_adequacy_grading/hyps_{m}.jsonl")]
          for m in ["dwpose_ctc", "vlp", "rgbfusion"]}
with open(HERE / "per_clip.jsonl", "w") as f:
    for i, n in enumerate(names):
        p = ROOT / "results/test_clips_dwpose" / n
        d = clip_features(np.load(p)["KPS"]) if p.exists() else None
        if d is None:
            continue
        rec = {"id": n[:-4], "video": "_".join(n.split("_")[:2]), "feat": d, "ref": refs[i]}
        for m, recs in models.items():
            assert recs[i]["ref"].strip() == refs[i].strip()
            rec["hyp_" + m] = recs[i]["hyp_beam"]
        f.write(json.dumps(rec) + "\n")
