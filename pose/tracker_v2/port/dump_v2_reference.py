"""Gate B reference: tracker_v2.choose for TUNED and DEFAULT on the 60 ground-truth clips (no labels are read).
Usage: dump_v2_reference.py ALLDET_DIR OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
import tracker_v2 as v2

src, out = Path(sys.argv[1]), Path(sys.argv[2])
clips = [c for part in json.load(open(Path(__file__).parents[1] / "gt_clips.json")).values() for c, _ in part]
ref = {"numpy": np.__version__, "clips": {}}
for c in clips:
    d = dict(np.load(src / f"{c}.npz"))
    ref["clips"][c] = {w: v2.choose(d, getattr(v2, w))[1].tolist() for w in ("TUNED", "DEFAULT")}
json.dump(ref, open(out, "w"))
print(f"{len(clips)} clips, {sum(len(v['TUNED']) for v in ref['clips'].values())} frames -> {out}")
