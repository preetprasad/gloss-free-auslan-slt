"""Gate B input: write <clip>.box.f32 (DET_BOX, float32 (T,N,4) xyxy, C order) for the 60 ground-truth clips next to
links to the existing flat json/kp/sc files. Usage: export_boxes.py ALLDET_DIR FLAT_DIR OUT_DIR"""
import json, os, sys
from pathlib import Path
import numpy as np

src, flat, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
clips = [c for part in json.load(open(Path(__file__).parents[1] / "gt_clips.json")).values() for c, _ in part]
for c in clips:
    d = np.load(src / f"{c}.npz")
    meta = json.load(open(flat / f"{c}.json"))
    assert d["DET_BOX"].dtype == np.float32 and d["DET_BOX"].shape[:2] == (meta["T"], meta["N"]), c
    assert (d["NDET"] == meta["NDET"]).all(), c
    for ext in ("json", "kp.f32", "sc.f32"):
        # the flat kp/sc must be the npz arrays byte for byte
        if ext != "json":
            key = "KP_ALL" if ext == "kp.f32" else "SC_ALL"
            assert (flat / f"{c}.{ext}").read_bytes() == np.ascontiguousarray(d[key]).tobytes(), (c, ext)
        dst = out / f"{c}.{ext}"
        if not dst.exists():
            os.symlink(flat / f"{c}.{ext}", dst)
    np.ascontiguousarray(d["DET_BOX"]).tofile(out / f"{c}.box.f32")
print(f"{len(clips)} clips -> {out}")
