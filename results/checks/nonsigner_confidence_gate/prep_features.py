"""One-off preparation (NOT run by make): derive the per-clip wrist speed used by the 'hardly moves' stratum from the pose arrays, which are not shipped.
Same function as src/tools/nonsigner_gate.py::wrist_speed (median wrist speed in shoulder widths per frame, joints 9 and 10, confidence >= 0.3).
Output: wrist_speed.csv  (id, wrist_speed) for the 800 validation signer clips (production pose) and the 1,678 non-signer crops."""
import json, sys
from pathlib import Path
import numpy as np
RES = Path(sys.argv[1])          # private tree results/ directory
CONF = 0.3
def wrist_speed(kps):
    sp = []
    for t in range(1, len(kps)):
        a, b = kps[t - 1], kps[t]
        if min(a[5, 2], a[6, 2], b[5, 2], b[6, 2]) < CONF:
            continue
        w = np.linalg.norm(b[5, :2] - b[6, :2])
        for j in (9, 10):
            if a[j, 2] >= CONF and b[j, 2] >= CONF and w > 1e-3:
                sp.append(np.linalg.norm(b[j, :2] - a[j, :2]) / w)
    return float(np.median(sp)) if sp else float("nan")
D = json.load(open(RES / "nonsigner" / "decodes.json"))
rows = []
for c in D["sig_val"]:
    rows.append((c, wrist_speed(np.load(RES / "val_clips_dwpose" / f"{c}.npz")["KPS"])))
for c in D["neg"]:
    rows.append((c, wrist_speed(np.load(RES / "nonsigner" / "npz" / f"{c}.npz")["KPS"])))
with open(Path(__file__).with_name("wrist_speed.csv"), "w") as f:
    f.write("id,wrist_speed\n")
    for c, v in rows:
        f.write(f"{c},{v!r}\n")
print(len(rows))
