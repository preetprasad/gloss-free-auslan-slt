"""One-off: build the compact saved inputs of check.py from the private working tree (not run by `make reproduce`).

usage: python prepare_data.py CROSS_POSE_DIR   (a folder with <arm>_s<seed>/beam5_<split>_best_on_<other>.json copied from the cluster)
Keeps only hypotheses, the recorded BLEU-4 and the run name: no references, no checkpoints, no pose arrays.
"""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "data"
KEY = "tokenize=13a;detok=True"
CROSS = Path(sys.argv[1])
R = ROOT / "results"


def names(p):
    return [l.split("\t")[0].rsplit("/", 1)[1][:-4] for l in open(p, encoding="utf-8") if "\t" in l]


def slim(path, run, pose):
    j = json.load(open(path))
    assert j["beam"] == 5 and len(j["beam_hyps"]) == 800
    return {"run": run, "decode_pose": pose, "bleu": j["beam_result"]["metrics"][KEY], "hyps": j["beam_hyps"]}


def dump(name, rows):
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, os.path.getsize(OUT / name))


order = {s: names(R / f"tracker_v2_translation_test/manifests/c0_{s}.tsv") for s in ("val", "test")}
assert all(order[s] == names(ROOT / f"comm_run/manifests/{s}.tsv") for s in order)
json.dump(order, open(OUT / "clip_order.json", "w"))

for split in ("val", "test"):
    for arm in ("c0", "v2"):
        rows = []
        for s in (303, 404, 505, 606, 707):
            rows.append(slim(R / f"wave12i_138391/{arm}_s{s}/beam5_{split}_best.json", f"{arm}_s{s}", arm))
            o = "v2" if arm == "c0" else "c0"
            rows.append(slim(CROSS / f"{arm}_s{s}/beam5_{split}_best_on_{o}.json", f"{arm}_s{s}", o))
        dump(f"w12i_{split}_{arm}.jsonl", rows)
    dump(f"w9_{split}.jsonl", [slim(R / f"tracker_v2_translation_test/runs/{a}_s{s}/beam5_{split}_best.json", f"{a}_s{s}", a)
                              for a in ("c0", "v2") for s in (42, 101, 202)])
    d = R / "switch_video_89_193/w3_val"
    fn = "{a}_s{s}.json" if split == "val" else "{a}_s{s}_test.json"
    dump(f"w3_{split}.jsonl", [slim(d / fn.format(a=a, s=s), f"{a}_s{s}", a) for a in ("w3C", "w3T") for s in (42, 101, 202, 303, 404)])

# tracker prevalence (job 6591): per-clip numerators + the job's own totals
prev = json.load(open(R / "switch_video_89_193/w3_val/tracking_prevalence.json"))
with open(OUT / "tracking_switched_clips.csv", "w") as f:
    f.write("clip,split,frames,switched\n")
    for c, v in sorted(prev["per_clip"].items()):
        f.write(f"{c},{v['split']},{v['frames']},{v['switched']}\n")
json.dump(prev["agg"], open(OUT / "tracking_prevalence_agg.json", "w"), indent=1)

# blind tracker audit (160 panels, two raters, hidden key already joined)
lab = json.load(open(R / "switch_video_89_193/audit/labels_scored.json"))
json.dump({k: {x: v[x] for x in ("stratum", "r1", "r2", "m1", "m2")} for k, v in lab.items()}, open(OUT / "tracker_audit_labels.json", "w"))

# hand speed: per-clip mean left-hand displacement per frame / shoulder width (tools/hand_visibility_cv.py::hand_stats)
sys.path.insert(0, str(ROOT / "src/tools"))
from hand_visibility_cv import hand_stats
tr = {Path(p).stem: p for p in glob.glob(str(R / "nmm_probe/tracked_train/*.npz"))}
fd = {Path(p).stem: p for p in glob.glob(str(R / "dwpose_experiment/dwpose_clips/*.npz"))}
with open(OUT / "hand_speed_per_clip.csv", "w") as f:
    f.write("clip,first_detection_speed,tracked_speed\n")
    for c in sorted(set(tr) & set(fd)):
        a, b = hand_stats(np.load(fd[c])["KPS"]), hand_stats(np.load(tr[c])["KPS"])
        if a and b:
            f.write(f"{c},{a['hand_speed']!r},{b['hand_speed']!r}\n")

# pose-jump census inputs: per-clip valid-shoulder share, jumps and valid transitions (definition of pose/tracker_v2/research/alphapose_identity_check.py::jumps)
def jumps(kps, thr=0.3, k=0.4):
    s = kps[:, [5, 6]]; ok = (s[:, :, 2] > thr).all(1)
    ctr = s[:, :, :2].mean(1); w = np.linalg.norm(s[:, 0, :2] - s[:, 1, :2], axis=1)
    if ok.sum() < 2:
        return 0, 0, float(ok.mean())
    ref = np.median(w[ok]); pair = ok[1:] & ok[:-1]
    big = (np.linalg.norm(ctr[1:] - ctr[:-1], axis=1) > k * ref) | (np.abs(w[1:] - w[:-1]) > k * ref)
    return int((big & pair).sum()), int(pair.sum()), float(ok.mean())

files = {}
for d in ("dwpose_experiment/dwpose_clips", "test_clips_dwpose", "val_clips_dwpose"):
    for p in glob.glob(str(R / d / "*.npz")):
        files.setdefault(Path(p).stem, p)
with open(OUT / "pose_jump_census.csv", "w") as f:
    f.write("clip,shoulder_valid_share,jumps,transitions\n")
    for c in sorted(files):
        j, t, sh = jumps(np.load(files[c])["KPS"])
        f.write(f"{c},{sh!r},{j},{t}\n")
