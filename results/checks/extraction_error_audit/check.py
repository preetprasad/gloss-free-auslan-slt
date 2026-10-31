"""Extraction-error audit: layout of the released keypoint files, the signer entry, registration, extractor input size,
bounding-box format.

Recomputes from pt_clip_table.csv (one row per clip of the released SP_Annotation_V1 pose files: entries listed, array
shapes, and whether the project's old extraction equals the first-listed or the 'Signer' entry), dwpose_k_counts.json
(keypoints per person in the stored DWPose files), the registration table of one clip, extractor_mode_table.txt (the
extraction library's mode table, copied from its source) and bbox_audit_sample200.csv (per-clip counts of the
bounding-box audit). Tables made by prepare_data.py.
"""
import csv, json, re, runpy, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
rows = list(csv.DictReader(open(HERE / "pt_clip_table.csv")))
n = len(rows)


def gate(ok, msg):
    print(f"GATE {msg}: {'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit("GATE FAILED")


# GATE: the Communication subset has 14,041 clips (official split 12,441 + 800 + 800; recorded in the dataset tables)
gate(n == 14041, f"clips in the released pose files {n:,} (expected 14,041)")
man = [ROOT / "comm_run" / "manifests" / f"{s}.tsv" for s in ("train", "val", "test")]
if all(m.exists() for m in man):
    ids = {Path(l.split("\t")[0]).name[:-4] for m in man for l in open(m) if "\t" in l}
    gate(ids == {r["clip"] for r in rows}, "clip ids of the manifests (make data) equal the pose-file ids")

# ---- layout
shp = Counter((int(r["K"]), int(r["C"])) for r in rows)
all136 = sum(int(r["all_K136"]) for r in rows)
print(f"\nAlphaPose HALPE layout: every entry of every clip stores keypoints of shape (T, 136, 3): {all136:,} of {n:,} clips "
      f"(shape counts {dict(shp)})  [c05-007 c06-018 c08-012 c15-003]")
K = int(rows[0]["K"])
print(f"keypoint count per frame in the layout: {K} (HALPE-136)  [c05-005 c05-007 c06-018 c08-012 c15-003]")
print(f"every clip has one array per entry with {int(rows[0]['bbox_cols'])} box columns; keypoint columns x, y, confidence = {int(rows[0]['C'])}  [c05-007]")

# Group sizes: NOT measurable from the keypoint arrays (they carry no group labels). Read from the project's HALPE-136 definition
# (docstring of the re-indexing tool, whose own asserts are executed here) and checked against the measured totals.
conv = ROOT / "src" / "tools" / "halpe_to_cwb133.py"
doc = runpy.run_path(str(conv))["__doc__"]                                   # runs the file's assertions (HALPE -> 133 re-indexing)
spans = re.findall(r"(\d+)-(\d+) ([a-z/ ]+?)(?: \(|,|\.)", doc.split("HALPE-136:")[1].split("COCO-WB-133:")[0])
grp = {}
for a, b, name in spans:
    key = "face" if name.strip() == "face" else "hand" if "hand" in name else "right_hand" if name.strip() == "right" else "body"
    grp[key] = grp.get(key, 0) + int(b) - int(a) + 1
covered = sorted(i for a, b, _ in spans for i in range(int(a), int(b) + 1))
assert covered == list(range(K)), "the definition does not partition the measured keypoint layout"
grp["hand"] = grp.get("hand", 0) + grp.pop("right_hand", 0)
print(f"HALPE-136 groups, from the definition in the re-indexing tool (not measured; the ranges partition all {K} measured keypoint indices): "
      f"{grp['body']} body (incl. head/neck/hip and feet), {grp['hand']} hand ({grp['hand']//2} per hand), {grp['face']} face; "
      f"total {sum(grp.values())}  [c05-005 c05-007]")
dwk = json.load(open(HERE / "dwpose_k_counts.json"))
kk = int(next(iter(dwk["keypoints_per_person_counts"])))
assert len(dwk["keypoints_per_person_counts"]) == 1
print(f"DWPose (COCO-WholeBody) stored arrays: {dwk['keypoints_per_person_counts'][str(kk)]:,} of {dwk['files']:,} files hold {kk} keypoints per person "
      f"and frame (body, hands, face)  [c02-002]")

# ---- signer entry
multi = [r for r in rows if int(r["n_keys"]) > 1]
single = [r for r in rows if int(r["n_keys"]) == 1]
eqf = sum(int(r["old_clip_equals_first_listed"]) for r in multi); eqs = sum(int(r["old_clip_equals_signer"]) for r in multi)
# GATE: the headline of the signer-selection error (Method, "Signer selection"): recorded 12,411 multi-person clips, 88.4%, 0 signer-equal
gate(len(multi) == 12411 and round(100 * len(multi) / n, 1) == 88.4 and eqs == 0 and eqf == len(multi),
     "multi-person clips recorded as 12,411 (88.4%), old extraction = first-listed entry in all and the signer entry in none")
print(f"\nclips whose file lists several people: {len(multi):,} of {n:,} ({100*len(multi)/n:.1f}%)  [c06-019]")
print(f"  of these, the entry marked 'Signer' is listed first in {sum(int(r['signer_first']) for r in multi)}; "
      f"every clip has a 'Signer' entry: {all(int(r['has_signer']) for r in rows)}  [c06-019]")
print(f"the old extraction (first array found in the entry order) equals the FIRST-LISTED entry in {eqf:,} of {len(multi):,} multi-person clips "
      f"and the 'Signer' entry in {eqs} -> the program kept a non-signer in {len(multi)-eqs:,} clips ({100*(len(multi)-eqs)/n:.1f}% of {n:,})  [c06-019]")
print(f"single-person clips ({len(single):,}): old extraction = the 'Signer' entry in {sum(int(r['old_clip_equals_signer']) for r in single):,}  [c06-019]")

miss = [r["clip"] for r in multi if not int(r["has_dwpose_alldet"])]
print(f"\nmulti-person clips with a DWPose file: {len(multi)-len(miss):,} of {len(multi):,}; without: {len(miss)} ({', '.join(miss)})  "
      f"[c21-015]")

# ---- registration of the distributed keypoints to the crop (video_11_6)
d = np.loadtxt(HERE / "registration_video_11_6.csv", delimiter=",", skiprows=1)
A, B = d[:, 2:4], d[:, 5:7]; keep = (d[:, 4] > 0.3) & (d[:, 7] > 0.5)
Am, Bm = A[keep].mean(0), B[keep].mean(0)
scale = ((A[keep] - Am) * (B[keep] - Bm)).sum() / ((A[keep] - Am) ** 2).sum(); shift = Bm - scale * Am
res = np.linalg.norm(scale * A[keep] + shift - B[keep], axis=1)
print(f"\nvideo_11_6: scale {scale:.3f}; median residual of the 11 shared body keypoints after alignment {np.median(res):.1f} pixels "
      f"({int(keep.sum())} keypoint pairs)  [c08-026]")
print(f"in that clip the first-listed array equals the signer array: {json.load(open(HERE/'registration_identity.json'))['first_listed_equals_signer_array']}")

# ---- extractor input size: the library's mode table (copied from its source) and the production setting
tab = (HERE / "extractor_mode_table.txt").read_text()
modes = {}
for name, body in re.findall(r"'(\w+)': \{(.*?)\n        \}", tab, re.S):
    wh = tuple(int(v) for v in re.search(r"'pose_input_size': \((\d+), (\d+)\)", body).groups())
    f = tuple(int(v) for v in re.search(r"-(\d+)x(\d+)_\d{8}\.zip", body).groups())   # checkpoint file name: HEIGHTxWIDTH
    assert wh == (f[1], f[0]), (name, wh, f)
    modes[name] = wh
worker = (ROOT / "pose" / "extraction" / "dwpose_extract_worker.py").read_text()
prod = re.search(r'"--mode", default="(\w+)"', worker).group(1)
print(f"\nextraction library mode table (input size height x width; the checkpoint file names agree): 'performance' {modes['performance'][1]}x{modes['performance'][0]}, "
      f"'balanced' {modes['balanced'][1]}x{modes['balanced'][0]}; the production extraction script defaults to '{prod}'  [c17-005]")

# ---- bounding-box audit: the audited sample (Communication training manifest, seed 42, 200 clips)
b = list(csv.DictReader(open(HERE / "bbox_audit_sample200.csv")))
frames = sum(int(x["frames"]) for x in b)
nv = {md: sum(int(x[md + "_valid_boxes"]) for x in b) for md in ("xywh", "xyxy")}
inside = {md: 100 * np.mean([int(x[md + "_inside"]) / int(x[md + "_sel"]) for x in b if int(x[md + "_sel"])]) for md in ("xywh", "xyxy")}
# GATE: recorded audit (research record, 5 Aug 2026): 200 clips, 15,231 frames, 14,910 / 6,271 valid boxes
gate(len(b) == 200 and frames == 15231 and nv == {"xywh": 14910, "xyxy": 6271},
     "audit sample recorded as 200 clips, 15,231 frames, 14,910 (xywh) and 6,271 (xyxy) valid boxes")
if all(m.exists() for m in man):
    import random
    tr = [Path(l.split("\t")[0]).name[:-4] for l in open(man[0]) if "\t" in l]
    order = list(range(len(tr))); random.Random(42).shuffle(order)
    gate([tr[i] for i in order[:200]] == [x["clip"] for x in b], "audited clips = first 200 of the seed-42 shuffle of the Communication training manifest")
print(f"\nbounding-box audit on {len(b)} Communication training clips ({frames:,} frames): confident joints inside the implied box: "
      f"box read as xywh {inside['xywh']:.1f}%, read as xyxy {inside['xyxy']:.1f}%  [c06-022]")
print(f"  valid (positive-size) boxes: xywh {nv['xywh']:,} of {frames:,}, xyxy {nv['xyxy']:,} of {frames:,}  [c06-022]")
# informative, untagged: the spread of the box-normalised coordinates under the natural definition (the original definition is not recorded)
for md in ("xywh", "xyxy"):
    N = sum(int(x[md + "_n"]) for x in b); S = sum(float(x[md + "_sum"]) for x in b); Q = sum(float(x[md + "_sumsq"]) for x in b)
    lo = min(float(x[md + "_min"]) for x in b if x[md + "_min"] != ""); hi = max(float(x[md + "_max"]) for x in b if x[md + "_max"] != "")
    print(f"  (not the audited table's definition) x/y normalised by the box, all 136 keypoints, frames with a positive box, read as {md}: "
          f"std {np.sqrt(Q/N-(S/N)**2):.2f}, extrema {lo:.1f} / {hi:.1f}")
