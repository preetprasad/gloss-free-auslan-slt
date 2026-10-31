"""One-off data preparation (run by hand, NOT by make). Reads the released Auslan-Daily pose files and the local
extraction outputs and writes small text tables used by check.py. Needs ~/Downloads/SP_Annotation_V1 (the released
`SP_Annotation_V1/video_*.pt`), full_run/clips (the old first-listed-person extraction) and
pose/extraction/alldet_local (DWPose, all detections). Python: honours_py312_twin, run with -I."""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/SP_Annotation_V1"))
CLIPS = ROOT / "full_run" / "clips"
DWALL = ROOT / "pose" / "extraction" / "alldet_local"

def arr(x): return x.numpy() if torch.is_tensor(x) else np.asarray(x)

SEC = set(sys.argv[2:]) or {'pt', 'dw', 'bbox', 'modes'}   # sections to run: pt dw bbox modes (argv[1] = path of SP_Annotation_V1)

if 'pt' in SEC:
    rows = []
    for f in sorted(glob.glob(str(PT / "video_*.pt")), key=lambda p: int(Path(p).stem.split("_")[1])):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for clip, rec in d.items():
            ann = rec["Annotation"]
            keys = [str(k) for k in ann.keys()]
            first = next(iter(ann.keys()))
            shapes = {str(k): (tuple(arr(v["KPS"]).shape), tuple(arr(v["BBOX"]).shape)) for k, v in ann.items()}
            sk = shapes["Signer"][0] if "Signer" in shapes else None
            fk = shapes[str(first)][0]
            old = CLIPS / (clip + ".npz")
            eq_first = eq_signer = ""
            if old.exists():
                o = np.load(old)["KPS"]
                eq_first = int(o.shape == tuple(fk) and np.array_equal(o, arr(ann[first]["KPS"]).astype(np.float32)))
                if "Signer" in ann:
                    eq_signer = int(o.shape == tuple(sk) and np.array_equal(o, arr(ann["Signer"]["KPS"]).astype(np.float32)))
            rows.append(dict(clip=clip, n_keys=len(keys), has_signer=int("Signer" in keys), first_key=str(first),
                             signer_first=int(str(first) == "Signer"), T=fk[0], K=fk[1], C=fk[2],
                             all_K136=int(all(s[0][1] == 136 and s[0][2] == 3 for s in shapes.values())),
                             bbox_cols=shapes[str(first)][1][1],
                             old_clip_equals_first_listed=eq_first, old_clip_equals_signer=eq_signer,
                             has_dwpose_alldet=int((DWALL / (clip + ".npz")).exists())))
        print(f, len(rows), flush=True)
    with open(HERE / "pt_clip_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if 'dw' in SEC:
    # DWPose layout: shape of one stored array
    z = np.load(DWALL / "video_1_0.npz")
    json.dump({"clip": "video_1_0", "KPS_shape": list(z["KPS"].shape), "KP_ALL_shape": list(z["KP_ALL"].shape)},
              open(HERE / "dwpose_layout.json", "w"), indent=1)

    # registration of the distributed keypoints to the crop, clip video_11_6: the 11 shared body keypoints of both sets
    ap = np.load(ROOT / "full_run" / "clips" / "video_11_6.npz")["KPS"]
    aps = np.load(ROOT / "full_run" / "clips_signer" / "video_11_6.npz")["KPS"]
    dw = np.load(ROOT / "pose" / "extraction" / "dwpose_clips" / "video_11_6.npz")["KPS"]
    with open(HERE / "registration_video_11_6.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "k", "ap_x", "ap_y", "ap_c", "dw_x", "dw_y", "dw_c"])
        for t in range(min(len(ap), len(dw))):
            for k in range(11):
                w.writerow([t, k, *ap[t, k], *dw[t, k]])
    json.dump({"first_listed_equals_signer_array": bool(ap.shape == aps.shape and np.array_equal(ap, aps))},
              open(HERE / "registration_identity.json", "w"))


    # DWPose keypoint count of every stored all-detections file (K = shape[2] of KP_ALL (T, persons, K, 2))
    from collections import Counter
    kc = Counter(int(np.load(f)["KP_ALL"].shape[2]) for f in sorted(DWALL.glob("*.npz")))
    json.dump({"files": sum(kc.values()), "keypoints_per_person_counts": {str(k): v for k, v in kc.items()}},
              open(HERE / "dwpose_k_counts.json", "w"), indent=1)

if 'bbox' in SEC:
    # bounding-box format audit. The audited sample of 2026-08-05 is reproduced by tools/audit_bbox_format.py on the
    # Communication training manifest (comm_run/manifests/train.tsv, seed 42, 200 clips: 15,231 frames, 14,910 / 6,271 valid boxes).
    # Per-clip counts use exactly that tool's rule (confident body+hand joints, confidence > 0.1, positive boxes).
    import random
    sys.path.insert(0, str(ROOT / "src"))
    from slt.data import load_pose, read_manifest
    rows_m = list(read_manifest(str(ROOT / "comm_run" / "manifests" / "train.tsv")))
    random.Random(42).shuffle(rows_m)
    BH = list(range(26)) + list(range(94, 136))
    out = []
    for p, _ in rows_m[:200]:
        kp, bb = load_pose(p); kp = np.asarray(kp, dtype=np.float64); bb = np.asarray(bb).reshape(len(kp), -1)[:, :4]
        xy, cf = kp[:, BH, :2], kp[:, BH, 2]
        rec = {"clip": Path(p).stem, "frames": len(kp)}
        for mode in ("xywh", "xyxy"):
            x1, y1 = bb[:, 0, None], bb[:, 1, None]
            x2, y2 = (x1 + bb[:, 2, None], y1 + bb[:, 3, None]) if mode == "xywh" else (bb[:, 2, None], bb[:, 3, None])
            ok = (x2 > x1) & (y2 > y1); sel = (cf > 0.1) & ok
            inside = (xy[..., 0] >= x1) & (xy[..., 0] <= x2) & (xy[..., 1] >= y1) & (xy[..., 1] <= y2) & sel
            rec[f"{mode}_sel"] = int(sel.sum()); rec[f"{mode}_inside"] = int(inside.sum()); rec[f"{mode}_valid_boxes"] = int(ok.sum())
            # informative only: x/y normalised by the box ([0,1] units), all 136 keypoints, frames with a positive box
            w, h = (x2 - x1)[:, 0], (y2 - y1)[:, 0]; fo = (w > 1e-3) & (h > 1e-3)
            z = np.concatenate([((kp[fo, :, 0] - x1[fo]) / w[fo, None]).ravel(), ((kp[fo, :, 1] - y1[fo]) / h[fo, None]).ravel()])
            rec[f"{mode}_n"] = z.size
            rec[f"{mode}_sum"] = float(z.sum()) if z.size else 0.0; rec[f"{mode}_sumsq"] = float((z ** 2).sum()) if z.size else 0.0
            rec[f"{mode}_min"] = float(z.min()) if z.size else ""; rec[f"{mode}_max"] = float(z.max()) if z.size else ""
        out.append(rec)
    with open(HERE / "bbox_audit_sample200.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

if 'modes' in SEC:
    # the MODE table of the extraction library's whole-body solution, copied line by line (program lines only)
    import importlib.util
    src = (Path(importlib.util.find_spec("rtmlib").origin).parent / "tools" / "solution" / "wholebody.py").read_text().splitlines()
    i = next(k for k, l in enumerate(src) if l.strip().startswith("MODE = {"))
    j = next(k for k in range(i, len(src)) if src[k] == "    }")
    (HERE / "extractor_mode_table.txt").write_text("\n".join(src[i:j + 1]) + "\n")
print("done")
