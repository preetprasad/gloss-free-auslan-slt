"""Which of {index-0 = KP_ALL[:,0], stored alldet KPS (v1 tracked)} are the training npz (results/dwpose_experiment/dwpose_clips)?"""
import sys, glob, os, csv, json
import numpy as np
sys.argv=['x']; import importlib.util
sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "signer_check.py")); sc = importlib.util.module_from_spec(sp); sp.loader.exec_module(sc)
rows = {r["clip"]: r for r in csv.DictReader(open(os.path.join(sc.OUT, "per_clip.csv")))}
H = sc.H; res = []
for f in sorted(glob.glob(f"{H}/results/dwpose_experiment/dwpose_clips/video_*.npz")):
    n = os.path.basename(f)[:-4]
    if n not in rows: continue
    o = np.load(f)["KPS"]; d = np.load(f"{sc.DW}/{n}.npz"); A = d["KP_ALL"][:, 0]; K = d["KPS"]
    T = min(len(o), len(A))
    res.append(dict(clip=n, multi=int(rows[n]["multi"]), frac_eq_idx0=float(np.mean(np.abs(o[:T, :, :2] - A[:T]).max((1, 2)) < 1.0)),
                    frac_eq_kps=float(np.mean(np.abs(o[:T, :, :2] - K[:T, :, :2]).max((1, 2)) < 1.0)), same_T=int(len(o) == len(A)),
                    idx0_eq_kps=int(rows[n]["idx0_eq_kps"])))
s = [r for r in res if not r["idx0_eq_kps"]]   # informative: clips where the two differ
out = dict(n_train_npz_checked=len(res), n_where_idx0_differs_from_kps=len(s),
           median_frac_frames_equal_idx0=float(np.median([r["frac_eq_idx0"] for r in s])) if s else None,
           median_frac_frames_equal_kps=float(np.median([r["frac_eq_kps"] for r in s])) if s else None,
           share_closer_to_idx0=float(np.mean([r["frac_eq_idx0"] > r["frac_eq_kps"] for r in s])) if s else None)
json.dump(out, open(f"{sc.OUT}/trainclips_check.json", "w"), indent=1); print(out)
