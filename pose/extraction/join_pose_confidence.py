"""Join the classical-CV per-clip table with shipped-AlphaPose hand confidence and
quantify what predicts pose failure (correlations + bucket analysis + resolution effect)."""
import csv, os, json, numpy as np

CV = os.path.expanduser("~/Honours/results/dwpose_experiment/classical_cv_all_clips.csv")
CLIPS = os.path.expanduser("~/Honours/full_run/clips")
OUT = os.path.expanduser("~/Honours/results/dwpose_experiment")
rows = list(csv.DictReader(open(CV)))

def conf(cid):
    p = os.path.join(CLIPS, cid + ".npz")
    if not os.path.exists(p):
        return None
    try:
        k = np.load(p)["KPS"].astype(np.float32)
        return float(k[:, 94:136, 2].mean()), float(k[:, 0:26, 2].mean())
    except Exception:
        return None

merged = []
for r in rows:
    c = conf(r["clip"])
    if c is None:
        continue
    r["alphapose_hand_conf"], r["alphapose_body_conf"] = round(c[0], 4), round(c[1], 4)
    merged.append(r)
print("joined clips:", len(merged), "/", len(rows))
col = lambda k: np.array([float(x[k]) for x in merged])
hc = col("alphapose_hand_conf")
print(f"AlphaPose hand_conf over all clips: mean={hc.mean():.3f} median={np.median(hc):.3f} p90={np.percentile(hc,90):.3f}")

corr = {}
for k in ["lap_var_mean","blurry_frame_frac","hand_px_mean","hand_blob_area_mean","width","height",
          "megapixels","framediff_p90","flow_mag_p95","hands_together_frac","multiperson_frac",
          "skin_frac_mean","face_size_frac","brightness_mean","contrast_mean","n_frames"]:
    corr[k] = round(float(np.corrcoef(col(k), hc)[0, 1]), 3)
print("\ncorr(AlphaPose hand_conf, X)  [what predicts hand confidence]:")
for k, v in sorted(corr.items(), key=lambda x: -abs(x[1])):
    print(f"  {k:20s} {v:+.3f}")

print("\nhand_conf: flagged-difficult clips vs the rest:")
buckets = {}
for f in ["flag_tiny_hands","flag_high_blur","flag_occlusion","flag_multiperson"]:
    y = [float(x["alphapose_hand_conf"]) for x in merged if int(float(x[f]))]
    n = [float(x["alphapose_hand_conf"]) for x in merged if not int(float(x[f]))]
    buckets[f] = {"flagged_mean": round(float(np.mean(y)), 3) if y else None, "n_flagged": len(y),
                  "rest_mean": round(float(np.mean(n)), 3) if n else None, "n_rest": len(n)}
    print(f"  {f:18s} flagged={buckets[f]['flagged_mean']} (n={len(y)})  rest={buckets[f]['rest_mean']} (n={len(n)})")

print("\nhand_conf by resolution quartile (megapixels):")
mp = col("megapixels"); q = np.percentile(mp, [25, 50, 75]); resq = {}
for lo, hi, lbl in [(-1, q[0], "Q1_smallest"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e9, "Q4_largest")]:
    m = [float(x["alphapose_hand_conf"]) for x in merged if lo < float(x["megapixels"]) <= hi]
    resq[lbl] = round(float(np.mean(m)), 3); print(f"  {lbl:12s} hand_conf={np.mean(m):.3f} (n={len(m)})")

with open(os.path.join(OUT, "cv_pose_joined.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(merged[0].keys())); w.writeheader(); w.writerows(merged)
json.dump({"n": len(merged), "hand_conf_mean": round(float(hc.mean()), 4),
           "correlations": corr, "buckets": buckets, "resolution_quartiles": resq},
          open(os.path.join(OUT, "cv_pose_findings.json"), "w"), indent=2)
print("\nsaved cv_pose_joined.csv + cv_pose_findings.json")
