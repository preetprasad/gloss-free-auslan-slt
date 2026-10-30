"""Do our DWPose keypoints follow the signer?  (2026-10-25)  Run: ~/.venvs/honours_py312_twin/bin/python -I signer_check.py
Stage 1 (this file): per-clip metrics for all 14,041 clips -> per_clip.csv, results.json.  panels.py makes the PNGs."""
import os, sys, json, glob, csv, time
import numpy as np
H = "."
OUT = os.path.dirname(os.path.abspath(__file__))
AP = f"{H}/full_run/clips_signer"; DW = f"{H}/results/dwpose_experiment/alldet_local"
PT = os.path.expanduser("~/Downloads/SP_Annotation_V1"); NS = f"{H}/results/nonsigner/npz"
sys.path.insert(0, f"{H}/results/tracker_v2"); import tracker_v2 as tv2

def sig(kps, conf_min=0.3, raw=False):          # identical to pairing_check.py (raw=True also returns pre-demean vector)
    k = kps[:, :11]
    ok = (k[:, 5, 2] > conf_min) & (k[:, 6, 2] > conf_min)
    if ok.sum() < 4:
        return (None, int(len(kps)), None) if raw else (None, int(len(kps)))
    k = k[ok]
    mid = (k[:, 5, :2] + k[:, 6, :2]) / 2
    w = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)[:, None]
    if np.median(w) <= 0:
        return (None, int(len(kps)), None) if raw else (None, int(len(kps)))
    rel = (k[:, [7, 8, 9, 10], :2] - mid[:, None]) / np.maximum(w, 1e-6)[:, None]
    rel = rel.reshape(len(k), -1)
    t = np.linspace(0, 1, len(k)); g = np.linspace(0, 1, 32)
    r = np.stack([np.interp(g, t, rel[:, j]) for j in range(rel.shape[1])], 1)
    r0 = r
    r = r - r.mean(0)
    return (r.ravel(), int(len(kps)), (r0.ravel(), int(ok.sum()))) if raw else (r.ravel(), int(len(kps)))

def corr(a, b):
    if a is None or b is None: return np.nan
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan

def nanmax(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return max(xs) if xs else np.nan

def person_path(KA, SA, j):
    out = np.zeros((KA.shape[0], 133, 3), np.float32)
    n = KA.shape[1]
    if j < n:
        out[:, :, :2] = KA[:, j]; out[:, :, 2] = SA[:, j]
    return out

def npeople():
    cp = f"{OUT}/npeople_cache.json"
    if os.path.exists(cp): return json.load(open(cp))
    import torch
    d = {}
    for f in sorted(glob.glob(f"{PT}/video_*.pt")):
        o = torch.load(f, map_location="cpu", weights_only=False)
        for name, c in o.items():
            d[name] = [str(k) for k in c["Annotation"].keys()]
    json.dump(d, open(cp, "w")); return d

def splits():
    s = {}
    for sp in ("train", "val", "test"):
        for l in open(f"{H}/full_run/manifests/{sp}.tsv"):
            s[os.path.basename(l.split("\t")[0])[:-4]] = sp
    return s

def main():
    t0 = time.time()
    keys = npeople(); spl = splits()
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{AP}/video_*.npz"))
    print("AP clips", len(names), "DW missing", sum(not os.path.exists(f"{DW}/{n}.npz") for n in names))
    nsmap = {}
    for f in glob.glob(f"{NS}/*_nonsigner.npz"):
        b = os.path.basename(f)[:-len("_nonsigner.npz")]; c, p = b.rsplit("_", 1); nsmap.setdefault(c, []).append(f)
    rows = []
    for ii, n in enumerate(names):
        p = f"{DW}/{n}.npz"
        if not os.path.exists(p): continue
        a = np.load(f"{AP}/{n}.npz")["KPS"]; d = np.load(p)
        K, KA, SA, CH, ND = d["KPS"], d["KP_ALL"], d["SC_ALL"], d["CHOSEN"].astype(int), d["NDET"].astype(int)
        T = len(K)
        sa, Tap, ra = sig(a, raw=True); sd, Tdw, rd = sig(K, raw=True)
        tr = np.zeros_like(K)
        for t in range(T):
            if CH[t] >= 0 and CH[t] < KA.shape[1]:
                tr[t, :, :2] = KA[t, CH[t]]; tr[t, :, 2] = SA[t, CH[t]]
        st = sig(tr)[0]                       # path rebuilt from CHOSEN (must equal stored KPS; checked below)
        P0 = person_path(KA, SA, 0); s0, _ = sig(P0)   # TRUE index-0 person = KP_ALL[:,0]
        rj = [corr(sa, sig(person_path(KA, SA, j))[0]) for j in range(KA.shape[1])]
        row = dict(clip=n, split=spl.get(n, "?"), keys="|".join(keys[n]), n_people=len(keys[n]) - 1 if "Signer" in keys[n] else len(keys[n]),
                   multi=int(len(keys[n]) > 1), T_ap=Tap, T_dw=Tdw, K0_eq_index0=int(np.allclose(K[:, :, :2], KA[:, 0]) if KA.shape[1] else 1),
                   nconf_ap=ra[1] if ra else 0, nconf_dw=rd[1] if rd else 0,
                   norm_ap=np.linalg.norm(ra[0]) if ra else np.nan, norm_dw=np.linalg.norm(rd[0]) if rd else np.nan,
                   rms_ap=np.linalg.norm(sa) / np.sqrt(256) if sa is not None else np.nan, rms_dw=np.linalg.norm(sd) / np.sqrt(256) if sd is not None else np.nan,
                   r0=corr(sa, sd), r_kps=corr(sa, sd), r_tracked=corr(sa, st), r_idx0=corr(sa, s0),
                   kps_eq_chosen=int(np.allclose(K[:, :, :2], tr[:, :, :2])), idx0_eq_kps=int(np.allclose(K[:, :, :2], P0[:, :, :2])),
                   rms_idx0=np.linalg.norm(s0) / np.sqrt(256) if s0 is not None else np.nan, nconf_idx0=int(((P0[:, 5, 2] > .3) & (P0[:, 6, 2] > .3)).sum()), r_best_index=nanmax(rj), best_index=int(np.nanargmax(rj)) if not np.all(np.isnan(rj)) else -1,
                   n_switch=int((CH > 0).sum()), n_nodet_chosen=int((CH < 0).sum()), n_chosen_nonzero_or_none=int((CH != 0).sum()),
                   share_multidet=float(np.mean(ND > 1)), max_ndet=int(ND.max()))
        row["r_best_ns"] = row["r_dw0_ns"] = np.nan; row["n_ns"] = 0
        rows.append(row)
        if ii % 2000 == 0: print(ii, round(time.time() - t0), flush=True)
    # non-signer crops (dev/test subset)
    for row in rows:
        fs = nsmap.get(row["clip"], [])
        if fs:
            sa = sig(np.load(f"{AP}/{row['clip']}.npz")["KPS"])[0]; sd = sig(np.load(f"{DW}/{row['clip']}.npz")["KPS"])[0]
            ss = [sig(np.load(f)["KPS"])[0] for f in fs]
            row["n_ns"] = len(fs); row["r_best_ns"] = nanmax([corr(sa, s) for s in ss]); row["r_dw0_ns"] = nanmax([corr(sd, s) for s in ss])
    return rows

def classify(rows, ns_ok=True):
    multi = [r for r in rows if r["multi"]]
    r0 = np.array([r["r0"] for r in multi])
    matched = [r for r in multi if r["r0"] > 0.5]
    p5_ap = float(np.percentile([r["rms_ap"] for r in matched], 5)); p5_dw = float(np.percentile([r["rms_dw"] for r in matched], 5))
    return multi, matched, p5_ap, p5_dw

if __name__ == "__main__":
    rows = main()
    # ---- stage 2: tracker_v2 path/tracklets for clips where the stored-KPS or true index-0 signature mismatches
    multi = [r for r in rows if r["multi"]]
    for r in rows: r["r_v2_path"] = r["r_best_tracklet"] = np.nan
    need = [r for r in rows if not (r["r_kps"] > 0.5) or not (r["r_idx0"] > 0.5)]
    for r in need:
        d = np.load(f"{DW}/{r['clip']}.npz"); sa = sig(np.load(f"{AP}/{r['clip']}.npz")["KPS"])[0]
        KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]; T = len(ND)
        try:
            KPS2, _ = tv2.choose(d); r["r_v2_path"] = corr(sa, sig(KPS2)[0])
            cands = [tv2.candidates(SA[t], DB[t], int(ND[t])) for t in range(T)]
            best = []
            for tk in tv2.tracklets(DB, cands):
                if len(tk) < 8: continue
                P = np.zeros((T, 133, 3), np.float32)
                for t, j in tk.items(): P[t, :, :2] = KA[t, j]; P[t, :, 2] = SA[t, j]
                best.append(corr(sa, sig(P)[0]))
            r["r_best_tracklet"] = nanmax(best)
        except Exception as e:
            print("v2 fail", r["clip"], e)
    NCONF_MIN = 8
    R = {"n_clips": len(rows), "n_multi": len(multi), "n_single": len(rows) - len(multi)}
    R["dw_missing"] = ["x"] if False else 14041 - len(rows)
    R["kps_eq_chosen_share"] = float(np.mean([r["kps_eq_chosen"] for r in rows])); R["idx0_eq_kps_share"] = float(np.mean([r["idx0_eq_kps"] for r in rows]))
    def gate(key, pop):
        x = np.array([r[key] for r in pop]); return dict(n=len(pop), share_r_gt_0p5=float(np.mean(x > 0.5)), share_r_gt_0p5_of_defined=float(np.mean(x[~np.isnan(x)] > 0.5)),
                                                         median_r=float(np.nanmedian(x)), n_undefined=int(np.isnan(x).sum()))
    R["gate_stored_KPS"] = gate("r_kps", multi); R["gate_true_index0"] = gate("r_idx0", multi)
    R["gate_note"] = "today's reference: 95.4% / 0.977 on 12,411 multi clips (file KPS). Mine has 12,409: 2 clips lack an alldet npz."
    def run(label, rkey, rmsap, rmsdw, ncdw, altkeys):
        matched = [r for r in multi if r[rkey] > 0.5]
        p5_ap = float(np.percentile([r["rms_ap"] for r in matched], 5)); p5_dw = float(np.percentile([r[rmsdw] for r in matched], 5))
        def cat(r):
            if np.isnan(r[rkey]) or r["nconf_ap"] < NCONF_MIN or r[ncdw] < NCONF_MIN: return "1_too_short_or_few_confident"
            if r["rms_ap"] < p5_ap or r[rmsdw] < p5_dw: return "2_low_arm_motion"
            if nanmax([r[k] for k in altkeys[0]]) > 0.5: return "3_switch_explained"
            if nanmax([r[k] for k in altkeys[1]]) > 0.5: return "4_other_person_matches"
            return "5_unexplained"
        c = {r["clip"]: ("matched" if r[rkey] > 0.5 else cat(r)) for r in multi}
        cats = ["1_too_short_or_few_confident", "2_low_arm_motion", "3_switch_explained", "4_other_person_matches", "5_unexplained"]
        tab = {}
        for sp in ("all", "train", "val", "test"):
            base = [r for r in multi if sp == "all" or r["split"] == sp]
            t = dict(n_multi=len(base), n_mismatch=sum(c[r["clip"]] != "matched" for r in base))
            for k in cats: t[k] = sum(c[r["clip"]] == k for r in base); t[k + "_share"] = t[k] / len(base)
            t["mismatch_share"] = t["n_mismatch"] / len(base); tab[sp] = t
        return c, dict(thresholds=dict(p5_rms_ap_matched=p5_ap, p5_rms_dw_matched=p5_dw, nconf_min=NCONF_MIN), table=tab)
    # A: stored KPS (= v1 tracked path).  'switch-explained' = the TRUE index-0 path or tracker_v2 path matches.
    cA, R["A_stored_KPS_v1tracked"] = run("A", "r_kps", "rms_ap", "rms_dw", "nconf_dw", (["r_idx0", "r_v2_path"], ["r_best_index", "r_best_tracklet"]))
    # B: true index-0 (training input).  'switch-explained' = the v1 tracked path (stored KPS) or tracker_v2 path matches.
    cB, R["B_true_index0"] = run("B", "r_idx0", "rms_ap", "rms_idx0", "nconf_idx0", (["r_kps", "r_v2_path"], ["r_best_index", "r_best_tracklet"]))
    for r in rows:
        r["catA_stored_KPS"] = cA.get(r["clip"], "single_person"); r["catB_index0"] = cB.get(r["clip"], "single_person")
    cols = list(rows[0].keys())
    with open(f"{OUT}/per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    # (4) analogue of the AlphaPose bug: index-0 does not match, something else does
    def alt(r): return nanmax([r["r_kps"], r["r_v2_path"], r["r_best_index"], r["r_best_tracklet"]])
    f4 = {}
    for nm_, pop in (("multi", multi), ("single", [r for r in rows if not r["multi"]]), ("all", rows)):
        wr = [(not np.isnan(r["r_idx0"])) and r["r_idx0"] <= 0.5 and alt(r) > 0.5 for r in pop]
        wr_t = [(not np.isnan(r["r_idx0"])) and r["r_idx0"] <= 0.5 and nanmax([r["r_kps"], r["r_v2_path"]]) > 0.5 for r in pop]
        f4[nm_] = dict(n=len(pop), index0_wrong_person_any_alternative=float(np.mean(wr)), count=int(sum(wr)),
                       index0_wrong_person_tracked_path_matches=float(np.mean(wr_t)), count_tracked=int(sum(wr_t)),
                       index0_undefined=float(np.mean([np.isnan(r["r_idx0"]) for r in pop])),
                       index0_r_le_0p5_any_cause=float(np.mean([(r["r_idx0"] <= 0.5) for r in pop if not np.isnan(r["r_idx0"])])))
    R["analogue_AlphaPose_bug"] = f4
    R["switch"] = dict(share_multi_clips_with_CHOSEN_ne_0_frames=float(np.mean([r["n_switch"] > 0 for r in multi])),
                       mean_share_frames_CHOSEN_ne_0=float(np.mean([r["n_switch"] / r["T_dw"] for r in multi])))
    for lab, cc in (("A", "catA_stored_KPS"), ("B", "catB_index0")):
        ns = [r for r in multi if r[cc] not in ("matched",) and r["n_ns"] > 0]
        key = "r_kps" if lab == "A" else "r_idx0"
        R["nonsigner_crops_" + lab] = dict(n_mismatched_with_nonsigner_crop=len(ns), dw_path_matches_a_nonsigner_gt0p5=int(sum(((r["r_dw0_ns"] if lab == "A" else r["r_dw0_ns"]) > 0.5) for r in ns)),
                                          ap_signer_matches_a_nonsigner_gt0p5=int(sum(r["r_best_ns"] > 0.5 for r in ns)))
    json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
