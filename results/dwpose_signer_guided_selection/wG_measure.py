"""Local measurement of arm G (signer-guided DWPose). Pre-registration: results/PREREG_WAVEG_SIGNERGUIDED_2026-10-26.md section 2.
Run: ~/.venvs/honours_py312_twin/bin/python -I wG_measure.py [--only N]   (twin venv: numpy only)

Per clip (all 14,039 with an all-detections file and a fixed AlphaPose signer clip): r (the signature correlation of signer_check.py,
imported, not copied) against the AlphaPose signer for index-0, v1 (stored KPS), v2 (TUNED, V1 if all masked) and G; frames G != index-0;
fallback; sources. Then the GT scoring on the 61 human-labelled clips (independent of AlphaPose) and the residual categories.
"""
import argparse, csv, importlib.util, json, os, sys, time
from collections import Counter
from multiprocessing import Pool
import numpy as np

H = "."; OUT = os.path.dirname(os.path.abspath(__file__))
AP = f"{H}/full_run/clips_signer"; DW = f"{H}/results/dwpose_experiment/alldet_local"
SC = f"{H}/results/dwpose_signer_tracking_check"; TV2 = f"{H}/results/tracker_v2"
sys.path.insert(0, f"{H}/src/tools")
import wG_signerguided as W
tv2 = W.load_tv2(TV2)
spec = importlib.util.spec_from_file_location("signer_check", f"{SC}/signer_check.py"); sck = importlib.util.module_from_spec(spec); spec.loader.exec_module(sck)
sig, corr = (lambda k: sck.sig(k)[0]), sck.corr          # THE measure of signer_check.py


def one(n):
    d = dict(np.load(f"{DW}/{n}.npz")); a = np.load(f"{AP}/{n}.npz")["KPS"]
    T = len(d["NDET"]); sa = sig(a)
    c0 = np.zeros((T, 133, 3), np.float32)
    for t in range(T):
        if d["NDET"][t] > 0:
            c0[t, :, :2] = d["KP_ALL"][t, 0]; c0[t, :, 2] = d["SC_ALL"][t, 0]
    g = W.build_g(d, a, tv2); v2 = g["v2"]
    ne = lambda x, y: ~np.isclose(x, y).all(axis=(1, 2))
    gs = sig(g["KPS"])
    info = g["info"]
    return dict(clip=n, T=T, T_ap=len(a), r_idx0=corr(sa, sig(c0)), r_v1=corr(sa, sig(d["KPS"])), r_v2=corr(sa, sig(v2)), r_g=corr(sa, gs),
                rms_ap=float(np.linalg.norm(sa) / 16) if sa is not None else np.nan, rms_g=float(np.linalg.norm(gs) / 16) if gs is not None else np.nan,
                c0_ne_g=int(ne(c0, g["KPS"]).sum()), v2_ne_g=int(ne(v2, g["KPS"]).sum()), v1_ne_g=int(ne(d["KPS"], g["KPS"]).sum()),
                g_masked=int((g["chosen"] < 0).sum()), reason=info["reason"], v2_to_v1=int(info.get("v2_to_v1", False)),
                s=info.get("s", np.nan), n_fit=info.get("n_fit", np.nan), n_inlier=info.get("n_inlier", np.nan), med_inlier=info.get("med_inlier", np.nan),
                **{f"src_{k}": int((g["source"] == v).sum()) for k, v in W.SRC.items()})


def gt_scoring():
    sys.path.insert(0, TV2)
    import score_gt
    key = json.load(open(f"{TV2}/gt_key.json")); labels = json.load(open(f"{TV2}/gt_labels.json"))
    src = f"{H}/results/dwpose_experiment/alldet_kaya_gt61"
    def g_choice(d, n):
        ap = np.load(f"{AP}/{n}.npz")["KPS"]
        return W.build_g(d, ap, tv2)["chosen"].astype(int)
    # the labelled panels name the clip in `key`, the tracker functions get only the dict: wrap with the clip name via a lookup on DET_BOX bytes
    clips = {k["clip"] for k in key.values()}
    ident = {np.load(f"{src}/{c}.npz")["DET_BOX"].tobytes(): c for c in clips}
    trackers = {"index0": lambda d: np.where(d["NDET"] > 0, 0, -1), "v1": lambda d: d["CHOSEN"].astype(int),
                "v2-tuned": lambda d: tv2.choose(d, tv2.TUNED)[1].astype(int),
                "G": lambda d: g_choice(d, ident[d["DET_BOX"].tobytes()])}
    out = {}
    for part in ("dev", "heldout"):
        res = score_gt.evaluate(trackers, key, labels, __import__("pathlib").Path(src), part)
        out[part] = {k: dict(c) for k, c in res.items()}
        for k, c in res.items():
            print(f"[GT {part}] {k:9s} {score_gt.summary(c)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    per = list(csv.DictReader(open(f"{SC}/per_clip.csv")))
    multi = {r["clip"]: int(r["multi"]) for r in per}; pc = {r["clip"]: r for r in per}
    names = [r["clip"] for r in per if os.path.exists(f"{DW}/{r['clip']}.npz") and os.path.exists(f"{AP}/{r['clip']}.npz")]
    if a.only: names = names[:: max(1, len(names) // a.only)][: a.only]
    with Pool(a.workers) as pool:
        rows = pool.map(one, names, chunksize=16)
    print("clips", len(rows), "time", round(time.time() - t0))
    for r in rows: r["multi"] = multi[r["clip"]]
    with open(f"{OUT}/per_clip_g{'_subset' if a.only else ''}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    M = [r for r in rows if r["multi"]]
    share = lambda k, pop=M: float(np.mean([r[k] > 0.5 for r in pop]))
    R = dict(n_clips=len(rows), n_multi=len(M))
    R["share_r_gt_0p5"] = {k: share("r_" + k) for k in ("idx0", "v1", "v2", "g")}
    R["median_r"] = {k: float(np.nanmedian([r["r_" + k] for r in M])) for k in ("idx0", "v1", "v2", "g")}
    R["n_undefined"] = {k: int(np.isnan([r["r_" + k] for r in M]).sum()) for k in ("idx0", "v1", "v2", "g")}
    Fm = sum(r["T"] for r in M); Fa = sum(r["T"] for r in rows)
    R["frames_changed_vs_idx0"] = dict(multi=sum(r["c0_ne_g"] for r in M) / Fm, all=sum(r["c0_ne_g"] for r in rows) / Fa,
                                       n_clips_any_multi=sum(r["c0_ne_g"] > 0 for r in M), v2_vs_idx0_all=None)
    R["frames_g_ne_v2"] = dict(multi=sum(r["v2_ne_g"] for r in M) / Fm, all=sum(r["v2_ne_g"] for r in rows) / Fa)
    R["frames_g_ne_v1"] = dict(multi=sum(r["v1_ne_g"] for r in M) / Fm, all=sum(r["v1_ne_g"] for r in rows) / Fa)
    R["g_masked_frames_share"] = dict(multi=sum(r["g_masked"] for r in M) / Fm, all=sum(r["g_masked"] for r in rows) / Fa)
    R["fallback_clips"] = dict(all=dict(Counter(r["reason"] or "none" for r in rows)), multi=dict(Counter(r["reason"] or "none" for r in M)),
                               share_all=sum(bool(r["reason"]) for r in rows) / len(rows), share_multi=sum(bool(r["reason"]) for r in M) / len(M))
    R["fallback_frames_share"] = dict(all=sum(r["src_v2_fallback"] for r in rows) / Fa, multi=sum(r["src_v2_fallback"] for r in M) / Fm)
    R["source_share_frames_all"] = {k: sum(r["src_" + k] for r in rows) / Fa for k in W.SRC}
    R["source_share_frames_multi"] = {k: sum(r["src_" + k] for r in M) / Fm for k in W.SRC}
    R["g_gain"] = dict(fixed_vs_idx0=sum(r["r_idx0"] <= 0.5 < r["r_g"] for r in M), broke_vs_idx0=sum(r["r_g"] <= 0.5 < r["r_idx0"] for r in M),
                       fixed_vs_v1=sum(r["r_v1"] <= 0.5 < r["r_g"] for r in M), broke_vs_v1=sum(r["r_g"] <= 0.5 < r["r_v1"] for r in M),
                       fixed_vs_v2=sum(r["r_v2"] <= 0.5 < r["r_g"] for r in M), broke_vs_v2=sum(r["r_g"] <= 0.5 < r["r_v2"] for r in M))
    # residual of G: why r <= 0.5 (or undefined)? same thresholds as the REPORT
    res = json.load(open(f"{SC}/results.json"))["A_stored_KPS_v1tracked"]["thresholds"]
    cat = Counter(); bad = [r for r in M if not (r["r_g"] > 0.5)]
    for r in bad:
        p = pc[r["clip"]]
        if int(p["nconf_ap"]) < res["nconf_min"] or r["T"] < 8: c = "1_too_short_or_few_confident"
        elif r["rms_ap"] < res["p5_rms_ap_matched"] or not (r["rms_g"] >= res["p5_rms_dw_matched"]): c = "2_low_arm_motion"
        elif r["reason"]: c = "3_clip_fallback:" + r["reason"]
        elif r["g_masked"] > 0.3 * r["T"]: c = "4_G_masked_>30%_frames"
        elif r["src_carry"] > 0.2 * r["T"]: c = "5_carry_>20%_frames"
        else: c = "6_other_followed_AP_signer_but_arm_pose_differs"
        cat[c] += 1
    R["residual_g_mismatch"] = dict(n=len(bad), share_of_multi=len(bad) / len(M), table=dict(sorted(cat.items())), share_table={k: v / len(M) for k, v in sorted(cat.items())})
    R["same_residual_for_stored_v1"] = dict(n=sum(not (r["r_v1"] > 0.5) for r in M))
    # gate: reproduce the REPORT
    R["gate_reproduces_report"] = dict(idx0=abs(R["share_r_gt_0p5"]["idx0"] - 0.8068337) < 5e-4 if not a.only else None, v1=abs(R["share_r_gt_0p5"]["v1"] - 0.9493916) < 5e-4 if not a.only else None)
    if not a.only:
        R["gt61"] = gt_scoring()
    R["time_s"] = round(time.time() - t0)
    json.dump(R, open(f"{OUT}/results{'_subset' if a.only else ''}.json", "w"), indent=1, default=float)
    print(json.dumps({k: R[k] for k in R if k != "gt61"}, indent=1, default=float))


if __name__ == "__main__":
    main()
