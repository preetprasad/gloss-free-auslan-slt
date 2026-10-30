"""Wave G: signer-guided DWPose (arm G). Pre-registration: results/PREREG_WAVEG_SIGNERGUIDED_2026-10-26.md (section 1 = this algorithm,
every constant below is frozen there).

Per clip, per frame, pick the DWPose detection that is the signer according to the dataset's own hand-corrected AlphaPose 'Signer'
track. The two live in different coordinates (AlphaPose: whole-video pixels, DWPose: signer-crop pixels), so a per-clip similarity
transform x' = s*x + t is fitted (RANSAC over shoulder pairs, then robust least squares on confident joints), frames are assigned by
distance with a Viterbi pass that keeps identity temporally consistent, and the clip falls back to tracker v2 when the AlphaPose
signer is missing or the fit is unreliable. The source that decided each frame is recorded in a side file.

  python tools/wG_signerguided.py selftest
  python tools/wG_signerguided.py build   --alldet D --ap D --w9man D --outpose D --outman D [--tv2 DIR] [--workers N] [--only N]
  python tools/wG_signerguided.py check   --alldet D --ap D --w9man D --w9pose D --outpose D --outman D [--tv2 DIR]
  python tools/wG_signerguided.py matchrate --alldet D --ap D --w9pose D --outpose D --npeople JSON [--min_gain 0.01]
tracker_v2.py is imported from --tv2 (or $WG_TV2_DIR); the file is md5-pinned by the job script.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# ---- frozen constants (pre-registration section 1)
CONF = 0.3
J = (0, 5, 6, 7, 8, 9, 10, 11, 12)          # nose, shoulders, elbows, wrists, hips (COCO order, shared by AlphaPose HALPE-136 and DWPose)
NJ = len(J)
MIN_COMMON = 3                              # joints needed for a distance
MIN_EVID_FIT = 8                            # fit frames needed at all
HYP_FRAMES = 24
TOL = 0.5                                   # inlier distance, in units of the (scaled) median AlphaPose shoulder width
S_LO, S_HI = 0.7, 1.4
MIN_INLIER_FRAMES, MIN_INLIER_SHARE = 8, 0.5
REFINE_ROUNDS = 3
UNARY_CAP = 2.0
MASK_COST_EVID = 1.0
PRIOR_OTHER = 0.2
SWITCH_COST = 0.6
LINK_IOU = 0.3
MASK_TRANSITION = 0.1
PRIOR_BIAS = 0.1                            # Deviation 2: on evidence frames every state other than tracker v2's choice costs +0.1
MAX_LEN_DIFF = 2
SRC = {"ap": 0, "ap_mask": 1, "carry": 2, "v2prior": 3, "v2_fallback": 4}
REASONS = ("", "no_ap", "length_mismatch", "fit_unreliable", "all_masked")


def load_tv2(d=None):
    d = d or os.environ.get("WG_TV2_DIR") or str(Path(__file__).resolve().parents[2] / "results" / "tracker_v2")
    spec = importlib.util.spec_from_file_location("tracker_v2", os.path.join(d, "tracker_v2.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["tracker_v2"] = m; spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ geometry
def _trim_mean_last(x, valid):
    """Mean of the best 75% (ceil) of the valid entries over the last axis; inf where fewer than MIN_COMMON are valid. x, valid: (..., NJ).
    (Deviation 1 of the pre-registration: a plain median let static torso joints decide, see the self-test.)"""
    cnt = valid.sum(-1)
    s = np.sort(np.where(valid, x, np.inf), axis=-1)
    cs = np.cumsum(np.where(np.isfinite(s), s, 0.0), axis=-1)
    k = np.clip(np.ceil(0.75 * cnt).astype(int), 1, NJ)
    m = np.take_along_axis(cs, (k - 1)[..., None], -1)[..., 0] / k
    return np.where(cnt >= MIN_COMMON, m, np.inf)


def dist_matrix(AK, AC, DK, DC, DV, s, t, w0):
    """(T, C) normalised distance of every candidate to the transformed AlphaPose signer; inf = no common joints / no candidate."""
    P = s * AK + t                                                   # (T, NJ, 2)
    dd = np.linalg.norm(DK - P[:, None], axis=-1)                    # (T, C, NJ)
    valid = AC[:, None, :] & DC
    med = _trim_mean_last(dd, valid) / max(w0, 1e-6)
    return np.where(DV, med, np.inf)


def fit_similarity(a, d):
    """Least-squares s, t with d ~ s*a + t (no rotation). a, d: (n, 2)."""
    am, dm = a.mean(0), d.mean(0)
    den = ((a - am) ** 2).sum()
    if den <= 1e-9:
        return None
    s = float(((a - am) * (d - dm)).sum() / den)
    return s, dm - s * am


def prepare(d, ap, tv2):
    KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]
    T = len(ND)
    cands = [tv2.candidates(SA[t], DB[t], int(ND[t])) for t in range(T)]
    C = max([len(c) for c in cands] + [1])
    DK = np.zeros((T, C, NJ, 2), np.float32); DC = np.zeros((T, C, NJ), bool); DV = np.zeros((T, C), bool)
    for t, cs in enumerate(cands):
        for k, i in enumerate(cs):
            DK[t, k] = KA[t, i][list(J)]; DC[t, k] = SA[t, i][list(J)] > CONF; DV[t, k] = True
    AK = np.zeros((T, NJ, 2), np.float32); AC = np.zeros((T, NJ), bool)
    n = min(T, len(ap))
    AK[:n] = ap[:n][:, list(J)][..., :2]; AC[:n] = ap[:n][:, list(J)][..., 2] > CONF
    return cands, DK, DC, DV, AK, AC


def fit_transform(AK, AC, DK, DC, DV):
    """RANSAC + refinement. Returns (s, t, w0, n_fit, n_inlier, med_inlier_dist) or None (reason no_ap) / 'unreliable' info."""
    sh = (AC[:, 1] & AC[:, 2])                                       # J index 1, 2 = left / right shoulder (COCO 5, 6)
    fit = np.where(sh)[0]
    if len(fit) < MIN_EVID_FIT:
        return "no_ap", dict(n_fit=int(len(fit)))
    w0a = float(np.median(np.linalg.norm(AK[fit, 1] - AK[fit, 2], axis=1)))
    if w0a <= 1e-6:
        return "no_ap", dict(n_fit=int(len(fit)))
    pick = fit[np.unique(np.linspace(0, len(fit) - 1, min(HYP_FRAMES, len(fit))).round().astype(int))]
    hyps = []
    for t0 in pick:
        wa = np.linalg.norm(AK[t0, 1] - AK[t0, 2])
        if wa <= 1e-6:
            continue
        for k in range(DK.shape[1]):
            if DV[t0, k] and DC[t0, k, 1] and DC[t0, k, 2]:
                wd = np.linalg.norm(DK[t0, k, 1] - DK[t0, k, 2]); s = wd / wa
                if S_LO <= s <= S_HI:
                    tr = (DK[t0, k, 1] + DK[t0, k, 2]) / 2 - s * (AK[t0, 1] + AK[t0, 2]) / 2
                    hyps.append((s, tr))

    def score(s, tr):
        dm = dist_matrix(AK, AC, DK, DC, DV, s, tr, s * w0a)
        dmin = dm.min(1)
        inl = sh & (dmin < TOL)
        credit = float(np.clip(1.0 - dmin[sh] / TOL, 0.0, 1.0).sum())
        return int(inl.sum()), credit, dm, inl

    best = None
    for h, (s, tr) in enumerate(hyps):
        n, cr, _, _ = score(s, tr)
        key = (-cr, h)
        if best is None or key < best[0]:
            best = (key, s, tr)
    if best is None:
        return "fit_unreliable", dict(n_fit=int(len(fit)), n_inlier=0)
    _, s, tr = best
    for _ in range(REFINE_ROUNDS):
        n, cr, dm, inl = score(s, tr)
        if n < 3:
            break
        A, D = [], []
        for t in np.where(inl)[0]:
            k = int(np.argmin(dm[t]))
            m = AC[t] & DC[t, k]
            A.append(AK[t][m]); D.append(DK[t, k][m])
        A, D = np.concatenate(A), np.concatenate(D)
        f = fit_similarity(A, D)
        if f is None:
            break
        r = np.linalg.norm(f[0] * A + f[1] - D, axis=1); keep = r <= 2 * max(np.median(r), 1e-6)
        if keep.sum() >= 3:
            f2 = fit_similarity(A[keep], D[keep])
            f = f2 if f2 is not None else f
        s, tr = f
        if not (S_LO <= s <= S_HI):
            break
    n, cr, dm, inl = score(s, tr)
    med = float(np.median(dm.min(1)[inl])) if inl.any() else float("inf")
    info = dict(n_fit=int(len(fit)), n_inlier=n, med_inlier=med, s=float(s), tx=float(tr[0]), ty=float(tr[1]))
    ok = n >= MIN_INLIER_FRAMES and n / len(fit) >= MIN_INLIER_SHARE and S_LO <= s <= S_HI
    return ("ok" if ok else "fit_unreliable"), dict(info, w0=float(s * w0a), s_=s, tr_=tr)


def viterbi(cands, DB, dist, evid, ch2, tv2):
    T = len(cands)
    prev_cost = None; back = []
    for t in range(T):
        n = len(cands[t]) + 1                                         # last state = MASK
        u = np.zeros(n)
        if evid[t]:
            for k in range(n - 1):
                dk = dist[t, k]; u[k] = UNARY_CAP if not np.isfinite(dk) else min(dk / TOL, UNARY_CAP)
                u[k] += 0.0 if cands[t][k] == ch2[t] else PRIOR_BIAS
            u[n - 1] = MASK_COST_EVID + (0.0 if ch2[t] < 0 else PRIOR_BIAS)
        else:
            for k in range(n - 1):
                u[k] = 0.0 if cands[t][k] == ch2[t] else PRIOR_OTHER
            u[n - 1] = 0.0 if ch2[t] < 0 else PRIOR_OTHER
        if prev_cost is None:
            prev_cost = u; back.append(None); continue
        m = len(cands[t - 1]) + 1
        tr = np.zeros((m, n))
        for a in range(m):
            for b in range(n):
                if a == m - 1 and b == n - 1:
                    tr[a, b] = 0.0
                elif a == m - 1 or b == n - 1:
                    tr[a, b] = MASK_TRANSITION
                else:
                    tr[a, b] = 0.0 if tv2.iou(DB[t - 1, cands[t - 1][a]], DB[t, cands[t][b]]) >= LINK_IOU else SWITCH_COST
        tot = prev_cost[:, None] + tr
        arg = tot.argmin(0)                                           # first minimum: lower index wins ties
        prev_cost = tot[arg, np.arange(n)] + u; back.append(arg)
    state = int(np.argmin(prev_cost)); path = [state]
    for t in range(T - 1, 0, -1):
        state = int(back[t][state]); path.append(state)
    path = path[::-1]
    return np.array([-1 if path[t] == len(cands[t]) else cands[t][path[t]] for t in range(T)], np.int16)


def v2_arm(d, tv2):
    """The wave-9 V2 arm exactly as materialise_arms.arms_for builds it: (KPS, chosen, fell_back_to_v1)."""
    k2, ch = tv2.choose(d, tv2.TUNED)
    if not (ch >= 0).any():
        return d["KPS"].astype(np.float32).copy(), d["CHOSEN"].astype(np.int16), True
    return k2, ch, False


def gather(d, chosen):
    KA, SA = d["KP_ALL"], d["SC_ALL"]
    out = np.zeros((len(chosen), 133, 3), np.float32)
    for t, i in enumerate(chosen):
        if i >= 0:
            out[t, :, :2] = KA[t, i]; out[t, :, 2] = SA[t, i]
    return out


def build_g(d, ap, tv2):
    """d: all-detections dict; ap: AlphaPose signer KPS (T_ap, 136, 3) or None. Returns dict(KPS, chosen, source, info)."""
    T = len(d["NDET"])
    k2, ch2, v1fb = v2_arm(d, tv2)

    def fallback(reason, **kw):
        return dict(KPS=k2, v2=k2, chosen=ch2.astype(np.int16), source=np.full(T, SRC["v2_fallback"], np.uint8),
                    info=dict(reason=reason, v2_to_v1=bool(v1fb), **kw))
    if ap is None or len(ap) == 0:
        return fallback("no_ap")
    if abs(len(ap) - T) > MAX_LEN_DIFF:
        return fallback("length_mismatch", T_ap=int(len(ap)))
    cands, DK, DC, DV, AK, AC = prepare(d, ap, tv2)
    status, info = fit_transform(AK, AC, DK, DC, DV)
    if status != "ok":
        return fallback("no_ap" if status == "no_ap" else "fit_unreliable", **{k: v for k, v in info.items() if not k.endswith("_")})
    s, tr, w0 = info["s_"], info["tr_"], info["w0"]
    dist = dist_matrix(AK, AC, DK, DC, DV, s, tr, w0)
    evid = AC.sum(1) >= MIN_COMMON
    chosen = viterbi(cands, d["DET_BOX"], dist, evid, ch2, tv2)
    if not (chosen >= 0).any():
        return fallback("all_masked", **{k: v for k, v in info.items() if not k.endswith("_")})
    src = np.zeros(T, np.uint8)
    for t in range(T):
        if evid[t]:
            if chosen[t] < 0:
                src[t] = SRC["ap_mask"]
            else:
                k = cands[t].index(int(chosen[t]))
                src[t] = SRC["ap"] if dist[t, k] < TOL else SRC["carry"]
        else:
            src[t] = SRC["v2prior"] if chosen[t] == ch2[t] else SRC["carry"]
    return dict(KPS=gather(d, chosen), v2=k2, chosen=chosen, source=src,
                info=dict(reason="", v2_to_v1=False, **{k: v for k, v in info.items() if not k.endswith("_")}))


# ------------------------------------------------------------------ signature measure (identical to results/dwpose_signer_tracking_check/signer_check.py)
def sig(kps, conf_min=0.3):
    k = kps[:, :11]
    ok = (k[:, 5, 2] > conf_min) & (k[:, 6, 2] > conf_min)
    if ok.sum() < 4:
        return None
    k = k[ok]
    mid = (k[:, 5, :2] + k[:, 6, :2]) / 2
    w = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1)[:, None]
    if np.median(w) <= 0:
        return None
    rel = (k[:, [7, 8, 9, 10], :2] - mid[:, None]) / np.maximum(w, 1e-6)[:, None]
    rel = rel.reshape(len(k), -1)
    t = np.linspace(0, 1, len(k)); g = np.linspace(0, 1, 32)
    r = np.stack([np.interp(g, t, rel[:, j]) for j in range(rel.shape[1])], 1)
    return (r - r.mean(0)).ravel()


def corr(a, b):
    if a is None or b is None:
        return np.nan
    dn = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / dn) if dn > 0 else np.nan


# ------------------------------------------------------------------ batch build / checks
def rows(p):
    return [l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8") if "\t" in l]


def clip_of(p):
    return p.rsplit("/", 1)[1][:-4]


def work(job):
    clip, alldet, apdir, outpose, tv2dir = job
    tv2 = load_tv2(tv2dir)
    d = dict(np.load(os.path.join(alldet, clip + ".npz")))
    pa = os.path.join(apdir, clip + ".npz")
    ap = np.load(pa)["KPS"] if os.path.exists(pa) else None
    g = build_g(d, ap, tv2)
    T = len(d["NDET"])
    c0 = np.zeros((T, 133, 3), np.float32)
    for t in range(T):
        if d["NDET"][t] > 0:
            c0[t, :, :2] = d["KP_ALL"][t, 0]; c0[t, :, 2] = d["SC_ALL"][t, 0]
    v2k = g["v2"]
    same = lambda a, b: int((~np.isclose(a, b).all(axis=(1, 2))).sum())
    for sub, arrs in (("g", dict(KPS=g["KPS"], BBOX=np.zeros((T, 4), np.float32))),
                      ("g_source", dict(source=g["source"], chosen=g["chosen"]))):
        p = os.path.join(outpose, sub, clip + ".npz")
        (np.savez_compressed if sub == "g" else np.savez_compressed)(p[:-4] + ".writing", **arrs)
        os.replace(p[:-4] + ".writing.npz", p)
    info = g["info"]
    st = dict(clip=clip, T=T, reason=info["reason"], fallback=int(info["reason"] != ""), g_masked=int((g["chosen"] < 0).sum()),
              c0_ne_g=same(c0, g["KPS"]), v2_ne_g=same(v2k, g["KPS"]), c0_ne_v2=same(c0, v2k),
              src={k: int((g["source"] == v).sum()) for k, v in SRC.items()},
              **{k: info[k] for k in ("s", "tx", "ty", "n_fit", "n_inlier", "med_inlier", "T_ap") if k in info})
    return st


def build(a):
    split_rows = {s: rows(os.path.join(a.w9man, f"c0_{s}.tsv")) for s in ("train", "val", "test")}
    clips = [clip_of(r[0]) for s in ("train", "val", "test") for r in split_rows[s]]
    if a.only:
        clips = clips[: a.only]
    for sub in ("g", "g_source"):
        os.makedirs(os.path.join(a.outpose, sub), exist_ok=True)
    os.makedirs(a.outman, exist_ok=True)
    miss = [c for c in clips if not os.path.exists(os.path.join(a.alldet, c + ".npz"))]
    assert not miss, f"{len(miss)} clips missing from the all-detections extraction, e.g. {miss[:5]}"
    with Pool(a.workers) as pool:
        stats = pool.map(work, [(c, a.alldet, a.ap, a.outpose, a.tv2) for c in clips], chunksize=20)
    json.dump(stats, open(os.path.join(a.outman, "arm_stats_g.json"), "w"))
    F = sum(x["T"] for x in stats)
    from collections import Counter
    rc = Counter(x["reason"] for x in stats)
    print(f"G built for {len(stats)} clips, {F} frames | G != C0 on {sum(x['c0_ne_g'] for x in stats) / F:.2%} of frames, "
          f"G != V2 on {sum(x['v2_ne_g'] for x in stats) / F:.2%} | G masks {sum(x['g_masked'] for x in stats) / F:.2%} | "
          f"clip fallbacks {sum(x['fallback'] for x in stats)} ({dict(rc)})")
    if not a.only:
        write_manifests(a, split_rows)


def write_manifests(a, split_rows):
    out = os.path.join(a.outpose, "g")
    for s in ("train", "val", "test"):
        with open(os.path.join(a.outman, f"g_{s}.tsv"), "w", encoding="utf-8") as f:
            for p, t in split_rows[s]:
                f.write(os.path.join(out, p.rsplit("/", 1)[1]) + "\t" + t + "\n")
    joint = rows(os.path.join(a.w9man, "joint_c0.tsv"))
    ntr = len(split_rows["train"])
    with open(os.path.join(a.outman, "joint_g.tsv"), "w", encoding="utf-8") as f:
        for p, t in joint[:ntr]:
            f.write(os.path.join(out, p.rsplit("/", 1)[1]) + "\t" + t + "\n")
        for p, t in joint[ntr:]:
            f.write(p + "\t" + t + "\n")


def check(a, n_expect=(12439, 800, 800, 26199), sample=200):
    """Gates of prep (pre-registration section 4). Raises AssertionError on any failure."""
    tv2 = load_tv2(a.tv2)
    ntr, nva, nte, nj = n_expect
    R = {n: rows(os.path.join(a.outman if n.startswith("g") else a.w9man, f"{n}.tsv"))
         for n in ("g_train", "g_val", "g_test", "joint_g", "c0_train", "c0_val", "c0_test", "joint_c0", "v2_val", "v2_test", "joint_v2")}
    assert [len(R[n]) for n in ("g_train", "g_val", "g_test", "joint_g")] == [ntr, nva, nte, nj], "G manifest row counts"
    nm = lambda r: [(clip_of(p), t) for p, t in r]
    for g_, c_ in (("g_train", "c0_train"), ("g_val", "c0_val"), ("g_test", "c0_test"), ("g_val", "v2_val"), ("g_test", "v2_test")):
        assert nm(R[g_]) == nm(R[c_]), f"{g_} differs from {c_} in clip order or English text"
    assert nm(R["joint_g"]) == nm(R["joint_c0"]) == nm(R["joint_v2"]) and [t for _, t in R["joint_g"]] == [t for _, t in R["joint_c0"]]
    assert R["joint_g"][ntr:] == R["joint_c0"][ntr:], "synthetic half is not verbatim"
    out = os.path.join(a.outpose, "g")
    real = R["g_train"] + R["g_val"] + R["g_test"]
    assert all(p == os.path.join(out, clip_of(p) + ".npz") for p, _ in real + R["joint_g"][:ntr]), "a G row does not read the G pose folder"
    assert all(R["joint_g"][i][0] == os.path.join(out, clip_of(R["joint_g"][i][0]) + ".npz") for i in range(ntr))
    clips = [clip_of(p) for p, _ in real]
    assert len(set(clips)) == len(clips) == ntr + nva + nte
    files = sorted(f for f in os.listdir(out) if f.endswith(".npz"))
    assert len(files) == len(clips) and set(f[:-4] for f in files) == set(clips), f"G folder holds {len(files)} files, expected {len(clips)}"
    stats = {x["clip"]: x for x in json.load(open(os.path.join(a.outman, "arm_stats_g.json")))}
    assert set(stats) == set(clips)
    c0dir = os.path.join(a.w9pose, "c0"); v2dir = os.path.join(a.w9pose, "v2")
    nfb = nfb_eq = 0
    for c in clips:
        z = np.load(os.path.join(out, c + ".npz")); assert sorted(z.files) == ["BBOX", "KPS"], (c, z.files)
        k, b = z["KPS"], z["BBOX"]
        assert k.dtype == np.float32 and b.dtype == np.float32 and k.shape == (stats[c]["T"], 133, 3) and b.shape == (len(k), 4) and not b.any(), c
    rng = np.random.RandomState(0)
    pick = [clips[i] for i in rng.choice(len(clips), min(sample, len(clips)), replace=False)]
    for c in pick:
        d = dict(np.load(os.path.join(a.alldet, c + ".npz")))
        T = len(d["NDET"]); c0 = np.zeros((T, 133, 3), np.float32)
        for t in range(T):
            if d["NDET"][t] > 0:
                c0[t, :, :2] = d["KP_ALL"][t, 0]; c0[t, :, 2] = d["SC_ALL"][t, 0]
        assert np.array_equal(c0, np.load(os.path.join(c0dir, c + ".npz"))["KPS"]), f"C0 recomputed from the extraction differs from the wave-9 file: {c}"
        assert np.array_equal(v2_arm(d, tv2)[0], np.load(os.path.join(v2dir, c + ".npz"))["KPS"]), f"V2 recomputed differs from the wave-9 file: {c}"
    for c in clips:
        if stats[c]["fallback"]:
            nfb += 1
            nfb_eq += int(np.array_equal(np.load(os.path.join(out, c + ".npz"))["KPS"], np.load(os.path.join(v2dir, c + ".npz"))["KPS"]))
    assert nfb == nfb_eq, f"{nfb - nfb_eq} fallback clips differ from the wave-9 V2 file"
    for c in pick[:50]:
        assert np.load(os.path.join(c0dir, c + ".npz"))["KPS"].shape == np.load(os.path.join(out, c + ".npz"))["KPS"].shape
    print(f"CHECK OK: {len(clips)} clips; G manifests {ntr}/{nva}/{nte}/{nj} rows, clip order and English text identical to the wave-9 C0 and V2 manifests, "
          f"synthetic half verbatim; every G file KPS (T,133,3) float32 + zero BBOX; C0 and V2 recomputed from the extraction equal the wave-9 files on {len(pick)} clips; "
          f"{nfb} fallback clips byte-equal to the wave-9 V2 file")


def _mr(job):
    c, alldet, apdir, w9pose, outpose = job
    ap = np.load(os.path.join(apdir, c + ".npz"))["KPS"]; sa = sig(ap)
    d = np.load(os.path.join(alldet, c + ".npz"))
    out = dict(clip=c)
    out["idx0"] = corr(sa, sig(np.load(os.path.join(w9pose, "c0", c + ".npz"))["KPS"]))
    out["v1"] = corr(sa, sig(d["KPS"]))
    out["v2"] = corr(sa, sig(np.load(os.path.join(w9pose, "v2", c + ".npz"))["KPS"]))
    out["g"] = corr(sa, sig(np.load(os.path.join(outpose, "g", c + ".npz"))["KPS"]))
    return out


def matchrate(a):
    multi = [c for c, k in json.load(open(a.npeople)).items() if len(k) > 1 and os.path.exists(os.path.join(a.alldet, c + ".npz"))
             and os.path.exists(os.path.join(a.ap, c + ".npz")) and os.path.exists(os.path.join(a.outpose, "g", c + ".npz"))]
    with Pool(a.workers) as pool:
        res = pool.map(_mr, [(c, a.alldet, a.ap, a.w9pose, a.outpose) for c in sorted(multi)], chunksize=20)
    share = {k: float(np.mean([r[k] > 0.5 for r in res])) for k in ("idx0", "v1", "v2", "g")}
    med = {k: float(np.nanmedian([r[k] for r in res])) for k in share}
    print(f"MATCH RATE on {len(res)} multi-person clips, share with r > 0.5 against the AlphaPose signer: " +
          " | ".join(f"{k} {100 * share[k]:.2f}% (median r {med[k]:.3f})" for k in share))
    json.dump(dict(n=len(res), share=share, median_r=med), open(os.path.join(a.outpose, "matchrate_g.json"), "w"))
    assert share["g"] >= share["v1"] + a.min_gain, f"G {share['g']:.4f} is not at least {a.min_gain} above v1 {share['v1']:.4f}"
    assert share["g"] >= share["v2"], f"G {share['g']:.4f} is below V2 {share['v2']:.4f}"
    print("MATCH-RATE GATE OK")


# ------------------------------------------------------------------ self-test
def _synthetic(T=60, flip=range(20, 30), occl=range(40, 46), noap=range(10, 14), seed=0):
    """Two people. Index 0 = a bystander (static, near), index 1 = the signer (moving arms); in `flip` frames the detector order is swapped.
    The signer is undetected in `occl`; AlphaPose has no signer in `noap`. AlphaPose coordinates = DWPose coordinates / 1.03 + offset."""
    rng = np.random.RandomState(seed)
    KA = np.zeros((T, 2, 133, 2), np.float32); SA = np.full((T, 2, 133), 0.9, np.float32); DB = np.zeros((T, 2, 4), np.float32); ND = np.full(T, 2, np.int16)
    sc, off = 1.03, np.array([-500.0, 120.0])
    ap = np.zeros((T, 136, 3), np.float32)

    def person(cx, cy, w, ph):
        k = np.zeros((133, 2), np.float32); k[:] = [cx, cy + 300]
        k[0] = [cx, cy - 60]; k[5] = [cx - w / 2, cy]; k[6] = [cx + w / 2, cy]; k[11] = [cx - w / 3, cy + 250]; k[12] = [cx + w / 3, cy + 250]
        k[7] = [cx - w, cy + 120 + 40 * np.sin(ph)]; k[8] = [cx + w, cy + 120 + 40 * np.cos(ph)]
        k[9] = [cx - w * 1.2, cy + 250 + 80 * np.sin(ph * 1.3)]; k[10] = [cx + w * 1.2, cy + 250 + 80 * np.cos(ph * 1.3)]
        return k
    for t in range(T):
        sg, bs = person(300, 300, 90, t * 0.4), person(520, 330, 80, 0.0)
        bs[7:11] += rng.normal(0, 1, (4, 2))
        order = [bs, sg] if t not in flip else [sg, bs]
        boxes = [[480, 250, 570, 700], [240, 200, 380, 700]]
        if t in flip:
            boxes = boxes[::-1]
        dets = list(zip(order, boxes))
        if t in occl:
            dets = [x for x in dets if not np.allclose(x[0][5], sg[5])]
        for i, (k, b) in enumerate(dets):
            KA[t, i] = k; DB[t, i] = b
        ND[t] = len(dets)
        if t not in noap:
            ap[t, :17, :2] = (sg[:17] - off) / sc; ap[t, :17, 2] = 0.95
    return dict(KP_ALL=KA, SC_ALL=SA, DET_BOX=DB, NDET=ND, WH=np.array([700, 800]), KPS=np.zeros((T, 133, 3), np.float32),
                CHOSEN=np.zeros(T, np.int16)), ap, (sc, off)


def selftest(tv2dir=None):
    tv2 = load_tv2(tv2dir)
    d, ap, (sc, off) = _synthetic()
    # make the stored v1 path/ index-0 = the bystander so a naive arm fails
    g = build_g(d, ap, tv2); info = g["info"]
    ch, src = g["chosen"], g["source"]
    assert info["reason"] == "", info
    assert abs(info["s"] - sc) < 0.02 and abs(info["tx"] - off[0]) < 15 and abs(info["ty"] - off[1]) < 15, info
    T = len(ch)
    sg_idx = [(1 if t not in range(20, 30) else 0) for t in range(T)]
    evid_ok = [t for t in range(T) if t not in range(10, 14) and t not in range(40, 46)]
    assert all(ch[t] == sg_idx[t] for t in evid_ok), [(t, ch[t], sg_idx[t]) for t in evid_ok if ch[t] != sg_idx[t]]
    assert all(src[t] == SRC["ap"] for t in evid_ok), src
    assert all(ch[t] == -1 and src[t] == SRC["ap_mask"] for t in range(40, 46)), (ch[40:46], src[40:46])   # signer not detected: masked, never the bystander
    assert all(ch[t] == sg_idx[t] for t in range(10, 14)), ch[10:14]                                             # no AlphaPose: identity carried through the gap
    assert (src[10:14] != SRC["ap"]).all()
    # the naive detection-0 arm is wrong on the non-flip frames: G differs from it there
    assert (g["KPS"][0, :, :2] == d["KP_ALL"][0, 1]).all() and not (g["KPS"][0, :, :2] == d["KP_ALL"][0, 0]).all()
    # a one-frame bystander flip needs evidence: a frame where the bystander is momentarily the closer candidate by a small margin must not flip
    d2, ap2, _ = _synthetic(flip=[], occl=[], noap=[])
    d2["KP_ALL"][30, 0, :, 0] = d2["KP_ALL"][30, 1, :, 0] - 20           # det 0 (bystander) slides next to the signer for one frame
    d2["DET_BOX"][30, 0] = d2["DET_BOX"][30, 1] + [60, 0, 60, 0]
    g2 = build_g(d2, ap2, tv2)
    assert g2["info"]["reason"] == "" and (g2["chosen"] == 1).all(), g2["chosen"]
    # fallbacks
    f = build_g(d, None, tv2); assert f["info"]["reason"] == "no_ap" and (f["source"] == SRC["v2_fallback"]).all()
    f = build_g(d, ap[:-5], tv2); assert f["info"]["reason"] == "length_mismatch"
    apn = ap.copy(); apn[:, :, 2] = 0.0
    assert build_g(d, apn, tv2)["info"]["reason"] == "no_ap"
    aps = ap.copy(); aps[:, :17, 0] += np.random.RandomState(1).uniform(-400, 400, (len(ap), 1))     # unrelated track: no consistent transform
    aps[:, :17, 1] += np.random.RandomState(2).uniform(-400, 400, (len(ap), 1))
    fr = build_g(d, aps, tv2); assert fr["info"]["reason"] == "fit_unreliable" and (fr["source"] == SRC["v2_fallback"]).all(), fr["info"]
    # a clip whose fit is unreliable returns exactly the V2 arm
    assert np.array_equal(fr["KPS"], v2_arm(d, tv2)[0])
    # the signature measure is identical to signer_check.py's
    sck = Path(__file__).resolve().parents[2] / "results" / "dwpose_signer_tracking_check" / "signer_check.py"
    if sck.exists():
        spec = importlib.util.spec_from_file_location("signer_check", sck); sc_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc_mod)
        k = np.random.RandomState(3).rand(40, 17, 3).astype(np.float32); k[:, 5:7, 2] = 0.9
        assert np.allclose(sig(k), sc_mod.sig(k)[0]) and corr(sig(k), sig(k[::-1])) == sc_mod.corr(sc_mod.sig(k)[0], sc_mod.sig(k[::-1])[0])
    print("selftest OK: transform recovered (s %.3f, t (%.1f, %.1f)); signer followed through a detector-order swap; occlusion masked, "
          "no-AlphaPose gap carried, one-frame bystander flip refused; four fallbacks return the V2 arm" % (info["s"], info["tx"], info["ty"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "build", "check", "matchrate"))
    for k in ("alldet", "ap", "w9man", "w9pose", "outpose", "outman", "npeople", "tv2"):
        ap.add_argument("--" + k, default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--min_gain", type=float, default=0.01); ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest(a.tv2)
    elif a.mode == "build":
        build(a)
    elif a.mode == "check":
        check(a, sample=a.sample)
    else:
        matchrate(a)


if __name__ == "__main__":
    main()
