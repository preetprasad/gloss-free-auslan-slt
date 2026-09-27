"""Is the dataset's own signer track (the shipped Auslan-Daily AlphaPose KPS) on the signer? (literature lens, 2026-09-25)

Why: the Auslan-Daily paper (NeurIPS 2023 D&B, Sec. 1 / Sec. 3.2, local Auslan-Daily.txt lines 213-218) says the authors ran
AlphaPose "a tool for keypoints estimation and person tracking" on every clip, then "manually modify the error of the
Alphapose results and labelled the signer". The shipped per-clip KPS (full_run/clips/<clip>.npz, HALPE-136, full-frame
pixels) is therefore a human-corrected signer trajectory -- which the DWPose re-extraction (detection 0 on the signer
crop) discarded. This script measures how often that shipped trajectory is on the signer, on the SAME 60 ground-truth
clips / panels that scored index-0, v1 and v2 (PREREG_TRACKER_V2 §2-3), and on every v1 override frame in val+test.

Mapping (no identity assumed): the signer crop is a fixed region of the frame resized to WHxH, so crop = (sx*x+tx, sy*y+ty)
for the whole clip. Candidate transforms are least-squares fits of the shipped body keypoints (COCO 0-12) to EACH detection
in EACH frame; the clip transform is the candidate under which the most frames have the mapped skeleton on SOME detection
(so it does not matter whom the shipped track follows), refit on those inliers. Each frame's shipped skeleton is then
assigned to the nearest detection (median keypoint distance / that detection's shoulder width < ACCEPT), else -1.

Modes:  gt <ALLDET_DIR>          score vs the two-rater ground truth (gate: reproduces the recorded index0/v1/v2-tuned)
        overrides <TRACKED_DIR>  on v1 override frames (CHOSEN != 0): does the shipped track side with v1 or index-0?
        (no args)                self-test only
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
AP_DIR = ROOT / "full_run/clips"                     # shipped AlphaPose HALPE-136, one npz per clip
BODY = np.arange(13)                                 # COCO body 0-12: same indices in HALPE-136 and COCO-WholeBody-133
CONF, INLIER, ACCEPT = 0.3, 0.3, 0.5


def fit(a, d):
    """a, d: (n,2). Per-axis least squares d = s*a + t. None if degenerate."""
    if len(a) < 4 or a[:, 0].std() < 1 or a[:, 1].std() < 1:
        return None
    p = []
    for k in (0, 1):
        s, t = np.linalg.lstsq(np.c_[a[:, k], np.ones(len(a))], d[:, k], rcond=None)[0]
        if s <= 0:
            return None
        p += [s, t]
    return np.array(p)                               # sx, tx, sy, ty


def apply(p, a):
    return np.c_[p[0] * a[:, 0] + p[1], p[2] * a[:, 1] + p[3]]


def err(p, ap_xy, ap_ok, kp, sc):
    """Median distance of the mapped shipped skeleton to one detection, in that detection's shoulder widths."""
    m = ap_ok & (sc[BODY] > CONF)
    if m.sum() < 4:
        return np.inf
    sw = max(np.linalg.norm(kp[5] - kp[6]), 10.0)
    return float(np.median(np.linalg.norm(apply(p, ap_xy[m]) - kp[BODY][m], axis=1)) / sw)


def assign(AP, KA, SA, ND):
    """AP (T,136,3) shipped; KA (T,D,133,2), SA (T,D,133), ND (T,). Returns choice (T,), inlier fraction, transform."""
    T = len(ND)
    ap_xy, ap_ok = AP[:, BODY, :2], AP[:, BODY, 2] > CONF
    cands = []
    for t in range(T):
        for j in range(int(ND[t])):
            m = ap_ok[t] & (SA[t, j, BODY] > CONF)
            p = fit(ap_xy[t][m], KA[t, j, BODY][m])
            if p is not None:
                cands.append(p)
    if not cands:
        return np.full(T, -1), 0.0, None

    def frame_errs(p):
        return [[err(p, ap_xy[t], ap_ok[t], KA[t, j], SA[t, j]) for j in range(int(ND[t]))] for t in range(T)]

    def support(p):
        return sum(1 for e in frame_errs(p) if e and min(e) < INLIER)
    best = max(cands, key=support)
    E = frame_errs(best)                              # refit on inlier (frame, nearest detection) pairs
    A, D = [], []
    for t, e in enumerate(E):
        if e and min(e) < INLIER:
            j = int(np.argmin(e)); m = ap_ok[t] & (SA[t, j, BODY] > CONF)
            A.append(ap_xy[t][m]); D.append(KA[t, j, BODY][m])
    p = fit(np.vstack(A), np.vstack(D)) if A else None
    p = best if p is None else p
    E = frame_errs(p)
    choice = np.array([int(np.argmin(e)) if e and min(e) < ACCEPT else -1 for e in E])
    usable = sum(1 for t in range(T) if ap_ok[t].sum() >= 4 and ND[t] > 0)
    return choice, sum(1 for e in E if e and min(e) < INLIER) / max(usable, 1), p


def load_ap(clip, T):
    z = np.load(AP_DIR / f"{clip}.npz")["KPS"]
    assert z.shape[1:] == (136, 3), (clip, z.shape)
    return z if len(z) == T else None                    # 9 of 1,600 val+test clips differ in length: alignment unknown, skipped


def mode_gt(alldet):
    import score_gt as S
    key = json.load(open(HERE.parent / "gt_key.json")); labels = json.load(open(HERE.parent / "gt_labels.json"))
    src = Path(alldet)
    # gate: the harness reproduces the recorded held-out and DEV scores before anything new is printed
    rec = {"heldout": {"index0": (224, 68), "v1": (263, 29), "v2-tuned": (270, 2)},
           "dev": {"index0": (213, 73), "v1": (270, 16)}}
    for part, want in rec.items():
        got = S.evaluate({k: S.base_trackers()[k] for k in want}, key, labels, src, part)
        for k, (h, w) in want.items():
            assert (got[k]["HIT"], got[k]["WRONG"]) == (h, w), (part, k, got[k])
    print("gate OK: index0 / v1 / v2-tuned reproduce the recorded DEV and HELD-OUT counts")
    cache, inl = {}, {}

    def shipped(d, clip):
        if clip not in cache:
            ap = load_ap(clip, len(d["NDET"]))
            if ap is None:
                cache[clip] = None; return None
            ch, f, p = assign(ap, d["KP_ALL"], d["SC_ALL"], d["NDET"])
            cache[clip], inl[clip] = ch, f
        return cache[clip]
    out = {}
    for part in ("dev", "heldout"):
        c = Counter(); per_bin = {}
        clips = {k["clip"] for k in key.values() if k["part"] == part}
        bins = {c_: b for c_, b in json.load(open(HERE.parent / "gt_clips.json"))[part]}
        for pid, k in key.items():
            if k["part"] != part:
                continue
            lab = labels[pid]
            truth = None if lab == "none" else k["letters"][lab]
            d = np.load(src / f"{k['clip']}.npz")
            ch = shipped(d, k["clip"])
            if ch is None:
                c["SKIPPED_LENGTH"] += 1; continue
            o = S.outcome(int(ch[k["frame"]]), truth, d["DET_BOX"][k["frame"]])
            o = "UNMATCHED" if o in ("MISS", "OKMASK") else o
            c[o] += 1; per_bin.setdefault(bins[k["clip"]], Counter())[o] += 1
        n = sum(v for k_, v in c.items() if k_ != "SKIPPED_LENGTH")
        print(f"[{part}] skipped (length mismatch) {c['SKIPPED_LENGTH']} panels: {sorted(c_ for c_ in clips if cache.get(c_, 0) is None)}")
        print(f"[{part}] shipped-AlphaPose n={n} HIT {c['HIT']} ({c['HIT']/n:.1%}) WRONG {c['WRONG']} ({c['WRONG']/n:.1%})"
              f" UNMATCHED {c['UNMATCHED']} ({c['UNMATCHED']/n:.1%})")
        for b in ("heavy", "medium", "light"):
            print(f"   {b:6s} {dict(per_bin.get(b, {}))}")
        f = np.array([inl[c_] for c_ in clips if c_ in inl])
        print(f"   transform inlier fraction per clip: median {np.median(f):.2f}, min {f.min():.2f}, "
              f"clips < 0.8: {sorted(c_ for c_ in clips if inl.get(c_, 1) < 0.8)}")
        out[part] = dict(c)
    json.dump({"counts": out, "inlier_fraction": inl}, open(HERE / "shipped_track_vs_gt.json", "w"), indent=1)


def mode_overrides(tracked):
    """v1 override frames in val+test: the shipped track's nearest of {v1 skeleton, index-0 skeleton}."""
    c = Counter(); per_clip_inl = []; n_clips = 0
    for f in sorted(Path(tracked).glob("video_*.npz")):
        tr = np.load(f)
        split = "val" if (ROOT / "results/val_clips_dwpose" / f.name).exists() else "test"
        st = np.load(ROOT / f"results/{split}_clips_dwpose" / f.name)
        ch = tr["CHOSEN"].astype(int)
        if (ch <= 0).all():
            continue
        T = len(ch)
        # two "detections" per frame: 0 = index-0 (stored), 1 = v1 (tracked; identical where CHOSEN == 0)
        KA = np.stack([st["KPS"][..., :2], tr["KPS"][..., :2]], 1); SA = np.stack([st["KPS"][..., 2], tr["KPS"][..., 2]], 1)
        ND = np.where(ch > 0, 2, 1)
        ap = load_ap(f.stem, T)
        if ap is None:
            c["skipped_clips"] += 1; continue
        choice, fr, _ = assign(ap, KA, SA, ND)
        n_clips += 1; per_clip_inl.append(fr)
        for t in np.where(ch > 0)[0]:
            c[{0: "index0", 1: "v1", -1: "neither"}[int(choice[t])]] += 1
    n = c["v1"] + c["index0"] + c["neither"]
    print(f"skipped (length mismatch) clips: {c['skipped_clips']}")
    print(f"override frames {n} in {n_clips} val+test clips: shipped track nearest v1 {c['v1']} ({c['v1']/n:.1%}), "
          f"index-0 {c['index0']} ({c['index0']/n:.1%}), neither {c['neither']} ({c['neither']/n:.1%}); "
          f"transform inlier fraction median {np.median(per_clip_inl):.2f}")
    json.dump(dict(c, clips=n_clips), open(HERE / "shipped_track_overrides.json", "w"), indent=1)


def _self_test():
    rng = np.random.default_rng(0)
    T = 30
    base = rng.uniform(100, 300, (13, 2))
    A = base.copy(); B = base + [400, 20]             # two people in crop coordinates
    p_true = np.array([0.6, -500.0, 0.55, -120.0])    # crop = s * frame + t
    inv = lambda q: np.c_[(q[:, 0] - p_true[1]) / p_true[0], (q[:, 1] - p_true[3]) / p_true[2]]
    KA = np.zeros((T, 2, 133, 2)); SA = np.ones((T, 2, 133))
    KA[:, 0, :13] = A; KA[:, 1, :13] = B
    KA[:, 0, 6] = A[5] + [60, 0]; KA[:, 1, 6] = B[5] + [60, 0]   # shoulder width 60
    AP = np.zeros((T, 136, 3)); AP[..., 2] = 0.9
    truth = np.where(np.arange(T) < 20, 0, 1)         # shipped track on A, switches to B at frame 20
    for t in range(T):
        AP[t, :13, :2] = inv(KA[t, truth[t], :13]) + rng.normal(0, 0.5, (13, 2))
    AP[25, :, 2] = 0                                  # a missing shipped frame
    ch, f, p = assign(AP, KA, SA, np.full(T, 2))
    assert np.allclose(p, p_true, rtol=0.02, atol=3), p
    exp = truth.copy(); exp[25] = -1
    assert (ch == exp).all(), ch
    assert f > 0.95, f
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) > 2 and sys.argv[1] == "gt":
        mode_gt(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "overrides":
        mode_overrides(sys.argv[2])
