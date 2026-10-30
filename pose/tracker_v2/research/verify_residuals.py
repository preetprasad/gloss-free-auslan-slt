"""Adversarial re-check of the emp-v2-residuals lens (independent code; imports only the tracker under test).

Recomputes from the raw all-detections npz + gt_key/gt_labels: the DEV/held-out outcome counts, why v2-tuned masks,
masked-gap lengths, hand quality with clip-clustered CIs, DEV-only v2.1 variants, and the DEV/held-out video overlap.
Held-out is DESCRIBED only; no variant is scored on it.  Run: python verify_residuals.py
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tracker_v2 as v2

NPZ = Path("/private/tmp/claude-501/-Users-preet-Honours/c69cfc0c-6e97-4020-b91d-72bcf24d0e82/scratchpad/gt/npz")
MP4 = Path("./results/dwpose_experiment/rgb_clips/Signer")
KEY = json.load(open(HERE.parent / "gt_key.json"))
LAB = json.load(open(HERE.parent / "gt_labels.json"))
LH, RH = slice(91, 112), slice(112, 133)
BASE = dict(CONF=v2.CONF, MIN_KP=v2.MIN_KP, DUP_IOU=v2.DUP_IOU, LINK_IOU=v2.LINK_IOU, MAX_GAP=v2.MAX_GAP,
            EXT_IOU=v2.EXT_IOU)


def box_iou(a, b):  # own IoU, not v2.iou
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0


def outcome(ch, truth, boxes):
    if truth is None:
        return "OKMASK" if ch < 0 else "WRONG"
    if ch < 0:
        return "MISS"
    return "HIT" if ch == truth or box_iou(boxes[ch], boxes[truth]) >= 0.6 else "WRONG"


def runs(mask):
    """(start, end_inclusive) of True runs."""
    out, t = [], 0
    while t < len(mask):
        if mask[t]:
            s = t
            while t < len(mask) and mask[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


def fullframe(d):
    """(T,N) bool: detection box is the rtmlib no-person fallback [0,0,W,H] (within 1 px)."""
    W, H = (float(x) for x in d["WH"]); DB = d["DET_BOX"]
    ff = (np.abs(DB[..., 0]) <= 1) & (np.abs(DB[..., 1]) <= 1) & (np.abs(DB[..., 2] - W) <= 1) & (np.abs(DB[..., 3] - H) <= 1)
    ff &= np.arange(DB.shape[1])[None, :] < d["NDET"][:, None]
    return ff


def run_v2(d, **over):
    for k, v in {**BASE, **over}.items():
        setattr(v2, k, v)
    try:
        return v2.choose(d, v2.TUNED)[1].astype(int)
    finally:
        for k, v in BASE.items():
            setattr(v2, k, v)


def cluster_boot(groups, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed); ks = list(groups); vals = []
    for _ in range(n):
        pick = [groups[ks[i]] for i in rng.integers(0, len(ks), len(ks))]
        v = stat(pick)
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def main():
    P = print
    # ---- data
    panels = defaultdict(list); part = {}
    for pid, k in KEY.items():
        lab = LAB[pid]; assert lab != "unclear"
        panels[k["clip"]].append((k["frame"], None if lab == "none" else k["letters"][lab]))
        part[k["clip"]] = k["part"]
    data = {c: dict(np.load(NPZ / f"{c}.npz")) for c in panels}
    assert len(data) == 60 and Counter(part.values()) == {"dev": 30, "heldout": 30}
    chos = {c: run_v2(d) for c, d in data.items()}

    # ---- 0. reproduce recorded scores with an independent outcome function
    cnt = {p: Counter() for p in ("dev", "heldout")}; percl = defaultdict(Counter)
    for c, pl in panels.items():
        for f, tr in pl:
            o = outcome(chos[c][f], tr, data[c]["DET_BOX"][f]); cnt[part[c]][o] += 1; percl[c][o] += 1
    for p, want in (("dev", (277, 5, 0, 4)), ("heldout", (270, 2, 17, 3))):
        got = tuple(cnt[p][x] for x in ("HIT", "WRONG", "MISS", "OKMASK"))
        assert got == want, (p, got); P(f"[0] {p} v2-tuned HIT/WRONG/MISS/OKMASK = {got}  (reproduces record)")

    # ---- leakage: video overlap between DEV and held-out
    vid = lambda c: re.match(r"video_(\d+)_", c).group(1)
    dv = {vid(c) for c in part if part[c] == "dev"}; hv = {vid(c) for c in part if part[c] == "heldout"}
    shared = sorted(dv & hv, key=int)
    P(f"\n[leak] videos in both DEV and held-out: {len(shared)} {shared}; held-out clips in those videos: "
      f"{sum(vid(c) in dv for c in part if part[c] == 'heldout')}/30; held-out MISS/WRONG clips: "
      f"{[c for c in percl if part[c] == 'heldout' and (percl[c]['MISS'] or percl[c]['WRONG'])]}")

    # ---- 1. DEV masking and held-out misses
    P("\n[1] DEV masked frames")
    for c in sorted(c for c in data if part[c] == "dev"):
        d, ch = data[c], chos[c]; ff = fullframe(d)
        for s, e in runs(ch < 0):
            labs = [("none" if tr is None else "signer") for f, tr in panels[c] if s <= f <= e]
            det = []
            for t in range(s, e + 1):
                for i in range(int(d["NDET"][t])):
                    det.append((t, i, int((d["SC_ALL"][t, i, :23] > 0.3).sum()), bool(ff[t, i])))
            P(f"  {c} masked f{s}-{e} ({e - s + 1} fr) labels {Counter(labs)} | dets (t,i,body>0.3,fullframe): {det[:8]}"
              f"{' ...' if len(det) > 8 else ''}")
    P("\n[1] DEV WRONG panels: chosen box vs signer box, full-frame flag")
    for c in sorted(c for c in data if part[c] == "dev"):
        d = data[c]
        for f, tr in panels[c]:
            ch = chos[c][f]
            if outcome(ch, tr, d["DET_BOX"][f]) != "WRONG":
                continue
            cb = d["DET_BOX"][f, ch]
            if tr is None:
                P(f"  {c} f{f}: signer not detected; chosen {cb.astype(int).tolist()} WH {d['WH'].tolist()} fullframe "
                  f"{bool(fullframe(d)[f, ch])}")
            else:
                sb = d["DET_BOX"][f, tr]; ia = (min(cb[2], sb[2]) - max(cb[0], sb[0])) * (min(cb[3], sb[3]) - max(cb[1], sb[1]))
                P(f"  {c} f{f}: IoU {box_iou(cb, sb):.2f} area ratio {((cb[2]-cb[0])*(cb[3]-cb[1]))/((sb[2]-sb[0])*(sb[3]-sb[1])):.2f} "
                  f"signer-inside {max(ia, 0)/((sb[2]-sb[0])*(sb[3]-sb[1])):.2f}")

    P("\n[1] held-out MISS panels (description only): does the signer detection pass v2's filter? which tracklet?")
    for c in sorted(c for c in data if part[c] == "heldout" and percl[c]["MISS"]):
        d = data[c]; T = len(d["NDET"])
        cands = [v2.candidates(d["SC_ALL"][t], d["DET_BOX"][t], int(d["NDET"][t])) for t in range(T)]
        trs = v2.tracklets(d["DET_BOX"], cands)
        W, H = (int(x) for x in d["WH"])
        sc = [v2.score(tr, d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], W, H, T, v2.TUNED) for tr in trs]
        seed = int(np.argmax(sc)); st = trs[seed]
        P(f"  {c} T={T}: seed tracklet len {len(st)} f{min(st)}-{max(st)} area-score {sc[seed]:.3f}; masked "
          f"{int((chos[c] < 0).sum())}/{T}; runs {runs(chos[c] < 0)}")
        for f, tr in panels[c]:
            if outcome(chos[c][f], tr, d["DET_BOX"][f]) != "MISS":
                continue
            inc = [i for i in cands[f] if i == tr or box_iou(d["DET_BOX"][f, i], d["DET_BOX"][f, tr]) >= 0.6]
            k = next((k for k, t_ in enumerate(trs) if t_.get(f) in inc), None)
            nb = int((d["SC_ALL"][f, tr, :23] > 0.3).sum())
            info = "not a candidate (filtered)" if not inc else (
                f"in tracklet len {len(trs[k])} f{min(trs[k])}-{max(trs[k])} area {sc[k]:.3f}" if k is not None else "?")
            P(f"    MISS f{f}: signer body kps>0.3 {nb}/23 -> {info}")

    # ---- full-frame fallback detections: prevalence and where v2 picks them
    P("\n[ff] rtmlib whole-frame fallback boxes [0,0,W,H]")
    for p in ("dev", "heldout"):
        nff = sum(int(fullframe(data[c]).sum()) for c in data if part[c] == p)
        pick = [(c, t) for c in data if part[c] == p for t in range(len(chos[c]))
                if chos[c][t] >= 0 and fullframe(data[c])[t, chos[c][t]]]
        lab = Counter(outcome(chos[c][f], tr, data[c]["DET_BOX"][f]) for c, t in pick for f, tr in panels[c] if f == t)
        P(f"  {p}: full-frame detections {nff}; frames where v2-tuned CHOSE one {len(pick)} {pick[:10]}; "
          f"labelled outcomes on those {dict(lab)}")

    # ---- 2. masked-gap lengths, all 60 clips
    tot, allm, rows = 0, 0, []
    for c, ch in chos.items():
        tot += len(ch); m = ch < 0; allm += int(m.sum())
        for s, e in runs(m):
            L = e - s + 1; rows.append((L, s == 0 or e == len(ch) - 1, c, s, e))
    assert sum(r[0] for r in rows) == allm
    share = lambda f: sum(r[0] for r in rows if f(r)) / allm
    P(f"\n[2] masked {allm}/{tot} = {allm / tot:.2%}; clips fully masked {sum((ch < 0).all() for ch in chos.values())}; "
      f"gap hist {dict(sorted(Counter(r[0] for r in rows).items()))}")
    P(f"  share in gaps <=2 {share(lambda r: r[0] <= 2):.1%}, <=5 {share(lambda r: r[0] <= 5):.1%}, interior <=5 "
      f"{share(lambda r: r[0] <= 5 and not r[1]):.1%} ({sum(r[0] for r in rows if r[0] <= 5 and not r[1])} fr), "
      f">15 {share(lambda r: r[0] > 15):.1%}, >15 at edge {sum(r[0] for r in rows if r[0] > 15 and r[1])} fr")
    cat = Counter()
    for L, edge, c, s, e in rows:
        labs = [tr for f, tr in panels[c] if s <= f <= e]
        cat["unlabelled" if not labs else ("signer-present" if any(t is not None for t in labs) else "signer-absent")] += L
    P(f"  masked frames by labels inside their run: {dict(cat)} (share signer-present {cat['signer-present'] / allm:.1%})")
    # per-panel version (no extrapolation to unlabelled frames)
    mp = Counter()
    for c in data:
        for f, tr in panels[c]:
            if chos[c][f] < 0:
                mp["signer-present" if tr is not None else "signer-absent"] += 1
    P(f"  masked LABELLED panels only: {dict(mp)}")

    # ---- 3. hand quality on chosen frames (own definitions: per-frame-pair shoulder width)
    per = defaultdict(lambda: dict(n=0, lo=0, tr=0, jump=0, lojump=0, lotr=0, hijump=0, hitr=0, lowiou=0))
    confL, confR = [], []
    for c, d in data.items():
        ch = chos[c]; g = per[c]
        for t in range(len(ch)):
            if ch[t] < 0:
                continue
            s = d["SC_ALL"][t, ch[t]]; confL.append(s[LH].mean()); confR.append(s[RH].mean()); g["n"] += 1
            if t + 1 >= len(ch) or ch[t + 1] < 0:
                continue
            k0, k1 = d["KP_ALL"][t, ch[t]], d["KP_ALL"][t + 1, ch[t + 1]]; s1 = d["SC_ALL"][t + 1, ch[t + 1]]
            sw = np.mean([np.linalg.norm(k0[5] - k0[6]), np.linalg.norm(k1[5] - k1[6])])
            if not (s[5] > 0.3 and s[6] > 0.3 and s1[5] > 0.3 and s1[6] > 0.3) or sw <= 0:
                continue
            j = max(np.linalg.norm(k1[h].mean(0) - k0[h].mean(0)) / sw for h in (LH, RH)) > 0.5
            low = min(s[LH].mean(), s[RH].mean(), s1[LH].mean(), s1[RH].mean()) < 0.3
            g["tr"] += 1; g["jump"] += j
            g["lowiou"] += box_iou(d["DET_BOX"][t, ch[t]], d["DET_BOX"][t + 1, ch[t + 1]]) < 0.3
            if low: g["lotr"] += 1; g["lojump"] += j
            else: g["hitr"] += 1; g["hijump"] += j
    confL, confR = np.array(confL), np.array(confR)
    S = lambda gs, a, b: sum(x[a] for x in gs) / max(1, sum(x[b] for x in gs))
    gs = list(per.values())
    P(f"\n[3] frames on v2 path {len(confL)}; L<0.3 {np.mean(confL < .3):.1%} R<0.3 {np.mean(confR < .3):.1%} either "
      f"{np.mean((confL < .3) | (confR < .3)):.1%}")
    P(f"  transitions (both shoulders conf>0.3) {sum(x['tr'] for x in gs)}; jump>0.5 SW {S(gs, 'jump', 'tr'):.2%} "
      f"CI(clip boot) {cluster_boot(per, lambda p: S(p, 'jump', 'tr'))}; box IoU<0.3 across consecutive path frames "
      f"{sum(x['lowiou'] for x in gs)}")
    diff = lambda p: S(p, "lojump", "lotr") - S(p, "hijump", "hitr") if sum(x["lotr"] for x in p) else np.nan
    P(f"  jump rate low-conf {S(gs, 'lojump', 'lotr'):.1%} (n={sum(x['lotr'] for x in gs)} in "
      f"{sum(x['lotr'] > 0 for x in gs)} clips) vs ok {S(gs, 'hijump', 'hitr'):.1%} (n={sum(x['hitr'] for x in gs)}); "
      f"difference CI (clip boot) {cluster_boot(per, diff)}")
    lo_by_clip = sorted(((x["lotr"], c) for c, x in per.items() if x["lotr"]), reverse=True)[:5]
    P(f"  low-conf transitions concentrated in: {lo_by_clip}")

    # ---- 4. DEV-only variants
    P("\n[4] DEV-only variants: HIT/WRONG/MISS/OKMASK | masked DEV frames  (held-out NOT scored)")
    def dev_row(fn):
        cc, mk = Counter(), 0
        for c, d in data.items():
            if part[c] != "dev":
                continue
            ch = fn(d); mk += int((ch < 0).sum())
            for f, tr in panels[c]:
                cc[outcome(ch[f], tr, d["DET_BOX"][f])] += 1
        return tuple(cc[x] for x in ("HIT", "WRONG", "MISS", "OKMASK")), mk
    def drop_ff(d):
        e = dict(d); e["SC_ALL"] = d["SC_ALL"].copy(); e["SC_ALL"][fullframe(d)] = 0.0
        return run_v2(e)
    variants = {"v2-tuned": lambda d: run_v2(d), "EXT_IOU 0.2": lambda d: run_v2(d, EXT_IOU=0.2),
                "LINK_IOU 0.2": lambda d: run_v2(d, LINK_IOU=0.2), "MAX_GAP 10": lambda d: run_v2(d, MAX_GAP=10),
                "MIN_KP 3": lambda d: run_v2(d, MIN_KP=3), "drop full-frame fallback boxes": drop_ff}
    res = {n: dev_row(fn) for n, fn in variants.items()}
    for n, (row, mk) in res.items():
        P(f"  {n:32s} {row} | {mk}")
    assert res["v2-tuned"][0] == (277, 5, 0, 4)

    # ---- seed rules on DEV (monkeypatched score; also reorders join candidates -- an approximation of the lens's rules)
    orig = v2.score
    def seed_rule(fn):
        v2.score = lambda tr, KA, SA, DB, W, H, T, w=None: fn(v2.features(tr, KA, SA, DB, W, H, T))
        try:
            return dev_row(lambda d: run_v2(d))
        finally:
            v2.score = orig
    P(f"  seed cov*area                     {seed_rule(lambda f: f['cov'] * f['area'])}")
    P(f"  seed cov>=0.25 else area-10       {seed_rule(lambda f: f['area'] - (0 if f['cov'] >= 0.25 else 10))}")
    d = data["video_78_68"]; T = len(d["NDET"])
    trs = v2.tracklets(d["DET_BOX"], [v2.candidates(d["SC_ALL"][t], d["DET_BOX"][t], int(d["NDET"][t])) for t in range(T)])
    W, H = (int(x) for x in d["WH"])
    sc = [v2.score(tr, d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], W, H, T, v2.TUNED) for tr in trs]; s = trs[int(np.argmax(sc))]
    P(f"  [describe] video_78_68 seed len {len(s)} f{min(s)}-{max(s)} area {max(sc):.3f}; outcomes {dict(percl['video_78_68'])}")

    # ---- cuts (pixels) at the claimed boundaries: mean abs diff of 96x96 grey frames
    try:
        import cv2
        P("\n[cut] 96x96 grey mean-abs-diff at claimed cuts (median over the clip in brackets)")
        for c, t in (("video_39_13", 82), ("video_80_66", 4), ("video_24_137", 40), ("video_26_116", 34),
                     ("video_78_68", 53), ("video_39_14", None)):
            cap = cv2.VideoCapture(str(MP4 / f"{c}_signer.mp4")); fr = []
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                fr.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (96, 96)).astype(np.float32))
            dfs = np.array([np.abs(fr[i + 1] - fr[i]).mean() for i in range(len(fr) - 1)])
            T = len(data[c]["NDET"]) if c in data else None
            big = [(i, round(float(dfs[i]), 1)) for i in np.flatnonzero(dfs > 30)]
            P(f"  {c}: video frames {len(fr)} vs npz T {T}; diff after f{t}: "
              f"{round(float(dfs[t]), 1) if t is not None else '-'} [{np.median(dfs):.1f}]; transitions >30: {big}")
    except ImportError:
        P("cv2 unavailable: cut check skipped")


def _selftest():
    assert runs(np.array([1, 1, 0, 1, 0, 0, 1], bool)) == [(0, 1), (3, 3), (6, 6)]
    assert outcome(-1, None, None) == "OKMASK" and outcome(-1, 0, None) == "MISS"
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    assert outcome(1, 0, b) == "HIT" and outcome(2, 0, b) == "WRONG" and outcome(0, None, b) == "WRONG"
    d = dict(DET_BOX=np.array([[[0, 0, 100, 200], [5, 5, 50, 50]]], float), WH=np.array([100, 200]), NDET=np.array([2]))
    assert fullframe(d).tolist() == [[True, False]]


if __name__ == "__main__":
    _selftest(); main()
