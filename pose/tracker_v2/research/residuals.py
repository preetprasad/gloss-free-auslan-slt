"""Residual defects of tracker v2 (TUNED weights) on the 60 ground-truth clips.

Two modes, run in this order (the held-out description runs only after the v2.1 choice is frozen on DEV):
  dev       Q1 why v2 masks frames although the signer was detected (DEV only: labelled panels + an IoU proxy on the
            unlabelled frames, the proxy itself validated on DEV labels); Q2/Q3 on DEV; Q4 v2.1 variants on DEV only.
            Writes V21_FROZEN.json.
  describe  Q2 masked-gap lengths and Q3 hand quality over all 60 clips; categorises the held-out v2-tuned MISS panels
            with the same code. Never evaluates a v2.1 variant on held-out. Refuses to run before V21_FROZEN.json exists.

v2.1 acceptance rule, fixed BEFORE any variant was run: on DEV labelled panels WRONG <= v2-tuned's and HIT >= v2-tuned's
and no 'none' panel picked. DEV has no labelled MISS to recover, so recoveries are judged only by (a) an IoU proxy on
unlabelled DEV frames and (b) a leave-frames-out check of that proxy on DEV HIT panels; both are stated as such.

Usage: residuals.py dev|describe|selftest [--src NPZ_DIR]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tracker_v2 as v2  # noqa: E402
import score_gt  # noqa: E402

SRC = Path("<local working folder>/gt/npz")
KEY, LABELS = HERE.parent / "gt_key.json", HERE.parent / "gt_labels.json"
FROZEN = HERE / "V21_FROZEN.json"
PROXY_IOU = 0.5             # a detection "is the signer" if its box overlaps the path box interpolated at t this much
LH, RH = slice(91, 112), slice(112, 133)   # COCO-WholeBody hands (= feature dims 273:336 / 336:399 in slt/data.py)


# ---------------------------------------------------------------- tracker internals (mirror of v2.choose)
def trace(d, w=v2.TUNED):
    """v2.choose with its join decisions recorded. Asserted equal to v2.choose on every clip before use."""
    KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]
    W, H = (int(x) for x in d["WH"]); T = len(ND)
    cands = [v2.candidates(SA[t], DB[t], int(ND[t])) for t in range(T)]
    tr = v2.tracklets(DB, cands)
    chosen = np.full(T, -1, int); dec = {}; best = None
    if tr:
        sc = {k: v2.score(tr[k], KA, SA, DB, W, H, T, w) for k in range(len(tr))}
        best = max(range(len(tr)), key=lambda k: sc[k])
        path = dict(tr[best])
        for k in sorted((k for k in range(len(tr)) if k != best), key=lambda k: -sc[k]):
            shared = [t for t in tr[k] if t in path]
            fresh = {t: v for t, v in tr[k].items() if t not in path}
            if not fresh:
                dec[k] = dict(kind="nofresh", val=None, joined=False); continue
            if shared:
                ious = [v2.iou(DB[t, path[t]], DB[t, tr[k][t]]) for t in shared]
                val = float(np.mean(ious)); dec[k] = dict(kind="overlap", val=val, n_shared=len(shared),
                                                          frac_same=float(np.mean(np.array(ious) >= 0.5)))
            else:
                t0, t1 = min(fresh), max(fresh)
                before = [t for t in path if t < t0]; after = [t for t in path if t > t1]
                links = []   # (path frame, tracklet frame, IoU), as v2.choose tests them
                if before:
                    tb = max(before); links.append((tb, t0, v2.iou(DB[tb, path[tb]], DB[t0, fresh[t0]])))
                if after:
                    ta = min(after); links.append((ta, t1, v2.iou(DB[ta, path[ta]], DB[t1, fresh[t1]])))
                link = max(links, key=lambda x: x[2]) if links else None
                dec[k] = dict(kind="boundary", val=link[2] if link else 0.0,
                              gap=abs(link[1] - link[0]) if link else None, link=link, path_det=path[link[0]] if link else None)
            dec[k]["joined"] = dec[k]["val"] >= v2.EXT_IOU
            dec[k]["fresh"] = set(fresh)          # the frames a join adds (what the decision is really about)
            if dec[k]["joined"]:
                path.update(fresh)
        for t, det in path.items():
            chosen[t] = det
    owner = {(t, dd): k for k, trk in enumerate(tr) for t, dd in trk.items()}
    return dict(cands=cands, tr=tr, best=best, dec=dec, chosen=chosen, owner=owner)


def why_masked(d, info, t, det):
    """Why detection `det` (the signer) at frame t is not on v2's path."""
    SA, DB = d["SC_ALL"], d["DET_BOX"]
    if det not in info["cands"][t]:
        nbody = int((SA[t, det, :23] > v2.CONF).sum())
        if nbody < v2.MIN_KP:
            return f"weak: only {nbody} of 23 body/feet keypoints > {v2.CONF} (needs {v2.MIN_KP})", "weak detection"
        kept = [j for j in info["cands"][t] if v2.iou(DB[t, j], DB[t, det]) >= v2.DUP_IOU]
        if not kept:
            return "filtered for an unknown reason", "other"
        det = kept[0]   # merged into this near-duplicate; categorise that one
    k = info["owner"][(t, det)]
    if k == info["best"]:
        return "on the chosen tracklet (should not happen)", "other"
    dc = info["dec"][k]; n = len(info["tr"][k])
    if dc["kind"] == "overlap":
        return (f"own tracklet ({n} frames) overlaps the path; mean box IoU on {dc['n_shared']} shared frames "
                f"{dc['val']:.2f} < {v2.EXT_IOU} (IoU>=0.5 on {dc['frac_same']:.0%} of them)"), "overlap join rejected"
    if dc["kind"] == "boundary":
        return (f"own tracklet ({n} frames) is disjoint in time; boundary IoU {dc['val']:.2f} < {v2.EXT_IOU} "
                f"across a {dc['gap']}-frame gap"), "boundary join rejected"
    return "tracklet had no new frames", "other"


def path_frames(chosen, exclude=()):
    return [s for s in np.flatnonzero(chosen >= 0) if s not in exclude]


def ref_box(DB, chosen, t, exclude=()):
    """Path box interpolated at t from the nearest path frames on each side (one side at a clip edge)."""
    pf = path_frames(chosen, exclude)
    before = [s for s in pf if s < t]; after = [s for s in pf if s > t]
    if not before and not after:
        return None, None
    if before and after:
        a, b = max(before), min(after); f = (t - a) / (b - a)
        return (1 - f) * DB[a, chosen[a]] + f * DB[b, chosen[b]], min(t - a, b - t)
    s = max(before) if before else min(after)
    return DB[s, chosen[s]].astype(float), abs(t - s)


def proxy_signer(d, chosen, t, exclude=()):
    """The detection at t overlapping the interpolated path box by >= PROXY_IOU (all detections, weak ones too)."""
    rb, dist = ref_box(d["DET_BOX"], chosen, t, exclude)
    if rb is None or int(d["NDET"][t]) == 0:
        return None, dist
    ious = [v2.iou(rb, d["DET_BOX"][t, i]) for i in range(int(d["NDET"][t]))]
    i = int(np.argmax(ious))
    return (i if ious[i] >= PROXY_IOU else None), dist


# ---------------------------------------------------------------- data
def load(part=None):
    key, labels = json.load(open(KEY)), json.load(open(LABELS))
    clips = {}
    for pid, k in key.items():
        clips.setdefault(k["clip"], k["part"])
    data = {c: dict(np.load(SRC / f"{c}.npz")) for c, p in clips.items() if part is None or p == part}
    panels = defaultdict(list)   # clip -> [(frame, truth or None)]
    for pid, k in key.items():
        lab = labels[pid]
        assert lab != "unclear"
        panels[k["clip"]].append((k["frame"], None if lab == "none" else k["letters"][lab]))
    return data, clips, panels


def outcome(choice, truth, DBt):
    return score_gt.outcome(int(choice), truth, DBt)


# ---------------------------------------------------------------- Q2 gaps, Q3 hands
def gaps(chosen):
    """Runs of masked frames: (start, length, interior?) -- interior = path frames on both sides."""
    out, T, t = [], len(chosen), 0
    while t < T:
        if chosen[t] < 0:
            s = t
            while t < T and chosen[t] < 0:
                t += 1
            out.append((s, t - s, s > 0 and t < T and (chosen[:s] >= 0).any() and (chosen[t:] >= 0).any()))
        else:
            t += 1
    return out


def gap_report(chos):
    rows, lens = [], Counter()
    tot = sum(len(c) for c in chos.values()); masked = sum(int((c < 0).sum()) for c in chos.values())
    by = Counter()
    for c in chos.values():
        for s, L, inner in gaps(c):
            b = "<=2" if L <= 2 else "3-5" if L <= 5 else "6-15" if L <= 15 else ">15"
            by[(b, "interior" if inner else "edge")] += L; lens[L] += 1
    rows.append(f"frames {tot}, masked {masked} ({masked / tot:.1%}); clips entirely masked "
                f"{sum(int((c >= 0).sum() == 0) for c in chos.values())}")
    for b in ("<=2", "3-5", "6-15", ">15"):
        i, e = by[(b, "interior")], by[(b, "edge")]
        rows.append(f"  gap {b:5s}: {i + e:4d} masked frames ({(i + e) / max(masked, 1):.1%}) -- interior {i}, edge {e}")
    le2 = by[("<=2", "interior")] + by[("<=2", "edge")]; le5 = le2 + by[("3-5", "interior")] + by[("3-5", "edge")]
    rows.append(f"  share of masked frames in gaps <=2: {le2 / max(masked, 1):.1%}; <=5: {le5 / max(masked, 1):.1%}; "
                f"interior-only <=2 / <=5: {by[('<=2', 'interior')] / max(masked, 1):.1%} / "
                f"{(by[('<=2', 'interior')] + by[('3-5', 'interior')]) / max(masked, 1):.1%}")
    rows.append(f"  gap-length histogram (length: count of gaps): {dict(sorted(lens.items()))}")
    return rows


def hand_report(data, chos, panels):
    conf_l, conf_r, jl, jr, same, confok, spikes, swaps = [], [], [], [], [], [], [], 0
    handonly, detach = [], []   # hand jumps while the body-model wrist does not; hand root far from the wrist
    for c, d in data.items():
        ch, KA, SA, DB = chos[c], d["KP_ALL"], d["SC_ALL"], d["DET_BOX"]
        on = np.flatnonzero(ch >= 0)
        if len(on) == 0:
            continue
        kp = np.stack([KA[t, ch[t]] for t in range(len(ch)) if ch[t] >= 0])
        sc = np.stack([SA[t, ch[t]] for t in range(len(ch)) if ch[t] >= 0])
        sh = (sc[:, 5] > v2.CONF) & (sc[:, 6] > v2.CONF)
        sw = float(np.median(np.linalg.norm(kp[sh, 5] - kp[sh, 6], axis=1))) if sh.any() else np.nan
        cl, cr = sc[:, LH].mean(1), sc[:, RH].mean(1)
        conf_l += list(cl); conf_r += list(cr)
        if not np.isfinite(sw) or sw <= 0:
            continue
        # COCO-WholeBody: body wrists 9 (left) / 10 (right); hand roots 91 / 112 should sit on them
        detach += list((np.linalg.norm(kp[:, 91] - kp[:, 9], axis=1) / sw > 0.5)
                       | (np.linalg.norm(kp[:, 112] - kp[:, 10], axis=1) / sw > 0.5))
        pos = {t: i for i, t in enumerate(on)}
        cen = lambda i, h: kp[i, h].mean(0)
        for t in on:
            if t + 1 in pos:
                i, j = pos[t], pos[t + 1]
                L0, R0, L1, R1 = cen(i, LH), cen(i, RH), cen(j, LH), cen(j, RH)
                a, b = np.linalg.norm(L1 - L0) / sw, np.linalg.norm(R1 - R0) / sw
                jl.append(a); jr.append(b)
                # one-frame spike: t+1 is far from both t and t+2 while t and t+2 are close (an estimator glitch,
                # not sustained motion). Needs t+2 on the signer too.
                sp = False
                if t + 2 in pos:
                    k = pos[t + 2]
                    for h, x in ((LH, a), (RH, b)):
                        if x > 0.5 and np.linalg.norm(cen(k, h) - cen(j, h)) / sw > 0.5 and \
                                np.linalg.norm(cen(k, h) - cen(i, h)) / sw < 0.25:
                            sp = True
                spikes.append(sp)
                wl, wr = (np.linalg.norm(kp[j, w] - kp[i, w]) / sw for w in (9, 10))
                handonly.append((a > 0.5 and wl < 0.5 * a) or (b > 0.5 and wr < 0.5 * b))
                same.append(v2.iou(DB[t, ch[t]], DB[t + 1, ch[t + 1]]) >= 0.3)
                confok.append(min(cl[i], cl[j], cr[i], cr[j]) >= 0.3)
                sa, sb = np.linalg.norm(R1 - L0) / sw, np.linalg.norm(L1 - R0) / sw
                if max(a, b) > 0.5 and max(sa, sb) < 0.5 and sa + sb < 0.5 * (a + b):
                    swaps += 1
    conf_l, conf_r = np.array(conf_l), np.array(conf_r)
    jl, jr, same, confok, spikes, handonly, detach = map(np.array, (jl, jr, same, confok, spikes, handonly, detach))
    big = (jl > 0.5) | (jr > 0.5)
    out = [f"frames on the signer (v2 chosen): {len(conf_l)}",
           f"  left-hand mean conf < 0.3: {np.mean(conf_l < 0.3):.1%}; right: {np.mean(conf_r < 0.3):.1%}; "
           f"either: {np.mean((conf_l < 0.3) | (conf_r < 0.3)):.1%}; both: {np.mean((conf_l < 0.3) & (conf_r < 0.3)):.1%}",
           f"  mean hand conf left {conf_l.mean():.3f} (median {np.median(conf_l):.3f}), right {conf_r.mean():.3f} "
           f"(median {np.median(conf_r):.3f})",
           f"consecutive on-signer transitions: {len(jl)}",
           f"  per-frame hand-centroid displacement / shoulder width: left median {np.median(jl):.3f} p95 "
           f"{np.percentile(jl, 95):.3f} p99 {np.percentile(jl, 99):.3f}; right median {np.median(jr):.3f} p95 "
           f"{np.percentile(jr, 95):.3f} p99 {np.percentile(jr, 99):.3f}",
           f"  jumps > 0.5 SW: left {np.mean(jl > 0.5):.2%}, right {np.mean(jr > 0.5):.2%}, either {big.mean():.2%} "
           f"({int(big.sum())} transitions)",
           f"    of those: same-person box (IoU>=0.3) {int((big & same).sum())}, box change {int((big & ~same).sum())}; "
           f"both hands conf>=0.3 on both frames {int((big & confok).sum())}, a hand <0.3 {int((big & ~confok).sum())}; "
           f"left/right swap pattern {swaps}; one-frame spike (out and straight back) {int((big & spikes).sum())}; hand moves but its body-model wrist moves "
           f"< half as far {int((big & handonly).sum())}",
           f"  frames with a hand root > 0.5 SW from its body-model wrist (hand detached from arm): {detach.mean():.2%}",
           f"  jump rate when all four hand confs >= 0.3: {np.mean(big[confok]) if confok.any() else float('nan'):.2%} "
           f"(n={int(confok.sum())}); when any < 0.3: {np.mean(big[~confok]) if (~confok).any() else float('nan'):.2%} "
           f"(n={int((~confok).sum())})"]
    # restricted to verified signer frames (labelled HIT panels)
    hl, hr = [], []
    for c, d in data.items():
        for f, truth in panels[c]:
            if truth is not None and outcome(chos[c][f], truth, d["DET_BOX"][f]) == "HIT":
                hl.append(d["SC_ALL"][f, chos[c][f], LH].mean()); hr.append(d["SC_ALL"][f, chos[c][f], RH].mean())
    hl, hr = np.array(hl), np.array(hr)
    out.append(f"  on labelled HIT panels only (n={len(hl)}): left < 0.3 {np.mean(hl < 0.3):.1%}, right < 0.3 "
               f"{np.mean(hr < 0.3):.1%}")
    return out


# ---------------------------------------------------------------- Q4 variants
class patched:
    def __init__(self, **kw): self.kw, self.old = kw, {}
    def __enter__(self):
        for k, v in self.kw.items(): self.old[k] = getattr(v2, k); setattr(v2, k, v)
    def __exit__(self, *a):
        for k, v in self.old.items(): setattr(v2, k, v)


def reattach(d, chosen, iou_min=PROXY_IOU, max_dist=None):
    """Assign a masked frame to the detection overlapping the interpolated path box (proxy_signer), using the ORIGINAL
    path only (single pass). max_dist: only frames within this many frames of a path frame."""
    out = chosen.copy()
    for t in np.flatnonzero(chosen < 0):
        i, dist = proxy_signer(d, chosen, t)
        if i is not None and (max_dist is None or dist <= max_dist):
            out[t] = i
    return out


VARIANTS = {
    "v2-tuned": lambda d: trace(d)["chosen"],
    "join EXT_IOU 0.2": lambda d: _with(d, EXT_IOU=0.2),
    "join EXT_IOU 0.1": lambda d: _with(d, EXT_IOU=0.1),
    "link LINK_IOU 0.2": lambda d: _with(d, LINK_IOU=0.2),
    "bridge MAX_GAP 10": lambda d: _with(d, MAX_GAP=10),
    "weak MIN_KP 3": lambda d: _with(d, MIN_KP=3),
    "weak CONF 0.2": lambda d: _with(d, CONF=0.2),
    "reattach IoU 0.5 (<=5 fr)": lambda d: reattach(d, trace(d)["chosen"], 0.5, 5),
    "reattach IoU 0.5 (any)": lambda d: reattach(d, trace(d)["chosen"], 0.5),
}


def _with(d, **kw):
    with patched(**kw):
        return trace(d)["chosen"]


# ---------------------------------------------------------------- modes
def run_dev(out):
    data, clips, panels = load("dev")
    P = lambda s="": (print(s), out.append(s))
    # self-check: the mirror reproduces v2.choose exactly, and reproduces the recorded DEV score
    chos = {}
    for c, d in data.items():
        info = trace(d); ref = v2.choose(d, v2.TUNED)[1].astype(int)
        assert np.array_equal(info["chosen"], ref), c
        chos[c] = info["chosen"]
    cnt = Counter(outcome(chos[c][f], tr, data[c]["DET_BOX"][f]) for c in data for f, tr in panels[c])
    assert (cnt["HIT"], cnt["WRONG"], cnt["MISS"], cnt["OKMASK"]) == (277, 5, 0, 4), cnt   # dev_scores.txt
    P(f"## DEV, v2-tuned (30 clips, {sum(cnt.values())} panels): {dict(cnt)} -- reproduces dev_scores.txt")

    P("\n## Q1 why v2 masks frames on DEV")
    P("Labelled DEV panels: MISS = 0 (v2-tuned) -- DEV contains no labelled example of the miss mode. "
      "Diagnosis therefore uses every masked DEV frame with an IoU proxy for 'the signer was detected'.")
    # proxy validation on DEV labels (leave the panel frame, and +-g frames, out of the path)
    for g in (0, 2, 5):
        v = Counter()
        for c, d in data.items():
            for f, truth in panels[c]:
                ex = set(range(f - g, f + g + 1))
                i, _ = proxy_signer(d, chos[c], f, ex)
                v2o = outcome(chos[c][f], truth, d["DET_BOX"][f])   # errors inherited from a wrong path show up here
                if truth is None:
                    v["none: no pick" if i is None else f"none: FALSE pick (v2 {v2o})"] += 1
                elif i is None:
                    v["signer: no pick"] += 1
                elif outcome(i, truth, d["DET_BOX"][f]) == "HIT":
                    v["signer: right"] += 1
                else:
                    v[f"signer: WRONG person (v2 {v2o} here)"] += 1
        P(f"proxy check, panel frame +-{g} hidden from the path: {dict(sorted(v.items()))}")
    cat, examples, n_masked, n_edge_nopath = Counter(), defaultdict(list), 0, 0
    for c, d in data.items():
        info = trace(d)
        for t in np.flatnonzero(info["chosen"] < 0):
            n_masked += 1
            i, dist = proxy_signer(d, info["chosen"], t)
            if i is None:
                cat["no detection overlaps the path box (signer likely not detected)"
                    if int(d["NDET"][t]) == 0 or dist is not None else "no path at all"] += 1
                continue
            why, k = why_masked(d, info, t, i)
            cat[k] += 1
            if len(examples[k]) < 4:
                sa = d["SC_ALL"][t, i]
                examples[k].append(f"{c} f{t}: {why}; that detection's mean conf body {sa[:23].mean():.2f} "
                                   f"L-hand {sa[LH].mean():.2f} R-hand {sa[RH].mean():.2f}")
    P(f"masked DEV frames: {n_masked}")
    for k, n in cat.most_common():
        P(f"  {n:4d} ({n / max(n_masked, 1):.1%})  {k}")
    for k, ex in examples.items():
        for e in ex:
            P(f"    e.g. {e}")

    P("\nDEV WRONG panels (v2-tuned):")
    for r in wrong_list(data, panels): P(r)
    P("\nDEV tracklets v2 did not choose that hold a labelled panel (the join decisions that matter):")
    for r in join_table(data, panels): P(r)

    P("\n## Q2 masked-gap lengths, DEV clips (all frames)")
    for r in gap_report(chos): P(r)
    P("\n## Q3 hand quality on v2-chosen frames, DEV clips")
    for r in hand_report(data, chos, panels): P(r)

    P("\n## Q4 v2.1 variants, DEV only (held-out NOT scored)")
    P("columns: labelled panels HIT/WRONG/MISS/OKMASK | masked frames | newly assigned (vs v2-tuned) "
      "| of those proxy-consistent | proxy-signer frames still masked")
    rows = {}
    for name, fn in VARIANTS.items():
        cnt, masked, new, cons, still = Counter(), 0, 0, 0, 0
        for c, d in data.items():
            ch = fn(d); base = chos[c]
            for f, tr in panels[c]:
                cnt[outcome(ch[f], tr, d["DET_BOX"][f])] += 1
            masked += int((ch < 0).sum())
            for t in np.flatnonzero((base < 0) & (ch >= 0)):
                new += 1; i, _ = proxy_signer(d, base, t)
                cons += int(i is not None and (i == ch[t] or v2.iou(d["DET_BOX"][t, i], d["DET_BOX"][t, ch[t]]) >= 0.6))
            for t in np.flatnonzero(ch < 0):
                still += int(proxy_signer(d, ch, t)[0] is not None)
        rows[name] = dict(HIT=cnt["HIT"], WRONG=cnt["WRONG"], MISS=cnt["MISS"], OKMASK=cnt["OKMASK"], masked=masked,
                          new=new, consistent=cons, proxy_still_masked=still)
        P(f"  {name:26s} {cnt['HIT']}/{cnt['WRONG']}/{cnt['MISS']}/{cnt['OKMASK']} | {masked:4d} | {new:4d} | "
          f"{cons:4d} | {still:4d}")
    base = rows["v2-tuned"]
    ok = [n for n, r in rows.items() if n != "v2-tuned" and r["WRONG"] <= base["WRONG"] and r["HIT"] >= base["HIT"]
          and r["OKMASK"] == base["OKMASK"]]
    # among accepted, most newly assigned frames that are proxy-consistent; ties -> fewer new inconsistent, then order
    pick = max(ok, key=lambda n: (rows[n]["consistent"], -(rows[n]["new"] - rows[n]["consistent"]),
                                  -list(VARIANTS).index(n))) if ok else None
    P(f"accepted by the DEV rule: {ok}")
    P(f"frozen v2.1 candidate: {pick}")
    json.dump(dict(frozen=pick, accepted=ok, dev_rows=rows, rule=__doc__.split("v2.1 acceptance rule")[1].split("Usage")[0].strip()),
              open(FROZEN, "w"), indent=1)


def scene(d, a, b, da, db):
    """Across a boundary link (path frame a, detection da) -> (tracklet frame b, detection db): box area ratio, centre
    shift in path-box widths, and scene continuity = share of the OTHER usable detections at a that reappear at b with
    IoU >= 0.5 (near 0 with several people present suggests a camera cut rather than the signer moving)."""
    DB, SA, ND = d["DET_BOX"], d["SC_ALL"], d["NDET"]
    A, B = DB[a, da], DB[b, db]
    ar = ((B[2] - B[0]) * (B[3] - B[1])) / max(1.0, (A[2] - A[0]) * (A[3] - A[1]))
    sh = np.hypot((A[0] + A[2] - B[0] - B[2]) / 2, (A[1] + A[3] - B[1] - B[3]) / 2) / max(1.0, A[2] - A[0])
    ca = [i for i in v2.candidates(SA[a], DB[a], int(ND[a])) if v2.iou(DB[a, i], A) < v2.DUP_IOU]
    cb = v2.candidates(SA[b], DB[b], int(ND[b]))
    cont = [max([v2.iou(DB[a, i], DB[b, j]) for j in cb] or [0.0]) >= 0.5 for i in ca]
    return (f"area ratio {ar:.2f}, centre shift {sh:.2f} box widths, other people persisting "
            f"{sum(cont)}/{len(cont)}")


def wrong_list(data, panels):
    """Where each WRONG panel's pose comes from: the chosen tracklet itself, or a tracklet joined to it."""
    out = []
    for c, d in data.items():
        info = trace(d); DB = d["DET_BOX"]
        for f, truth in panels[c]:
            ch = info["chosen"][f]
            if outcome(ch, truth, DB[f]) != "WRONG":
                continue
            k = info["owner"].get((f, ch))
            src = ("chosen tracklet" if k == info["best"] else
                   f"joined tracklet ({info['dec'][k]['kind']}, IoU {info['dec'][k]['val']:.2f})")
            tb = "signer not detected" if truth is None else f"IoU(chosen, signer box) {v2.iou(DB[f, ch], DB[f, truth]):.2f}"
            out.append(f"  {c} f{f}: from the {src}; {tb}")
    return out


def join_table(data, panels):
    """Every tracklet v2 did NOT choose that holds a labelled panel: the join test v2 applied, its value, and whose
    tracklet it is by the labels (signer / another person / frames where the signer is absent)."""
    rows = []
    for c, d in data.items():
        info = trace(d); DB = d["DET_BOX"]
        for k, dc in info["dec"].items():
            if dc["kind"] == "nofresh":
                continue
            trk, lab = info["tr"][k], Counter()
            for f, truth in panels[c]:
                if f in trk:
                    who = ("none-label" if truth is None else
                           "signer" if outcome(trk[f], truth, DB[f]) == "HIT" else "other")
                    lab[("new:" if f in dc["fresh"] else "shared:") + who] += 1
            if not lab:
                continue
            extra = ""
            if dc["kind"] == "boundary" and dc["link"]:
                a, b, _ = dc["link"]; extra = f" | link f{a}->f{b}: " + scene(d, a, b, dc["path_det"], trk[b])
            rows.append((dc["kind"], dc["val"], f"  {c} {dc['kind']:8s} IoU {dc['val']:.2f} joined={dc['joined']} "
                                                f"len {len(trk):3d} labels {dict(lab)}{extra}"))
    return [r[2] for r in sorted(rows, key=lambda r: (r[0], r[1]))]


def run_describe(out):
    assert FROZEN.exists(), "run the dev mode first: the v2.1 choice must be frozen before held-out is described"
    P = lambda s="": (print(s), out.append(s))
    data, clips, panels = load()
    chos = {c: trace(d)["chosen"] for c, d in data.items()}
    for c, d in data.items():
        assert np.array_equal(chos[c], v2.choose(d, v2.TUNED)[1].astype(int)), c
    ho = {c: d for c, d in data.items() if clips[c] == "heldout"}
    cnt = Counter(outcome(chos[c][f], tr, ho[c]["DET_BOX"][f]) for c in ho for f, tr in panels[c])
    assert (cnt["HIT"], cnt["WRONG"], cnt["MISS"], cnt["OKMASK"]) == (270, 2, 17, 3), cnt   # heldout_scores.txt
    P(f"frozen v2.1 candidate (from DEV, not scored here): {json.load(open(FROZEN))['frozen']}")
    P(f"## HELD-OUT v2-tuned: {dict(cnt)} -- reproduces heldout_scores.txt")
    P("\n## Q1 (description only) why v2-tuned MISSED the 17 held-out panels, exact labels")
    k_cnt, rows = Counter(), []
    for c, d in ho.items():
        info = trace(d); g = {s + i: (L, inner) for s, L, inner in gaps(info["chosen"]) for i in range(L)}
        for f, truth in panels[c]:
            if outcome(chos[c][f], truth, d["DET_BOX"][f]) != "MISS":
                continue
            why, k = why_masked(d, info, f, truth); k_cnt[k] += 1
            px, _ = proxy_signer(d, info["chosen"], f)
            L, inner = g[f]
            rows.append(f"  {c} f{f} gap {L} ({'interior' if inner else 'edge'}) proxy "
                        f"{'right' if px is not None and outcome(px, truth, d['DET_BOX'][f]) == 'HIT' else 'no pick' if px is None else 'WRONG'}: {why}")
    for k, n in k_cnt.most_common(): P(f"  {n:3d}  {k}")
    for r in rows: P(r)
    P("\nHELD-OUT WRONG panels (v2-tuned, description only):")
    for r in wrong_list(ho, panels): P(r)
    P("\nHELD-OUT tracklets v2 did not choose that hold a labelled panel (description only):")
    for r in join_table(ho, panels): P(r)
    P("\nHELD-OUT masked frames, all frames, proxy categories (description only):")
    cat = Counter()
    for c, d in ho.items():
        info = trace(d)
        for t in np.flatnonzero(info["chosen"] < 0):
            i, _ = proxy_signer(d, info["chosen"], t)
            cat["no detection overlaps the path box" if i is None else why_masked(d, info, t, i)[1]] += 1
    for k, n in cat.most_common(): P(f"  {n:4d}  {k}")
    P("\n## labelled masked panels (all 60 clips) by the gap they sit in")
    for c, d in data.items():
        g = {s + i: (L, inner) for s, L, inner in gaps(chos[c]) for i in range(L)}
        for f, truth in panels[c]:
            o = outcome(chos[c][f], truth, d["DET_BOX"][f])
            if o in ("MISS", "OKMASK"):
                P(f"  [{clips[c]}] {c} f{f} {o}: gap {g[f][0]} {'interior' if g[f][1] else 'edge'}")
    for part in ("dev", "heldout", None):
        sel = {c: chos[c] for c in data if part is None or clips[c] == part}
        P(f"\n## Q2 masked-gap lengths, {part or 'ALL 60'} clips")
        for r in gap_report(sel): P(r)
    P("\n## Q3 hand quality on v2-chosen frames, ALL 60 clips")
    for r in hand_report(data, chos, panels): P(r)
    P("\n## Q3 hand quality, HELD-OUT clips")
    for r in hand_report(ho, {c: chos[c] for c in ho}, panels): P(r)


def selftest():
    ch = np.array([-1, 0, 0, -1, -1, 1, -1])
    assert gaps(ch) == [(0, 1, False), (3, 2, True), (6, 1, False)], gaps(ch)
    DB = np.zeros((3, 2, 4)); DB[0, 0] = [0, 0, 10, 10]; DB[2, 0] = [10, 0, 20, 10]
    DB[1, 0] = [5, 0, 15, 10]; DB[1, 1] = [100, 100, 110, 110]
    d = dict(DET_BOX=DB, NDET=np.array([1, 2, 1]))
    rb, dist = ref_box(DB, np.array([0, -1, 0]), 1)
    assert np.allclose(rb, [5, 0, 15, 10]) and dist == 1
    assert proxy_signer(d, np.array([0, -1, 0]), 1) == (0, 1)
    assert list(reattach(d, np.array([0, -1, 0]))) == [0, 0, 0]
    with patched(EXT_IOU=0.05):
        assert v2.EXT_IOU == 0.05
    assert v2.EXT_IOU == 0.3
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dev", "describe", "selftest"])
    ap.add_argument("--src", default=str(SRC)); a = ap.parse_args(); SRC = Path(a.src)
    selftest()
    if a.mode != "selftest":
        lines = []
        run_dev(lines) if a.mode == "dev" else run_describe(lines)
        (HERE / f"{a.mode}_output.txt").write_text("\n".join(lines) + "\n")
