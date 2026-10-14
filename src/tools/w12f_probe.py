"""Wave 12f: two measurements of sign identity in conversational Auslan poses.  No model is trained.

  spans   leak-safe Sign_Spotting spans (Communication TRAIN clips only), in the CSV form the recorded probe reads
  facts   what the located table says about the isolated set (pixels and labels only; no pose, no accuracy)
  iso     measurement 1: Auslan-Daily's own isolated-sign clips as the reference set
  depth   measurement 2: does an estimated third coordinate carry sign identity?
  self_test

Pre-registration: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md (rules, readings, what had been seen).
Everything that decides a number comes from tools/continuous_id_probe.py, the probe that gave the recorded 18.1%:
the descriptor, the nearest-centroid rule, the clip-disjoint folds and the row selection.  This file only chooses
which rows are queries and which are references, and tests the differences.  Needs numpy and openpyxl only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuous_id_probe import GROUPS, clip_disjoint_eval, nearest_centroid, select_rows, span_descriptor  # noqa: E402

# The 45 Sign_Spotting glosses that are also MM-WLAuslan glosses: the class set of the recorded run
# (results/CONTINUOUS_ID_PROBE_2026-09-09.md; rebuilt from both label files on 13 Oct 2026).
GLOSSES45 = ("arrange bad bat bottle build cake chisel column container corner cream dark drill early flour front "
             "goal group hop if jigsaw m moon oil onion pattern performance pie possum push rainbow rectangle rid roof "
             "rope s sad sport stack tail third to tomorrow weave wool").split()
# The recorded run (results/PROBE_CALIBRATION_2026-09-22.md, reproduced on 13 Oct 2026 with the unedited probe):
# top-1 correct of 337 scored spans, default flags with --max_per_class 8, seed 42.
RECORDED = {"body": 52, "face": 11, "hands": 61, "body_hands": 63, "all": 61, "hands [HAND-LOCAL]": 83}
VARIANTS = [("hands", "hands", False), ("body_hands", "body_hands", False), ("body", "body", False),
            ("face", "face", False), ("all", "all", False), ("hands [HAND-LOCAL]", "hands", True)]
PRIMARY = "hands"
N_T, MIN_CONF, MAX_PER, MIN_PER, FOLDS, SEED = 8, 0.10, 8, 4, 5, 42   # the recorded settings
SAFE_SET, SPAN_SET, N_RECORDED = (1008, 929, 120), (382, 361, 45), 337   # spans / clips / glosses; scored spans of the recorded run
MMWL_CLIPS = 630                                                         # MM-WLAuslan Train+Valid recordings of the 45 glosses (14 each)
SENS_SEEDS = (1, 7, 101, 202)                                            # the other seeds of the September run: sensitivity only, no reading


# ---------------------------------------------------------------- small pieces
def video_of(clip: str) -> str:
    return clip.rsplit("_", 1)[0]                      # video_10_70 -> video_10


def read_spans(path, glosses=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = r["Query_Gloss"].strip().lower()
            if glosses is None or g in glosses:
                rows.append((re.sub(r"_signer\.mp4$", "", r["Video_Name_Clip"].strip()), g,
                             int(r["Begin_Frame"]), int(r["End_Frame"])))
    return rows


class Clips:
    """Lazy cache of per-clip arrays from <dir>/<id>.npz."""

    def __init__(self, d):
        self.d, self.c = d, {}

    def has(self, cid):
        return os.path.exists(os.path.join(self.d, cid + ".npz"))

    def get(self, cid, key="KPS"):
        if (cid, key) not in self.c:
            with np.load(os.path.join(self.d, cid + ".npz")) as z:
                self.c[(cid, key)] = z[key].astype(np.float32)
        return self.c[(cid, key)]


class MultiClips(Clips):
    """The same over several directories (first hit wins)."""

    def __init__(self, dirs):
        self.dirs, self.c = dirs, {}

    def _path(self, cid):
        return next((p for p in (os.path.join(d, cid + ".npz") for d in self.dirs) if os.path.exists(p)), None)

    def has(self, cid):
        return self._path(cid) is not None

    def get(self, cid, key="KPS"):
        if (cid, key) not in self.c:
            with np.load(self._path(cid)) as z:
                self.c[(cid, key)] = z[key].astype(np.float32)
        return self.c[(cid, key)]


def seg_bounds(n, b, e):
    return max(0, b), min(n, max(b + 3, e))            # the recorded rule: at least three frames


def length_feature(ln, scale=None):
    """Length-only control for a cosine nearest-centroid rule: (length / scale, 1).  The angle of this vector is a
    monotone function of length, so the nearest centroid is the class whose mean length is nearest in that angle.
    (The recorded probe passes the bare length; the cosine of two positive scalars is always 1, so its
    'length-only' column is the frequency of one class, not a length baseline.)"""
    ln = np.asarray(ln, float)
    return np.stack([ln / (scale or max(float(np.median(ln)), 1.0)), np.ones_like(ln)], 1)


def mcnemar(pa: dict, pb: dict):
    """Exact two-sided McNemar test on the keys scored in both: (n, a correct, b correct, only a, only b, p)."""
    keys = sorted(set(pa) & set(pb))
    oa = sum(bool(pa[k]) and not pb[k] for k in keys)
    ob = sum(bool(pb[k]) and not pa[k] for k in keys)
    n = oa + ob
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(oa, ob) + 1)) / 2 ** n)
    return len(keys), sum(bool(pa[k]) for k in keys), sum(bool(pb[k]) for k in keys), oa, ob, p


def boot_diff(pa: dict, pb: dict, clips, n_boot=2000, seed=0):
    """95% interval of top-1(a) - top-1(b), resampling CLIPS (spans of one clip stay together)."""
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return float("nan"), float("nan")
    uc, inv = np.unique(np.asarray(clips)[keys], return_inverse=True)
    d = np.bincount(inv, weights=[float(pa[k]) - float(pb[k]) for k in keys], minlength=len(uc))
    c = np.bincount(inv, minlength=len(uc)).astype(float)
    ix = np.random.default_rng(seed).integers(0, len(uc), (n_boot, len(uc)))
    v = d[ix].sum(1) / c[ix].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fold_table(feats: dict, y, cl, ln, seed=SEED, folds=FOLDS):
    """The recorded protocol on one fixed set of rows: clip-disjoint folds, nearest centroid; plus the
    shuffled-label control (5 permutations, as recorded) and the length-only control."""
    out = {}
    for name, X in feats.items():
        per = {}
        acc, acc5, n = clip_disjoint_eval(X, y, cl, np.random.default_rng(seed), folds, per)
        r2 = np.random.default_rng(seed)
        sh = [clip_disjoint_eval(X, r2.permutation(y), cl, np.random.default_rng(seed + i), folds)[0] for i in range(5)]
        out[name] = dict(top1=acc, top5=acc5, n=n, correct=int(sum(per.values())), shuffled=float(np.mean(sh)), per=per)
    out["_length"] = clip_disjoint_eval(length_feature(ln), y, cl, np.random.default_rng(seed), folds)[0]
    out["_classes"] = int(len(np.unique(y)))
    return out


def show(title, t, order=None):
    print(f"\n{title}   [classes {t['_classes']}, chance {1 / t['_classes']:.2%}, length-only {t['_length']:.2%}]")
    for name in order or [k for k in t if not k.startswith("_")]:
        r = t[name]
        print(f"  {name:<34} n {r['n']:>4}  top-1 {r['top1']:6.1%} ({r['correct']})  top-5 {r['top5']:6.1%}  shuffled {r['shuffled']:5.2%}")


def strip(t):
    return {k: ({a: b for a, b in v.items() if a != "per"} if isinstance(v, dict) else v) for k, v in t.items()}


def report_pair(label, pa, pb, clips):
    n, ca, cb, oa, ob, p = mcnemar(pa, pb)
    lo, hi = boot_diff(pa, pb, clips)
    diff = (ca - cb) / n if n else float("nan")
    print(f"  {label}: {ca}/{n} = {ca / max(n, 1):.1%} against {cb}/{n} = {cb / max(n, 1):.1%} | difference {diff:+.1%} "
          f"(95% interval over clips {lo:+.1%} to {hi:+.1%}) | only first {oa}, only second {ob}, exact McNemar p = {p:.3g}")
    return dict(n=n, first=ca, second=cb, diff=diff, lo=lo, hi=hi, only_first=oa, only_second=ob, p=p)


# ---------------------------------------------------------------- spans
def cmd_spans(a):
    import openpyxl
    rows = list(openpyxl.load_workbook(a.xlsx, read_only=True).active.iter_rows(values_only=True))
    col = {str(c): i for i, c in enumerate(rows[0]) if c is not None}
    train = {os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(a.train) if "\t" in l}
    out = []
    for r in rows[1:]:
        if r[col["Video_Name_Clip"]] is None:
            continue
        cid = re.sub(r"_signer\.mp4$", "", str(r[col["Video_Name_Clip"]]).strip())
        if cid in train:
            out.append((str(r[col["Video_Name_Clip"]]).strip(), str(r[col["Query_Gloss"]]).strip(),
                        int(r[col["Begin_Frame"]]), int(r[col["End_Frame"]])))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Video_Name_Clip", "Query_Gloss", "Begin_Frame", "End_Frame"])
        w.writerows(out)
    g = lambda rs: (len(rs), len({r[0] for r in rs}), len({r[1].lower() for r in rs}))
    all_, s45 = g(out), g([r for r in out if r[1].lower() in GLOSSES45])
    print(f"annotation rows {len(rows) - 1} | on Communication TRAIN clips: spans/clips/glosses {all_} | 45-gloss set {s45} -> {a.out}")
    assert all_ == SAFE_SET and s45 == SPAN_SET, "not the span set of the recorded run"


# ---------------------------------------------------------------- the two rules (pre-registration, sections 5.2 and 6)
def reading_iso(t, tm, t2m, sens):
    """t: A2 against A0, tm: A2m against A0, t2m: A2 against A2m (dicts with p and diff), sens: A2 against A0 at the other seeds."""
    if t["p"] >= 0.05:
        reading = "R1: no measurable difference from the conversational reference"
    elif t["diff"] > 0:
        if tm["p"] < 0.05 and tm["diff"] > 0:
            reading = "R2a: better already with the query set's own twins as references (A2m): leave-one-clip-out against five folds"
        elif t2m["p"] < 0.05 and t2m["diff"] > 0:      # the direct test: absence of a gain in A2m alone attributes nothing
            reading = "R2b: better than A0 and better than the twins alone (A2 against A2m): the further isolated clips of these signs add references per sign"
        else:
            reading = "R2u: better than A0, gain not attributed (A2m is not measurably better than A0, and A2 is not measurably better than A2m)"
    else:
        reading = "R3: worse than the conversational reference"
    same_sign = sum(x["diff"] * t["diff"] > 0 for x in sens)
    if not reading.startswith("R1") and same_sign < 3:
        reading += f" [not stable across subsamples: {same_sign} of {len(SENS_SEEDS)} other seeds share the sign]"
    return reading, same_sign


def reading_depth(p_z, test, valid, sens):
    """p_z: permutation p of z alone, test: x/y/z against x/y (p and diff), valid: both validity checks hold."""
    if p_z >= 0.05:
        reading = "C1: the estimated depth carries no measurable sign identity; closed"
    elif not (test["p"] < 0.05 and test["diff"] > 0):
        reading = "C2: depth carries sign identity but adds nothing measurable to x/y; closed"
    else:
        reading = "C3: depth adds identity to x/y in this probe; a re-proposal, not a training run"
        pos = sum(x["diff"] > 0 for x in sens)
        if pos < 3:
            reading += f" [not stable across subsamples: {pos} of {len(SENS_SEEDS)} other seeds are positive]"
    if not valid:
        reading += " [VALIDITY CHECK FAILED: report as not interpretable, see the pre-registration, section 6]"
    return reading


# ---------------------------------------------------------------- facts (no pose)
def cmd_facts(a):
    """Dataset facts the design rests on, from the located table and the span table alone.  Stops the chain before
    the GPU step if the isolated clips of the spotting glosses are not the spotting spans (pre-registration, section 2)."""
    with open(a.located, newline="") as f:
        loc = list(csv.DictReader(f))
    spans = read_spans(a.spans)
    tally = {}
    for r in loc:
        tally[(r["status"], r["comm_split"])] = tally.get((r["status"], r["comm_split"]), 0) + 1
    ok = [r for r in loc if r["status"] == "located"]
    exact0 = sum(r["d_min"] != "" and float(r["d_min"]) == 0.0 for r in loc)
    at = {(r["Video_Name_Clip"], r["Query_Gloss"].strip().lower(), int(r["Begin_Frame"])): int(r["frames"])
          for r in ok if r["comm_split"] == "train"}
    twins = sum((c, g, max(0, b)) in at for c, g, b, _ in spans)
    same_len = sum(at.get((c, g, max(0, b))) == e - b for c, g, b, e in spans)      # the whole range [Begin, End), not only its first frame
    s45 = [s for s in spans if s[1] in GLOSSES45]
    tw45 = sum((c, g, max(0, b)) in at for c, g, b, _ in s45)
    len45 = sum(at.get((c, g, max(0, b))) == e - b for c, g, b, e in s45)
    ref45 = [r for r in ok if r["comm_split"] == "train" and r["Query_Gloss"].strip().lower() in GLOSSES45]
    print(f"isolated clips {len(loc)}, glosses {len({r['Query_Gloss'].strip().lower() for r in loc})}; thumbnail distance exactly 0: {exact0}")
    print("status and split of the source clip: " + ", ".join(f"{s} {c or '-'} {k}" for (s, c), k in sorted(tally.items())))
    print(f"leak-safe spotting spans {len(spans)}; with an isolated clip at the same clip, gloss and first frame: {twins}; "
          f"of these with exactly End - Begin frames: {same_len}")
    print(f"45-gloss set: spans {len(s45)}, with such a twin {tw45} (exactly End - Begin frames: {len45}); "
          f"isolated clips of these glosses located on a TRAIN clip: {len(ref45)}")
    res = dict(isolated=len(loc), tally={f"{s}|{c}": k for (s, c), k in tally.items()}, distance_exactly_0=exact0, safe_spans=len(spans),
               safe_spans_with_twin=twins, safe_spans_with_twin_of_equal_length=same_len, spans45=len(s45), spans45_with_twin=tw45, spans45_with_twin_of_equal_length=len45,
               isolated45_on_train=len(ref45))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    assert len(ok) >= 0.9 * len(loc), "fewer than 90% of the isolated clips were located in a signer clip"
    assert twins >= 0.95 * len(spans) and tw45 >= 0.95 * len(s45), "the isolated clips of the spotting glosses are not the spotting spans"
    print("FACTS OK: the isolated clips of the spotting glosses are the spotting spans (same clip, same first frame)")
    if a.prod_dir:
        harness_gate(s45, Clips(a.prod_dir))


def harness_gate(spans45, prod):
    """Gate H: on this machine the probe reproduces the recorded run (correct spans of 337, every variant), from the
    production poses alone.  Run in the CPU staging step, so a platform difference stops the chain before the GPU step."""
    for name, grp, hloc in VARIANTS:
        Xq, yq, clq, _, _ = query_set(spans45, prod, GROUPS[grp], hloc)
        per = {}
        _, _, n = clip_disjoint_eval(Xq, yq, clq, np.random.default_rng(SEED), FOLDS, per)
        assert (int(sum(per.values())), n) == (RECORDED[name], N_RECORDED), \
            f"gate H FAILED for {name}: {sum(per.values())}/{n}, recorded {RECORDED[name]}/{N_RECORDED}"
    print("GATE H PASSED: the recorded run is reproduced here for all six variants " +
          ", ".join(f"{k} {v}/{N_RECORDED}" for k, v in RECORDED.items()))


# ---------------------------------------------------------------- measurement 1
def query_set(spans, prod, idx, hloc, seed=SEED):
    """Descriptors of the conversational spans from the production DWPose clips, then the recorded row selection."""
    X, y, cl, ln, key = [], [], [], [], []
    for cid, g, b, e in spans:
        k = prod.get(cid)
        b2, e2 = seg_bounds(len(k), b, e)
        v = span_descriptor(k[b2:e2], idx, N_T, MIN_CONF, True, hloc)
        if v is None:
            continue
        X.append(v); y.append(g); cl.append(cid); ln.append(e2 - b2); key.append((cid, b))
    X, y, cl, ln = np.array(X), np.array(y), np.array(cl), np.array(ln, float)
    sel = select_rows(y, MAX_PER, MIN_PER, seed)
    return X[sel], y[sel], cl[sel], ln[sel], [key[i] for i in sel]


def reference_arm(Xq, yq, Xb, yb, ok, drop):
    """Nearest centroid of each query among the reference rows `ok & ~drop(i)`.  A query is scored only if its own
    class still has a reference row.  Returns ({query index: top-1 correct}, {query index: in top 5})."""
    per, top5 = {}, {}
    for i in range(len(yq)):
        m = ok & ~drop(i)
        if not (m & (yb == yq[i])).any():
            continue
        pred, topk = nearest_centroid(Xb[m], yb[m], Xq[i:i + 1])
        per[i] = bool(pred[0] == yq[i])
        top5[i] = bool((topk[0] == yq[i]).any())
    return per, top5


def arm_line(name, per, top5, extra=""):
    n = len(per)
    print(f"  {name:<58} n {n:>4}  top-1 {sum(per.values()) / max(n, 1):6.1%} ({sum(per.values())})  "
          f"top-5 {sum(top5.values()) / max(n, 1):6.1%}{extra}")
    return dict(n=n, correct=int(sum(per.values())), top1=sum(per.values()) / max(n, 1), top5=sum(top5.values()) / max(n, 1))


def keypoint_agreement(loc, bank, prod):
    """Median |dxy| (pixels, keypoints confident in both) between the DWPose of each isolated clip located on a TRAIN
    clip and the production DWPose of the frame range it was cut from.  share_under_1px is a share of ALL checked
    clips: a clip whose frame count differs, or that has no keypoint confident in both poses (an all-zero pose from a
    failed extraction, for example), counts as not under 1 pixel."""
    med, checked, bad_len = [], 0, 0
    for r in loc:
        if r["status"] != "located" or r["comm_split"] != "train" or not bank.has(r["Video_ID"]) or not prod.has(r["Video_Name_Clip"]):
            continue
        kb, kp = bank.get(r["Video_ID"]), prod.get(r["Video_Name_Clip"])[int(r["Begin_Frame"]):int(r["End_Frame"])]
        checked += 1
        if kb.shape != kp.shape:
            bad_len += 1
            continue
        m = (kb[..., 2] >= 0.3) & (kp[..., 2] >= 0.3)
        if m.any():
            med.append(float(np.median(np.linalg.norm(kb[..., :2] - kp[..., :2], axis=-1)[m])))
    med = np.array(med)
    return dict(checked=checked, frame_count_differs=bad_len, compared=int(len(med)),
                median_px=float(np.median(med)) if len(med) else float("nan"),
                share_under_1px=float((med < 1.0).sum() / checked) if checked else float("nan"))


def cmd_iso(a):
    spans = read_spans(a.spans, set(GLOSSES45))
    assert (len(spans), len({s[0] for s in spans}), len({s[1] for s in spans})) == SPAN_SET, "span set of the recorded run"
    prod, bank = Clips(a.prod_dir), Clips(a.bank_dir)
    with open(a.located, newline="") as f:
        loc = list(csv.DictReader(f))
    for r in loc:
        r["g"] = r["Query_Gloss"].strip().lower()
    res = {"settings": dict(n_t=N_T, min_conf=MIN_CONF, max_per_class=MAX_PER, min_per_class=MIN_PER, folds=FOLDS, seed=SEED)}
    mmwl, mm = [], None
    if a.mmwl_labels and not a.gate_only:          # {recording id: GLOSS}; the pose files are <dir>/<id>.npz
        mm = MultiClips(a.mmwl_dirs)
        for lab in a.mmwl_labels:
            with open(lab) as f:
                mmwl += [(str(i), str(g).strip().lower()) for i, g in json.load(f).items() if str(g).strip().lower() in GLOSSES45]
        missing = [i for i, _ in mmwl if not mm.has(i)]
        assert len(mmwl) == MMWL_CLIPS and not missing, f"MM-WLAuslan: {len(mmwl)} recordings of the 45 signs (expected {MMWL_CLIPS}), {len(missing)} without a pose file"

    # ---- dataset facts (no accuracy)
    tally = {}
    for r in loc:
        tally[(r["status"], r["comm_split"])] = tally.get((r["status"], r["comm_split"]), 0) + 1
    all_spans = read_spans(a.spans)
    span_keys = {(c, g, b) for c, g, b, _ in all_spans}
    twins = sum((r["Video_Name_Clip"], r["g"], int(r["Begin_Frame"])) in span_keys for r in loc if r["status"] == "located")
    print(f"isolated clips {len(loc)}; by status and split of the source clip: " +
          ", ".join(f"{s} {c or '-'} {k}" for (s, c), k in sorted(tally.items())))
    print(f"leak-safe spotting spans {len(all_spans)}; located isolated clips that ARE one of them (same clip, gloss, first frame): {twins}")
    res["facts"] = dict(isolated=len(loc), tally={f"{s}|{c}": k for (s, c), k in tally.items()}, safe_spans=len(all_spans),
                        isolated_that_are_safe_spans=twins)

    # ---- gate E: the DWPose of the isolated clips is the production DWPose of the frames they were cut from
    ka = keypoint_agreement(loc, bank, prod)
    print(f"gate E (extractor and source): {ka}")
    res["gate_E"] = ka
    assert ka["compared"] >= 50, "gate E: too few located isolated clips could be compared with production poses"
    assert ka["median_px"] < 1.0 and ka["share_under_1px"] >= 0.95 and ka["frame_count_differs"] <= 0.02 * ka["checked"], \
        "gate E FAILED: the isolated clips' poses are not the production poses of their located frame range"

    res["variants"] = {}
    for name, grp, hloc in VARIANTS:
        idx = GROUPS[grp]
        Xq, yq, clq, lnq, keyq = query_set(spans, prod, idx, hloc)
        A0 = {}
        acc0, acc0_5, n0 = clip_disjoint_eval(Xq, yq, clq, np.random.default_rng(SEED), FOLDS, A0)
        # ---- gate H: the harness reproduces the recorded run
        assert (int(sum(A0.values())), n0) == (RECORDED[name], N_RECORDED), \
            f"gate H FAILED for {name}: {sum(A0.values())}/{n0}, recorded {RECORDED[name]}/{N_RECORDED}"

        # reference rows: isolated clips of the query classes
        classes = set(yq.tolist())
        Xb, yb, cb, vb, lb, st, sp_, bg = [], [], [], [], [], [], [], []
        cand = 0           # reference clips that SHOULD give a descriptor: located on a TRAIN clip, at least 3 frames
        for r in loc:      # only clips located on a TRAIN clip: no pose of a clip cut from validation or test footage is read
            if r["g"] not in classes or r["status"] != "located" or r["comm_split"] != "train":
                continue
            cand += int(r["frames"]) >= 3
            if not bank.has(r["Video_ID"]):
                continue
            v = span_descriptor(bank.get(r["Video_ID"]), idx, N_T, MIN_CONF, True, hloc)
            if v is None:
                continue
            Xb.append(v); yb.append(r["g"]); cb.append(r["Video_Name_Clip"]); lb.append(float(r["frames"]))
            st.append(r["status"]); sp_.append(r["comm_split"]); bg.append(int(r["Begin_Frame"]))
        Xb, yb, cb, lb, st, sp_ = np.array(Xb), np.array(yb), np.array(cb), np.array(lb), np.array(st), np.array(sp_)
        # a reference clip without a pose file, or whose pose gives no descriptor (an all-zero pose from a failed
        # extraction), is dropped above without a word: the reference set must not shrink that way
        assert len(yb) >= 0.95 * cand, f"reference set FAILED for {name}: {len(yb)} of {cand} reference clips gave a descriptor"
        vb = np.array([video_of(c) if c else "" for c in cb])
        safe = (st == "located") & (sp_ == "train")
        twin_of = {(c, g, b): j for j, (c, g, b) in enumerate(zip(cb, yb, bg)) if safe[j]}
        tw = np.array([twin_of.get((keyq[i][0], yq[i], max(0, keyq[i][1])), -1) for i in range(len(yq))])
        is_twin = np.zeros(len(yb), bool)
        is_twin[tw[tw >= 0]] = True

        # A1: same rows, folds and seed as A0, each span described from its isolated twin
        X1 = Xq.copy()
        X1[tw >= 0] = Xb[tw[tw >= 0]]
        A1 = {}
        clip_disjoint_eval(X1, yq, clq, np.random.default_rng(SEED), FOLDS, A1)
        agree = float(np.mean([A1[k] == A0[k] for k in A0 if k in A1]))
        out = dict(queries=int(len(yq)), queries_with_twin=int((tw >= 0).sum()), A1_agreement_with_A0=agree,
                   reference_candidates=int(cand), reference_rows=int(len(yb)), reference_rows_safe=int(safe.sum()),
                   A0=dict(n=n0, correct=int(sum(A0.values())), top1=acc0, top5=acc0_5))
        print(f"\n=== {name}: {len(yq)} query spans, {len(classes)} classes | reference clips (isolated clips of these classes "
              f"located on a TRAIN clip, at least 3 frames) {cand}, with a descriptor {len(yb)} | "
              f"queries with an isolated twin {int((tw >= 0).sum())} | A1 agrees with A0 on {agree:.1%} of spans")
        if a.gate_only:
            assert (tw >= 0).mean() >= 0.95 and agree >= 0.95, "gate T FAILED: the isolated clips are not the query spans"
            res["variants"][name] = out
            continue

        qv = np.array([video_of(c) for c in clq])
        arms = {
            "A2 isolated, own clip excluded [PRIMARY]": (safe, lambda i: cb == clq[i]),
            "A3 isolated, own video excluded": (safe, lambda i: vb == qv[i]),
            "A2m isolated twins of the query set only, own clip excluded": (safe & is_twin, lambda i: cb == clq[i]),
            "A4 isolated, nothing excluded (self-match; leak size only)": (safe, lambda i: np.zeros(len(yb), bool)),
        }
        # length-only control for A0 with a feature the cosine rule can use (the recorded probe's own column cannot: see length_feature)
        len0 = clip_disjoint_eval(length_feature(lnq), yq, clq, np.random.default_rng(SEED), FOLDS)[0]
        out["A0"]["length_only"] = len0
        print(f"  {'A0 conversational spans, 5 clip-disjoint folds (recorded)':<58} n {n0:>4}  top-1 {acc0:6.1%} "
              f"({sum(A0.values())})  top-5 {acc0_5:6.1%}   length only {len0:.2%}")
        per = {"A0": A0}
        for arm, (ok, drop) in arms.items():
            p, t5 = reference_arm(Xq, yq, Xb, yb, ok, drop)
            cnt = np.array([int((ok & ~drop(i) & (yb == yq[i])).sum()) for i in range(len(yq))])
            out[arm] = arm_line(arm, p, t5, f"   same-class references per query: mean {cnt.mean():.1f}, min {cnt.min()}")
            per[arm.split()[0]] = p
        # controls for the primary arm: shuffled reference labels (5 permutations) and length only
        ok, drop = arms["A2 isolated, own clip excluded [PRIMARY]"]
        r2, sh = np.random.default_rng(SEED), []
        for _ in range(5):
            ys = yb.copy()
            ys[ok] = r2.permutation(yb[ok])
            p, _ = reference_arm(Xq, yq, Xb, ys, ok, drop)
            sh.append(sum(p.values()) / max(len(p), 1))
        s_len = max(float(np.median(lnq)), 1.0)
        pl, _ = reference_arm(length_feature(lnq, s_len), yq, length_feature(lb, s_len), yb, ok, drop)
        out["A2_controls"] = dict(chance=1 / len(classes), shuffled=float(np.mean(sh)), length_only=sum(pl.values()) / max(len(pl), 1))
        print(f"  A2 controls: chance {1 / len(classes):.2%} | shuffled reference labels {np.mean(sh):.2%} | "
              f"length only {out['A2_controls']['length_only']:.2%}")
        print("  paired comparisons on the spans scored in both:")
        out["tests"] = {k: report_pair(f"{k} against A0", per[k], A0, clq) for k in ("A2", "A3", "A2m", "A4")}
        out["tests"]["A2_vs_A2m"] = report_pair("A2 against A2m", per["A2"], per["A2m"], clq)
        out["tests"]["A1"] = report_pair("A1 (twin descriptors, A0 protocol) against A0", A1, A0, clq)
        if name == PRIMARY:
            print("  sensitivity (other 8-per-sign subsamples and folds; no reading is taken from these):")
            out["seed_sensitivity"] = {}
            for sd in SENS_SEEDS:
                Xs, ys, cs, _, _ = query_set(spans, prod, idx, hloc, sd)
                a0 = {}
                clip_disjoint_eval(Xs, ys, cs, np.random.default_rng(sd), FOLDS, a0)
                p2, _ = reference_arm(Xs, ys, Xb, yb, safe & np.isin(yb, ys), lambda i: cb == cs[i])
                out["seed_sensitivity"][str(sd)] = report_pair(f"seed {sd}: A2 against A0", p2, a0, cs)
        if mmwl:
            Xm = [(g, span_descriptor(mm.get(i), idx, N_T, MIN_CONF, True, hloc)) for i, g in mmwl if g in classes]
            ym = np.array([g for g, v in Xm if v is not None]); Xm = np.array([v for _, v in Xm if v is not None])
            p5, t5 = reference_arm(Xq, yq, Xm, ym, np.ones(len(ym), bool), lambda i: np.zeros(len(ym), bool))
            out["A5"] = arm_line("A5 MM-WLAuslan citation recordings of the same signs", p5, t5, f"   reference clips {len(ym)}")
            out["tests"]["A5"] = report_pair("A5 against A0", p5, A0, clq)
            out["tests"]["A5_vs_A2"] = report_pair("A5 against A2", p5, per["A2"], clq)
        res["variants"][name] = out

    if a.gate_only:
        print("\nGATES PASSED (H: recorded run reproduced; E: isolated poses = production poses; T: isolated clips are the query spans)")
    else:
        # ---- the reading fixed in the pre-registration (section 5), on the primary group
        v = res["variants"][PRIMARY]
        t, tm = v["tests"]["A2"], v["tests"]["A2m"]
        same_footage = v["queries_with_twin"] >= 0.95 * v["queries"]
        reading, same_sign = reading_iso(t, tm, v["tests"]["A2_vs_A2m"], v["seed_sensitivity"].values())
        res["reading"] = dict(same_footage=bool(same_footage), reading=reading, other_seeds_same_sign=int(same_sign))
        print(f"\nREADING ({PRIMARY}): isolated clips are the query spans' own footage: {same_footage} | {reading}")

        # ---- secondary, descriptive: the recorded protocol inside the isolated set (source clips on TRAIN only)
        res["within"] = {}
        for title, keep in (("120 spotting glosses", {g for _, g, _, _ in all_spans}), ("all glosses of the isolated set", None)):
            rows = [r for r in loc if r["status"] == "located" and r["comm_split"] == "train" and bank.has(r["Video_ID"])
                    and (keep is None or r["g"] in keep)]
            feats, base = {}, None
            for name, grp, hloc in VARIANTS[:5]:
                d = [span_descriptor(bank.get(r["Video_ID"]), GROUPS[grp], N_T, MIN_CONF, True, hloc) for r in rows]
                feats[name] = d
                base = [x is not None and (base is None or base[i]) for i, x in enumerate(d)]
            rows = [r for r, k in zip(rows, base) if k]
            y = np.array([r["g"] for r in rows])
            sel = select_rows(y, 0, MIN_PER, SEED)
            feats = {n: np.array([x for x, k in zip(d, base) if k])[sel] for n, d in feats.items()}
            t = fold_table(feats, y[sel], np.array([r["Video_Name_Clip"] for r in rows])[sel],
                           np.array([float(r["frames"]) for r in rows])[sel])
            show(f"within the isolated set, {title}, clip-disjoint folds, classes with >= {MIN_PER} clips", t)
            res["within"][title] = strip(t)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")


# ---------------------------------------------------------------- measurement 2
def z_descriptor(z, conf, idx):
    """The recorded descriptor applied to depth alone.  span_descriptor standardises x and y of a frame together;
    given (z, z) it standardises z by its own mean and spread, anchors it at the shoulders, masks, resamples and
    adds velocity exactly as for x/y.  Each value appears twice, so 1/sqrt(2) gives depth the weight of ONE coordinate."""
    v = span_descriptor(np.stack([z, z, conf], -1), idx, N_T, MIN_CONF, True, False)
    return None if v is None else v / math.sqrt(2.0)


def perm_p(X, y, cl, n_perm, seed=SEED):
    obs = clip_disjoint_eval(X, y, cl, np.random.default_rng(seed), FOLDS)[0]
    r = np.random.default_rng(seed + 1000)
    null = np.array([clip_disjoint_eval(X, r.permutation(y), cl, np.random.default_rng(seed), FOLDS)[0] for _ in range(n_perm)])
    return obs, float((1 + (null >= obs).sum()) / (n_perm + 1)), float(np.percentile(null, 95))


def cmd_depth(a):
    prod, d3 = Clips(a.prod_dir), Clips(a.d3_dir)
    res = {"settings": dict(n_t=N_T, min_conf=MIN_CONF, max_per_class=MAX_PER, min_per_class=MIN_PER, folds=FOLDS, seed=SEED)}
    names = ["x/y, production DWPose", "x/y, 3-D model", "x/y/z, 3-D model", "z only, 3-D model"]
    for title, glosses in (("45 glosses of the recorded run", set(GLOSSES45)), ("all 120 spotting glosses", None)):
        spans = read_spans(a.spans, glosses)
        missing = sorted({s[0] for s in spans if not d3.has(s[0])})
        assert not missing, f"{len(missing)} clips have no 3-D extraction, e.g. {missing[:5]}"
        res[title] = {}
        for name, grp, _ in VARIANTS[:3] + VARIANTS[4:5]:        # hands (primary), body+hands, body, all
            idx = GROUPS[grp]
            F = {n: [] for n in names}
            y, cl, ln, differ, flat, zin, zt = [], [], [], 0, [], [], []
            for cid, g, b, e in spans:
                kp, k3, z3 = prod.get(cid), d3.get(cid), d3.get(cid, "Z")
                differ += len(kp) != len(k3)
                b2, e2 = seg_bounds(min(len(kp), len(k3)), b, e)
                fp = span_descriptor(kp[b2:e2], idx, N_T, MIN_CONF, True, False)
                fx = span_descriptor(k3[b2:e2], idx, N_T, MIN_CONF, True, False)
                fz = z_descriptor(z3[b2:e2], k3[b2:e2, :, 2], idx)
                if fp is None or fx is None or fz is None:
                    continue
                zh = z3[b2:e2][:, GROUPS["hands"]]          # depth of the 42 hand keypoints, in the model's root-relative units
                flat.append(float((zh.std(1) < 1e-6).mean())); zin.append(float(np.median(zh.std(1)))); zt.append(float(np.median(zh.std(0))))
                F[names[0]].append(fp); F[names[1]].append(fx); F[names[2]].append(np.concatenate([fx, fz])); F[names[3]].append(fz)
                y.append(g); cl.append(cid); ln.append(e2 - b2)
            y, cl, ln = np.array(y), np.array(cl), np.array(ln, float)
            sel = select_rows(y, MAX_PER, MIN_PER, SEED)
            feats = {n: np.array(v)[sel] for n, v in F.items()}
            t = fold_table(feats, y[sel], cl[sel], ln[sel])
            show(f"{title} | {name} | spans whose frame count differs between the two extractions: {differ} | "
                 f"frames with constant hand depth: {np.mean(flat):.2%} | hand depth spread (median over spans): "
                 f"across the hand's keypoints {np.median(zin):.3f}, over time {np.median(zt):.3f}", t, names)
            out = strip(t)
            out["frame_count_differs"], out["constant_hand_depth_frames"] = int(differ), float(np.mean(flat))
            out["hand_depth_spread_across_keypoints"], out["hand_depth_spread_over_time"] = float(np.median(zin)), float(np.median(zt))
            if name == PRIMARY:
                primary = title.startswith("45")       # the reading comes from the 45-gloss set; the 120-gloss set is secondary
                h = t[names[0]]
                rows_recorded = (h["correct"], h["n"]) == (RECORDED[PRIMARY], N_RECORDED)
                if primary:
                    # gate H: the production x/y on these rows is the recorded run, unless the 3-D extraction dropped a span
                    print(f"  harness: production x/y {h['correct']}/{h['n']} (recorded {RECORDED[PRIMARY]}/{N_RECORDED})")
                    if not rows_recorded:
                        print("  WARNING: these are not the rows of the recorded run (a span has no 3-D descriptor); "
                              "the comparison below is still paired on identical rows")
                obs, p_z, null95 = perm_p(feats[names[3]], y[sel], cl[sel], a.n_perm)
                print(f"  z only: top-1 {obs:.1%}; label-permutation null ({a.n_perm} permutations) 95th percentile {null95:.1%}; p = {p_z:.4f}")
                test = report_pair("x/y/z against x/y (3-D model)", t[names[2]]["per"], t[names[1]]["per"], cl[sel])
                gap = t[names[1]]["top1"] - t[names[0]]["top1"]
                comparable = abs(gap) <= 0.06
                print(f"  validity: 3-D model x/y minus production x/y = {gap:+.1%} (comparable if within 6 points: {comparable}); "
                      f"constant hand depth in {np.mean(flat):.2%} of frames (valid if under 1%: {np.mean(flat) < 0.01})")
                d = dict(z_only_top1=obs, z_only_p=p_z, z_null95=null95, xyz_vs_xy=test, xy_model_minus_production=gap,
                         xy_comparable=bool(comparable), depth_not_constant=bool(np.mean(flat) < 0.01))
                if primary:
                    print("  sensitivity (other 8-per-sign subsamples and folds; no reading is taken from these):")
                    d["seed_sensitivity"] = {}
                    for sd in SENS_SEEDS:
                        ss = select_rows(y, MAX_PER, MIN_PER, sd)
                        pers = []
                        for n_ in names[1:]:
                            pr = {}
                            acc = clip_disjoint_eval(np.array(F[n_])[ss], y[ss], cl[ss], np.random.default_rng(sd), FOLDS, pr)[0]
                            pers.append((acc, pr))
                        r_ = report_pair(f"seed {sd}: x/y/z against x/y", pers[1][1], pers[0][1], cl[ss])
                        print(f"    seed {sd}: z only top-1 {pers[2][0]:.1%}")
                        d["seed_sensitivity"][str(sd)] = dict(r_, z_only_top1=pers[2][0])
                if primary:
                    reading = reading_depth(p_z, test, comparable and np.mean(flat) < 0.01, d["seed_sensitivity"].values())
                    print(f"READING ({PRIMARY}, 45 glosses): {reading}")
                    out["primary"] = dict(d, rows_are_the_recorded_run=bool(rows_recorded), reading=reading)
                else:
                    out["secondary"] = d
            res[title][name] = out
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")


# ---------------------------------------------------------------- self-test
def self_test():
    # exact McNemar: the two pairs recorded in results/PROBE_CALIBRATION_2026-09-22.md (scipy binomtest at the time)
    mk = lambda oa, ob: ({i: i < oa for i in range(oa + ob)}, {i: i >= oa for i in range(oa + ob)})
    assert f"{mcnemar(*mk(59, 14))[5]:.1e}" == "1.0e-07" and f"{mcnemar(*mk(58, 16))[5]:.1e}" == "9.7e-07"
    assert mcnemar(*mk(3, 3))[5] == 1.0 and mcnemar({}, {})[5] == 1.0 and abs(mcnemar(*mk(1, 5))[5] - 0.21875) < 1e-12
    lo, hi = boot_diff(*mk(30, 5), np.arange(35))
    assert 0.4 < lo < 25 / 35 < hi <= 1.0

    # row selection (the recorded rule): at most 2 per class, then classes with at least 2
    y = np.array(list("aaaabbc"))
    s = select_rows(y, 2, 2, 0)
    assert sorted(y[s].tolist()) == list("aabb") and np.array_equal(select_rows(y, 0, 4, 0), np.arange(4))

    # length-only control: cosine nearest centroid on (length/median, 1) picks the class of nearest mean length
    yl = np.array(["short", "short", "long", "long"])
    assert nearest_centroid(length_feature([10, 12, 40, 44], 20), yl, length_feature([11, 42], 20))[0].tolist() == ["short", "long"]
    assert len(set(nearest_centroid(np.array([[10.], [12.], [40.], [44.]]), yl, np.array([[11.], [42.]]))[0])) == 1   # the bare length cannot

    # reference arms on descriptors with NO class signal, where every reference row is a copy of a query (its twin):
    # with nothing excluded the query finds itself; with its own clip excluded it is at chance.
    rng = np.random.default_rng(0)
    C, per_c = 20, 4
    Xq = rng.normal(size=(C * per_c, 64)); yq = np.repeat(np.arange(C).astype(str), per_c)
    cq = np.array([f"video_{i}_{i}" for i in range(len(yq))])
    ok, none = np.ones(len(yq), bool), (lambda i: np.zeros(len(yq), bool))
    naive, _ = reference_arm(Xq, yq, Xq, yq, ok, none)
    clean, _ = reference_arm(Xq, yq, Xq, yq, ok, lambda i: cq == cq[i])
    assert sum(naive.values()) / len(naive) > 0.9 and sum(clean.values()) / len(clean) < 0.25, (sum(naive.values()), sum(clean.values()))
    # a query whose class has no reference left is not scored
    one, _ = reference_arm(Xq[:1], yq[:1], Xq[:1], yq[:1], ok[:1], lambda i: np.ones(1, bool))
    assert one == {}

    # depth descriptor: (z, z) through the recorded descriptor ranks classes exactly as a direct single-channel
    # implementation does (standardise per frame, anchor at the shoulders, mask, resample to 8 steps, add velocity)
    def direct(z, conf, idx):
        zz = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)
        zz = zz - zz[:, [5, 6]].mean(1, keepdims=True)
        zz = np.where(conf < MIN_CONF, np.nan, zz)[:, idx]
        g = np.linspace(0, len(zz) - 1, N_T); lo = np.floor(g).astype(int); hi = np.minimum(lo + 1, len(zz) - 1)
        w = (g - lo)[:, None]
        d = (1 - w) * zz[lo] + w * zz[hi]
        return np.nan_to_num(np.concatenate([d, np.diff(d, axis=0, prepend=d[:1])], 1).reshape(-1), nan=0.0)
    idx = GROUPS["hands"]
    Z = [rng.normal(size=(int(rng.integers(5, 30)), 133)) for _ in range(40)]
    Cf = [np.where(rng.random(z.shape) < 0.1, 0.0, 0.9) for z in Z]
    A = np.array([z_descriptor(z, c, idx) for z, c in zip(Z, Cf)]); B = np.array([direct(z, c, idx) for z, c in zip(Z, Cf)])
    assert A.shape[1] == 2 * B.shape[1] and np.allclose(A[:, 0::2] * math.sqrt(2), B, atol=1e-5) and np.allclose(A[:, 0::2], A[:, 1::2])
    yy = np.array(list("abcd") * 10)
    assert np.array_equal(nearest_centroid(A[:20], yy[:20], A[20:])[0], nearest_centroid(B[:20], yy[:20], B[20:])[0])
    # one coordinate's weight: |z part|^2 equals the single-channel norm
    assert np.allclose((A ** 2).sum(1), (B ** 2).sum(1), rtol=1e-5)
    # the two rules, on hand-made test results
    T = lambda p, d: dict(p=p, diff=d)
    up, down, flat4 = [T(1, .01)] * 4, [T(1, -.01)] * 4, [T(1, 0.0)] * 4
    yes, no = T(.01, .03), T(.2, .01)                                               # A2 against A2m: measurably higher / not
    assert reading_iso(T(.2, .05), T(.01, .05), yes, up)[0].startswith("R1")
    assert reading_iso(T(.01, .05), T(.01, .04), no, up)[0].startswith("R2a") and reading_iso(T(.01, .05), T(.01, .04), yes, up)[0].startswith("R2a")
    assert reading_iso(T(.01, .05), T(.2, .04), yes, up)[0].startswith("R2b")
    assert reading_iso(T(.01, .05), T(.2, .04), no, up)[0].startswith("R2u")        # A2m not better AND A2 not better than A2m: no attribution
    assert reading_iso(T(.01, .05), T(.2, .04), T(.01, -.03), up)[0].startswith("R2u")
    assert reading_iso(T(.01, .05), T(.01, -.04), yes, up)[0].startswith("R2b")     # A2m significantly LOWER is not R2a
    assert reading_iso(T(.01, -.05), T(.01, -.04), no, down) == ("R3: worse than the conversational reference", 4)
    assert "not stable" in reading_iso(T(.01, .05), T(.2, 0), no, [T(1, .01)] * 2 + [T(1, -.01)] * 2)[0]
    assert "not stable" in reading_iso(T(.01, .05), T(.2, 0), no, flat4)[0] and "not stable" not in reading_iso(T(.2, .05), T(.2, 0), no, down)[0]
    # gate E counts over ALL checked clips: a zero pose (failed extraction) and a wrong frame count are not "under 1 pixel"
    class _C:
        def __init__(self, d): self.d = d
        def has(self, c): return c in self.d
        def get(self, c, key="KPS"): return self.d[c]
    kp = np.concatenate([rng.normal(300, 50, (30, 133, 2)), np.full((30, 133, 1), 0.9)], -1).astype(np.float32)
    row = lambda i, b, e: dict(status="located", comm_split="train", Video_ID=str(i), Video_Name_Clip="video_1_1", Begin_Frame=str(b), End_Frame=str(e))
    ka = keypoint_agreement([row(1, 0, 10), row(2, 10, 20), row(3, 20, 30), row(4, 0, 9)],
                            _C({"1": kp[0:10], "2": np.zeros((10, 133, 3), np.float32), "3": kp[20:30] + np.float32([5, 5, 0]), "4": kp[0:10]}),
                            _C({"video_1_1": kp}))
    assert (ka["checked"], ka["frame_count_differs"], ka["compared"]) == (4, 1, 2) and ka["share_under_1px"] == 0.25, ka
    assert reading_depth(.2, T(.01, .05), True, up).startswith("C1") and reading_depth(.05, T(.01, .05), True, up).startswith("C1")
    assert reading_depth(.01, T(.2, .05), True, up).startswith("C2") and reading_depth(.01, T(.01, -.05), True, up).startswith("C2")
    assert reading_depth(.01, T(.01, .05), True, up) == "C3: depth adds identity to x/y in this probe; a re-proposal, not a training run"
    assert "not stable" in reading_depth(.01, T(.01, .05), True, down) and "VALIDITY CHECK FAILED" in reading_depth(.2, T(1, 0), False, up)
    print("w12f_probe self-test passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spans")
    s.add_argument("--xlsx", required=True); s.add_argument("--train", required=True); s.add_argument("--out", required=True)
    i = sub.add_parser("iso")
    for k in ("spans", "prod_dir", "bank_dir", "located"):
        i.add_argument("--" + k, required=True)
    i.add_argument("--out", default=""); i.add_argument("--gate_only", action="store_true")
    i.add_argument("--mmwl_labels", nargs="*", default=[]); i.add_argument("--mmwl_dirs", nargs="*", default=[])
    d = sub.add_parser("depth")
    for k in ("spans", "prod_dir", "d3_dir"):
        d.add_argument("--" + k, required=True)
    d.add_argument("--out", default=""); d.add_argument("--n_perm", type=int, default=200)
    fa = sub.add_parser("facts")
    fa.add_argument("--located", required=True); fa.add_argument("--spans", required=True); fa.add_argument("--out", default="")
    fa.add_argument("--prod_dir", default="", help="production DWPose clips: also run gate H (the recorded run is reproduced)")
    sub.add_parser("self_test")
    a = ap.parse_args()
    {"spans": cmd_spans, "facts": cmd_facts, "iso": cmd_iso, "depth": cmd_depth, "self_test": lambda _: self_test()}[a.cmd](a)


if __name__ == "__main__":
    main()
