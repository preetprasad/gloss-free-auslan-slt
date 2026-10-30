"""Tracker v2: choose the signer from ALL detections of a whole clip (offline; the live demo is utterance-final too).

v1 (dwpose_extract_worker_tracked.py) commits frame by frame on shoulders alone and, when the track is lost (typically
the signer bending out of view), re-seeds on the largest torso -- often a bystander. v2 instead:
  1. drops weak detections and merges duplicates of one person (detector-box IoU >= DUP_IOU, keep the more confident);
  2. links detections into tracklets across frames by detector-box IoU (Hungarian), bridging gaps of <= MAX_GAP frames;
  3. scores each tracklet over the WHOLE clip -- the crop is built around the signer, so size and centrality -- plus
     hand motion, and takes the best as the signer;
  4. attaches other tracklets only where they are the signer: same person on any frames they share with the path (box
     IoU), or continuing it at a boundary; only their new frames are added;
  5. MASKS every frame where no detection belongs to the signer's path, instead of substituting another person.
Output: KPS (T,133,3) with zeros on masked frames, and CHOSEN (T,) with -1 on masked frames.
"""
import itertools
import numpy as np


def linear_sum_assignment(cost):
    """Exact minimum-cost assignment for the small matrices here (<= ~7 people per frame), without SciPy (absent from the
    Kaya environment). Same contract as scipy.optimize.linear_sum_assignment: min(r, c) pairs, rows ascending."""
    cost = np.asarray(cost, float); r, c = cost.shape
    if r <= c:
        best = min(itertools.permutations(range(c), r), key=lambda p: (sum(cost[i, p[i]] for i in range(r)), p))
        return np.arange(r), np.asarray(best)
    best = min(itertools.permutations(range(r), c), key=lambda p: (sum(cost[p[j], j] for j in range(c)), p))
    rows, cols = np.asarray(best), np.arange(c)
    o = np.argsort(rows)
    return rows[o], cols[o]

CONF, MIN_KP, DUP_IOU, LINK_IOU, MAX_GAP, EXT_IOU = 0.3, 5, 0.6, 0.3, 5, 0.3


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def candidates(sc, db, n):
    """Indices of usable detections in one frame, duplicates of one person merged."""
    ok = [i for i in range(n) if (sc[i, :23] > CONF).sum() >= MIN_KP]
    ok.sort(key=lambda i: -float(sc[i].mean()))
    keep = []
    for i in ok:
        if all(iou(db[i], db[j]) < DUP_IOU for j in keep):
            keep.append(i)
    return keep


def tracklets(DB, cands):
    """Link per-frame candidates into tracklets: list of {frame: detection index}."""
    tracks, live = [], []                      # live: indices into tracks still open
    for t, cs in enumerate(cands):
        live = [k for k in live if t - max(tracks[k]) <= MAX_GAP + 1]
        if live and cs:
            cost = np.array([[1 - iou(DB[max(tracks[k])][tracks[k][max(tracks[k])]], DB[t][c]) for c in cs] for k in live])
            r, c = linear_sum_assignment(cost)
            used = set()
            for i, j in zip(r, c):
                if 1 - cost[i, j] >= LINK_IOU:
                    tracks[live[i]][t] = cs[j]; used.add(j)
            new = [cs[j] for j in range(len(cs)) if j not in used]
        else:
            new = list(cs)
        for d in new:
            tracks.append({t: d}); live.append(len(tracks) - 1)
    return tracks


DEFAULT = dict(w_cov=1.0, w_centre=1.0, w_area=1.0, w_hands=1.0)
# Chosen on the 30 DEV clips only (PREREG_TRACKER_V2 §3): the grid's best wrong-person rate (1.7%) is tied by many
# settings; the simplest -- box area alone, also first in the grid's fixed order -- is fixed here BEFORE held-out.
TUNED = dict(w_cov=0.0, w_centre=0.0, w_area=1.0, w_hands=0.0)


def features(track, KA, SA, DB, W, H, T):
    """Per-tracklet evidence that it is the signer, each on a 0..~1 scale, averaged over the tracklet's frames:
    coverage (fraction of the clip), horizontal centrality (the crop is built around the signer), box area as a
    fraction of the frame, and hand motion per frame relative to body height (the signer signs)."""
    cen, area, mot = [], [], []; prev = None
    for t in sorted(track):
        d = track[t]; b = DB[t, d]; bh = max(1.0, b[3] - b[1])
        area.append((b[2] - b[0]) * bh / float(W * H))
        cen.append(1.0 - min(1.0, abs((b[0] + b[2]) / 2.0 - W / 2.0) / (W / 2.0)))
        hands, hs = KA[t, d, 91:133], SA[t, d, 91:133] > CONF
        if prev is not None:
            m = hs & prev[1]
            if m.any():
                mot.append(min(1.0, 10.0 * float(np.linalg.norm(hands[m] - prev[0][m], axis=1).mean()) / bh))
        prev = (hands, hs)
    return dict(cov=len(track) / float(T), centre=float(np.mean(cen)), area=float(np.mean(area)),
                hands=float(np.mean(mot)) if mot else 0.0)


def score(track, KA, SA, DB, W, H, T, w=DEFAULT):
    f = features(track, KA, SA, DB, W, H, T)
    return w["w_cov"] * f["cov"] + w["w_centre"] * f["centre"] + w["w_area"] * f["area"] + w["w_hands"] * f["hands"]


def choose(d, w=DEFAULT):
    """d: an all-detections npz (dict-like). Returns (KPS (T,133,3), CHOSEN (T,))."""
    KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]
    W, H = (int(x) for x in d["WH"])
    T = len(ND)
    cands = [candidates(SA[t], DB[t], int(ND[t])) for t in range(T)]
    tr = tracklets(DB, cands)
    chosen = np.full(T, -1, np.int16)
    if tr:
        best = max(range(len(tr)), key=lambda k: score(tr[k], KA, SA, DB, W, H, T, w))
        path = dict(tr[best])
        # extend with other tracklets that continue the signer's path and do not overlap it in time
        others = sorted((k for k in range(len(tr)) if k != best), key=lambda k: -score(tr[k], KA, SA, DB, W, H, T, w))
        for k in others:
            shared = [t for t in tr[k] if t in path]
            fresh = {t: v for t, v in tr[k].items() if t not in path}
            if not fresh:
                continue
            if shared:     # overlaps the path: join only if it is the same person on the shared frames
                same = float(np.mean([iou(DB[t, path[t]], DB[t, tr[k][t]]) for t in shared])) >= EXT_IOU
            else:          # disjoint in time: join only if it continues the path at a boundary
                t0, t1 = min(fresh), max(fresh)
                before = [t for t in path if t < t0]; after = [t for t in path if t > t1]
                links = []
                if before: tb = max(before); links.append(iou(DB[tb, path[tb]], DB[t0, fresh[t0]]))
                if after: ta = min(after); links.append(iou(DB[ta, path[ta]], DB[t1, fresh[t1]]))
                same = bool(links) and max(links) >= EXT_IOU
            if same:
                path.update(fresh)
        for t, det in path.items():
            chosen[t] = det
    KPS = np.zeros((T, 133, 3), np.float32)
    for t in range(T):
        if chosen[t] >= 0:
            KPS[t, :, :2] = KA[t, chosen[t]]; KPS[t, :, 2] = SA[t, chosen[t]]
    return KPS, chosen


def _self_test():
    """Two people; the signer (centre, large, moving hands) bends out of detection for frames 10-14 while a bystander
    (edge, smaller) stays visible. v2 must follow the signer and MASK 10-14, never switch to the bystander."""
    rng = np.random.default_rng(0); T = 30; W, H = 600, 900
    KA = np.zeros((T, 3, 133, 2), np.float32); SA = np.zeros((T, 3, 133), np.float32); DB = np.zeros((T, 3, 4), np.float32)
    ND = np.zeros(T, np.int16)
    for t in range(T):
        dets = []
        if not 10 <= t < 15:
            box = np.array([200, 100, 420, 850], np.float32)
            kp = rng.uniform([200, 100], [420, 850], (133, 2)).astype(np.float32)
            kp[91:133] += rng.normal(0, 8, (42, 2))                 # moving hands
            dets.append((kp, box)); dets.append((kp + 1.0, box + 2))  # plus a duplicate of the signer
        bbox = np.array([0, 300, 120, 850], np.float32)
        dets.insert(0, (rng.uniform([0, 300], [120, 850], (133, 2)).astype(np.float32), bbox))  # bystander is index 0
        for i, (k, b) in enumerate(dets):
            KA[t, i] = k; SA[t, i] = 0.9; DB[t, i] = b
        ND[t] = len(dets)
    _, ch = choose(dict(KP_ALL=KA, SC_ALL=SA, DET_BOX=DB, NDET=ND, WH=np.array([W, H])))
    assert all(ch[t] in (1, 2) for t in range(T) if not 10 <= t < 15), ch
    assert all(ch[t] == -1 for t in range(10, 15)), ch
    print("self-test OK: follows the signer, masks the lost frames, never switches to the bystander")


if __name__ == "__main__":
    _self_test()
