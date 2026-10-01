"""Parity fixture for demo/tests/engine_tracklets.test.mjs: random all-detection clips from the same 32-bit PRNG the
JS test uses (mulberry32), built to hit the edges tracker_v2 has to get right -- identical boxes (assignment ties),
more live tracks than detections, near-duplicates, zero-area boxes, integer coordinates, keypoint scores exactly at
float32(0.3), gaps, and clips with no usable detection. Stores the seed, a sha256 of each clip's float32 inputs and
tracker_v2.choose for TUNED and DEFAULT. Usage: make_tracklets_fixture.py OUT.json"""
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
import tracker_v2 as v2

K, NMAX, SEED, CLIPS = 133, 8, 20260930, 40


def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def gen_clip(r):
    """Mirrors genClip() in engine_tracklets.test.mjs statement for statement."""
    T = 8 + int(r() * 40); W = 320 + int(r() * 960); H = 240 + int(r() * 720)
    P = 1 + int(r() * 5); quant = r() < 0.4
    actors = []
    for p in range(P):
        a = dict(x=r() * W * 0.8, y=r() * H * 0.5, w=20 + r() * W * 0.4, h=40 + r() * H * 0.5, vis=0.4 + r() * 0.6,
                 step=r() * 20)
        if p > 0 and r() < 0.2:
            a.update({k: actors[p - 1][k] for k in ("x", "y", "w", "h")})
        actors.append(a)
    KA = np.zeros((T, NMAX, K, 2), np.float32); SA = np.zeros((T, NMAX, K), np.float32)
    DB = np.zeros((T, NMAX, 4), np.float32); ND = np.zeros(T, np.int16)
    for t in range(T):
        boxes = []
        for a in actors:
            if r() < a["vis"]:
                a["x"] += (r() - 0.5) * a["step"]; a["y"] += (r() - 0.5) * a["step"]
                b = [a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]]
                boxes.append([float(int(v)) for v in b] if quant else b)
        if boxes and r() < 0.15:
            s = boxes[int(r() * len(boxes))]; o = (r() - 0.5) * 6
            boxes.append([s[0] + o, s[1] + o, s[2] + o, s[3] + o])
        if r() < 0.05:
            x, y = r() * W, r() * H; boxes.append([x, y, x, y + r() * 100])
        if r() < 0.1:
            x, y = r() * W, r() * H; boxes.append([x, y, x + 1 + r() * 30, y + 1 + r() * 30])
        keys = [r() for _ in boxes]
        boxes = [boxes[i] for i in sorted(range(len(boxes)), key=lambda i: (keys[i], i))][:NMAX]
        ND[t] = len(boxes)
        for j, b in enumerate(boxes):
            DB[t, j] = b
            for k in range(K):
                SA[t, j, k] = r()
                KA[t, j, k, 0] = b[0] + r() * (b[2] - b[0]); KA[t, j, k, 1] = b[1] + r() * (b[3] - b[1])
            u = r()
            if u < 0.1:
                for k in range(23): SA[t, j, k] = r() * 0.35
            elif u < 0.15:
                for k in range(23): SA[t, j, k] = 0.3 if k % 2 else 0.9
    return dict(KP_ALL=KA, SC_ALL=SA, DET_BOX=DB, NDET=ND, WH=np.array([W, H], np.int32))


def main():
    stats = dict(assign=0, ties=0, rows_gt_cols=0)
    orig = v2.linear_sum_assignment

    def counting(cost):
        cost = np.asarray(cost, float); r, c = cost.shape
        stats["assign"] += 1; stats["rows_gt_cols"] += int(r > c)
        sums = sorted(sum(cost[i, p[i]] for i in range(r)) for p in itertools.permutations(range(c), r)) if r <= c else \
            sorted(sum(cost[p[j], j] for j in range(c)) for p in itertools.permutations(range(r), c))
        stats["ties"] += int(len(sums) > 1 and sums[0] == sums[1])
        return orig(cost)
    v2.linear_sum_assignment = counting
    r = mulberry32(SEED); first = [r() for _ in range(4)]
    clips = []
    for _ in range(CLIPS):
        d = gen_clip(r)
        h = hashlib.sha256()
        for key in ("KP_ALL", "SC_ALL", "DET_BOX"):
            h.update(np.ascontiguousarray(d[key]).tobytes())
        exp = {w: v2.choose(d, getattr(v2, w))[1].tolist() for w in ("TUNED", "DEFAULT")}
        clips.append(dict(T=len(d["NDET"]), W=int(d["WH"][0]), H=int(d["WH"][1]), sha256=h.hexdigest(), **exp))
    frames = sum(c["T"] for c in clips)
    masked = sum(int(x < 0) for c in clips for x in c["TUNED"])
    differ = sum(c["TUNED"] != c["DEFAULT"] for c in clips)
    stats.update(frames=frames, masked_tuned=masked, clips_tuned_ne_default=differ)
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=SEED, nmax=NMAX, first=first, numpy=np.__version__, stats=stats, clips=clips),
              open(sys.argv[1], "w"))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
