"""DWPose extraction that follows ONE person across a clip.

WHY THIS EXISTS. The original worker (dwpose_extract_worker.py:64) keeps `kp[0]` -- the first
detection the model happens to return -- independently for every frame, with no tracking and no
association of a person across frames. Auslan-Daily Communication is multi-person conversation and
the shipped signer crops still contain bystanders, so in a frame with more than one person the
saved skeleton can belong to someone else than in the previous frame. Measured over 2,128 locally
held clips (CLAUDE.md 2026-09-22b): 25.2% of clips contain at least one such discontinuity, 2.31%
of all frame transitions, and of 552 sampled events 87.5% were identity switches rather than camera
cuts. The official validation and test splits are affected too, so this is a defect in the
evaluation inputs as well as the training inputs.

WHAT THIS DOES INSTEAD.
  frame 0   seed on the detection that is most plausibly the signer: largest torso, tie-broken
            toward the frame centre, and requiring confident shoulders.
  frame t   choose the detection whose shoulder midpoint is nearest the previous accepted one,
            in units of that person's shoulder width. If the nearest is further than GATE shoulder
            widths, the track is considered lost and re-seeded by the frame-0 rule -- so a genuine
            camera cut re-seeds rather than dragging the track onto the wrong person.
  always    write a REAL bounding box from the accepted person's confident keypoints. The original
            wrote an all-zero dummy, which is why coord_norm="bbox" is degenerate on these npz
            (CLAUDE.md 2026-08-22); a real box re-opens signer-relative normalisation for free.

IT ALSO RECORDS WHAT IT CHANGED. `CHOSEN` is the detection index taken at each frame and `NDET` the
number of detections available, so the corrected corpus carries its own evidence: frames where
CHOSEN != 0 are exactly the frames the old code got wrong, and they can be counted without
re-running anything.

Output is a superset of the old format -- KPS (T,133,3) and BBOX (T,4) keep their names, shapes and
dtypes -- so every existing loader reads these files unchanged.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys, glob, argparse, numpy as np, cv2
cv2.setNumThreads(1)

import onnxruntime as ort
_ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))
_orig_session = ort.InferenceSession
def _capped_session(*a, **k):
    so = k.get("sess_options") or ort.SessionOptions()
    so.intra_op_num_threads = _ORT_THREADS
    so.inter_op_num_threads = 1
    k["sess_options"] = so
    return _orig_session(*a, **k)
ort.InferenceSession = _capped_session

CONF = 0.3          # keypoint confidence floor, matching slt/data.py's reading of these files
GATE = 1.6          # re-seed if the nearest candidate is further than this many shoulder widths
L_SH, R_SH = 5, 6   # COCO-WholeBody shoulder indices


def _torso(k, s):
    """Shoulder midpoint, shoulder width and a validity flag for one detection."""
    if s[L_SH] < CONF or s[R_SH] < CONF:
        return None, None, False
    mid = (k[L_SH] + k[R_SH]) / 2.0
    wid = float(np.linalg.norm(k[L_SH] - k[R_SH]))
    return mid, (wid if wid > 1e-3 else None), wid > 1e-3


def seed_index(kp, sc, frame_wh):
    """Frame-0 rule: the largest confident torso, tie-broken toward the centre of the frame."""
    W, H = frame_wh
    best, best_score = 0, -1e18
    for i in range(len(kp)):
        mid, wid, ok = _torso(kp[i], sc[i])
        if not ok:
            continue
        centrality = 1.0 - abs(mid[0] - W / 2.0) / (W / 2.0)
        score = wid * (1.0 + 0.35 * centrality) + 0.02 * float(sc[i].mean())
        if score > best_score:
            best, best_score = i, score
    return best


def track_index(kp, sc, prev_mid, prev_wid, frame_wh):
    """Frame-t rule: nearest confident torso to the previous accepted one, else re-seed."""
    if prev_mid is None:
        return seed_index(kp, sc, frame_wh), True
    best, best_d = None, 1e18
    for i in range(len(kp)):
        mid, wid, ok = _torso(kp[i], sc[i])
        if not ok:
            continue
        d = float(np.linalg.norm(mid - prev_mid)) / max(prev_wid, 1e-3)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return seed_index(kp, sc, frame_wh), True
    if best_d > GATE:                       # track lost: a cut, or the signer left frame
        return seed_index(kp, sc, frame_wh), True
    return best, False


def bbox_from(k, s):
    """A real xywh box around the confident keypoints, in the release's own convention."""
    m = s > CONF
    if m.sum() < 4:
        return np.zeros(4, np.float32)
    pts = k[m]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([x0, y0, x1 - x0, y1 - y0], np.float32)


def main():
    from rtmlib import Wholebody
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--device", default=os.environ.get("DWPOSE_DEVICE", "cpu"))
    a = ap.parse_args()

    RGB = os.environ.get("RGB_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/rgb_clips"))
    OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/Honours/results/dwpose_tracked"))
    ONLY = os.environ.get("CLIP_LIST", "")           # optional newline-separated clip ids
    os.makedirs(OUT, exist_ok=True)
    keep = set(l.strip() for l in open(ONLY) if l.strip()) if ONLY else None

    _SUFFIX = os.environ.get("CLIP_SUFFIX", "_signer.mp4")
    mp4s = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
    def cid_of(v):
        c = os.path.basename(v)
        return c[:-len(_SUFFIX)] if c.endswith(_SUFFIX) else os.path.splitext(c)[0]
    if keep is not None:
        mp4s = [v for v in mp4s if cid_of(v) in keep]
    mine = [v for i, v in enumerate(mp4s) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}] {len(mine)} clips (of {len(mp4s)} selected)", flush=True)

    model = Wholebody(mode=a.mode, backend="onnxruntime", device=a.device)
    done = skip = 0
    tot_frames = tot_changed = tot_reseed = 0
    for v in mine:
        cid = cid_of(v)
        outp = os.path.join(OUT, cid + ".npz")
        if os.path.exists(outp):
            skip += 1; continue
        cap = cv2.VideoCapture(v)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        frames, boxes, chosen, ndet = [], [], [], []
        prev_mid = prev_wid = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            try:
                kp, sc = model(fr)
            except Exception:
                kp, sc = None, None
            if sc is None or len(sc) == 0:
                frames.append(np.zeros((133, 3), np.float32)); boxes.append(np.zeros(4, np.float32))
                chosen.append(-1); ndet.append(0)
                continue
            kp = np.asarray(kp, np.float32); sc = np.asarray(sc, np.float32)
            i, reseeded = track_index(kp, sc, prev_mid, prev_wid, (W, H))
            k, s = kp[i], sc[i]
            mid, wid, ok_t = _torso(k, s)
            if ok_t:
                prev_mid, prev_wid = mid, wid
            frames.append(np.concatenate([k, s[:, None]], axis=1))
            boxes.append(bbox_from(k, s))
            chosen.append(i); ndet.append(len(kp))
            tot_frames += 1; tot_changed += (i != 0); tot_reseed += reseeded
        cap.release()
        if not frames:
            frames = [np.zeros((133, 3), np.float32)]; boxes = [np.zeros(4, np.float32)]
            chosen = [-1]; ndet = [0]
        tmp = os.path.join(OUT, cid + ".writing")
        np.savez_compressed(tmp,
                            KPS=np.stack(frames).astype(np.float32),
                            BBOX=np.stack(boxes).astype(np.float32),
                            CHOSEN=np.asarray(chosen, np.int16),
                            NDET=np.asarray(ndet, np.int16))
        os.replace(tmp + ".npz", outp)
        done += 1
        if done % 25 == 0:
            print(f"[shard {a.shard}] done={done} skip={skip} / {len(mine)} | "
                  f"frames where the old code took a different person: "
                  f"{tot_changed}/{tot_frames} ({100*tot_changed/max(1,tot_frames):.2f}%)", flush=True)
    print(f"[shard {a.shard}] FINISHED done={done} skip={skip} | frames {tot_frames} | "
          f"detection index != 0 on {tot_changed} ({100*tot_changed/max(1,tot_frames):.2f}%) | "
          f"re-seeds {tot_reseed}", flush=True)


if __name__ == "__main__":
    main()
