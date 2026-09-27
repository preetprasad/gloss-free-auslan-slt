"""DWPose extraction that keeps EVERY detected person, so trackers can be built and compared offline.

WHY. dwpose_extract_worker.py keeps detection 0 per frame; dwpose_extract_worker_tracked.py keeps the one person its
rule chose. Both throw the other people away, so every new tracker would need a new extraction. The blind audit of the
v1 tracker (results/switch_video_89_193/README.md, 2026-09-25) found it right in 91% of overrides, wrong mainly when the
signer bends out of view, and 14% of overrides swapping between duplicate detections of the signer. Fixing those needs
all detections.

WHAT IT WRITES (superset of the v1 tracked format, so every existing loader still works):
  KPS (T,133,3), BBOX (T,4), CHOSEN (T,), NDET (T,)  -- byte-for-byte the v1 tracker's outputs (same functions, imported)
  KP_ALL (T,N,133,2), SC_ALL (T,N,133)                -- every detection's keypoints and scores, N = max NDET, zero-padded
  DET_BOX (T,N,4)                                      -- the detector's own boxes (x0, y0, x1, y1), zero-padded; when
                                                          the detector finds nobody, rtmlib runs pose on the whole
                                                          frame and the box is [0, 0, W, H] (job 31508 crashed on this)
  WH (2,)                                              -- frame width and height
The detector and pose model are called in exactly the order rtmlib's Wholebody.__call__ uses (det, then pose on its
boxes), so the keypoints are those the v1 worker saw.
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dwpose_extract_worker_tracked as v1        # sets the same ORT/OMP thread caps on import
import numpy as np, cv2


def extract_clip(model, path):
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    frames, boxes, chosen, ndet, allkp, allsc, alldb = [], [], [], [], [], [], []
    prev_mid = prev_wid = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        try:
            db = np.asarray(model.det_model(fr), np.float32).reshape(-1, 4)
            kp, sc = model.pose_model(fr, bboxes=db)
            if len(db) == 0:
                # rtmlib's RTMPose.__call__ runs the pose model on the whole frame when the detector returns nothing
                # (bboxes = [[0, 0, w, h]]); record that box so every skeleton has the box it was estimated in.
                db = np.asarray([[0, 0, fr.shape[1], fr.shape[0]]], np.float32)
        except Exception:
            db, kp, sc = np.zeros((0, 4), np.float32), None, None
        if sc is None or len(sc) == 0:
            frames.append(np.zeros((133, 3), np.float32)); boxes.append(np.zeros(4, np.float32))
            chosen.append(-1); ndet.append(0)
            allkp.append(np.zeros((0, 133, 2), np.float32)); allsc.append(np.zeros((0, 133), np.float32))
            alldb.append(np.zeros((0, 4), np.float32))
            continue
        kp = np.asarray(kp, np.float32); sc = np.asarray(sc, np.float32)
        assert len(db) == len(kp), (len(db), len(kp))
        i, _ = v1.track_index(kp, sc, prev_mid, prev_wid, (W, H))
        k, s = kp[i], sc[i]
        mid, wid, ok_t = v1._torso(k, s)
        if ok_t:
            prev_mid, prev_wid = mid, wid
        frames.append(np.concatenate([k, s[:, None]], axis=1)); boxes.append(v1.bbox_from(k, s))
        chosen.append(i); ndet.append(len(kp)); allkp.append(kp); allsc.append(sc); alldb.append(db)
    cap.release()
    if not frames:
        frames = [np.zeros((133, 3), np.float32)]; boxes = [np.zeros(4, np.float32)]; chosen = [-1]; ndet = [0]
        allkp = [np.zeros((0, 133, 2), np.float32)]; allsc = [np.zeros((0, 133), np.float32)]; alldb = [np.zeros((0, 4), np.float32)]
    T, N = len(frames), max(1, max(ndet))
    KA = np.zeros((T, N, 133, 2), np.float32); SA = np.zeros((T, N, 133), np.float32); DB = np.zeros((T, N, 4), np.float32)
    for t in range(T):
        n = len(allkp[t]); KA[t, :n] = allkp[t]; SA[t, :n] = allsc[t]; DB[t, :n] = alldb[t]
    return dict(KPS=np.stack(frames).astype(np.float32), BBOX=np.stack(boxes).astype(np.float32),
                CHOSEN=np.asarray(chosen, np.int16), NDET=np.asarray(ndet, np.int16),
                KP_ALL=KA, SC_ALL=SA, DET_BOX=DB, WH=np.asarray([W, H], np.int32))


def main():
    from rtmlib import Wholebody
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--device", default=os.environ.get("DWPOSE_DEVICE", "cpu"))
    a = ap.parse_args()
    RGB, OUT = os.environ["RGB_DIR"], os.environ["OUT_DIR"]
    ONLY = os.environ.get("CLIP_LIST", "")
    _SUFFIX = os.environ.get("CLIP_SUFFIX", "_signer.mp4")
    os.makedirs(OUT, exist_ok=True)
    keep = set(l.strip() for l in open(ONLY) if l.strip()) if ONLY else None
    mp4s = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
    cid_of = lambda v: (os.path.basename(v)[:-len(_SUFFIX)] if os.path.basename(v).endswith(_SUFFIX)
                        else os.path.splitext(os.path.basename(v))[0])
    if keep is not None:
        mp4s = [v for v in mp4s if cid_of(v) in keep]
    mine = [v for i, v in enumerate(mp4s) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}] {len(mine)} clips (of {len(mp4s)} selected)", flush=True)
    model = Wholebody(mode=a.mode, backend="onnxruntime", device=a.device)
    done = skip = 0
    for v in mine:
        cid = cid_of(v); outp = os.path.join(OUT, cid + ".npz")
        if os.path.exists(outp):
            skip += 1; continue
        d = extract_clip(model, v)
        tmp = os.path.join(OUT, cid + ".writing")
        np.savez_compressed(tmp, **d); os.replace(tmp + ".npz", outp)
        done += 1
        if done % 50 == 0:
            print(f"[shard {a.shard}] done={done} skip={skip} / {len(mine)}", flush=True)
    print(f"[shard {a.shard}] FINISHED done={done} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
