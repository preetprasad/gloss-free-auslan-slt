"""Classical-CV per-clip analysis of the raw signer videos — NO neural nets.

Characterises the ground-truth difficulty of every clip directly from pixels, so
pose/translation failures can be attributed to the input signal rather than the
estimator. Shardable + resumable. One streaming pass per clip (low memory).

Per-clip metrics (aggregated over the clip):
  geometry :  width, height, fps, n_frames, duration_s, megapixels
  exposure :  brightness_mean, contrast_mean, underexposed_frac, overexposed_frac
  sharpness:  lap_var_mean, lap_var_p10, blurry_frame_frac   (Laplacian focus metric)
  clutter  :  edge_density_mean                              (Canny)
  motion   :  framediff_mean, framediff_p90, flow_mag_mean, flow_mag_p95  (Farneback)
  skin/hand:  skin_frac_mean, n_skin_blobs_mean, hand_blob_area_mean,
              largest_blob_frac(face), hand_blobs_frac, hands_together_frac
  person   :  faces_mean, face_det_frac, multiperson_frac, face_size_frac (Haar)
  temporal :  black_frame_frac, static_frame_frac, scene_cuts
  flags    :  tiny_hands, high_blur, high_motion, occlusion, multiperson, dark
"""
import os, sys, glob, csv, argparse
import numpy as np, cv2
cv2.setNumThreads(1)

def _face_cascade():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "haarcascade_frontalface_default.xml"),
              cv2.data.haarcascades + "haarcascade_frontalface_default.xml"):
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    return None
FACE = _face_cascade()   # None if unavailable -> face metrics gracefully skipped
SKIN_LO = np.array([0, 133, 77], np.uint8)      # YCrCb skin range (lighting-robust-ish)
SKIN_HI = np.array([255, 173, 127], np.uint8)
KERNEL = np.ones((3, 3), np.uint8)


def skin_mask(bgr):
    m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb), SKIN_LO, SKIN_HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)


def analyze(path, sample=24, scale=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, N // sample) if N > 0 else 5
    area = 1.0                                    # downscaled-frame area, set per frame

    fd = []                                   # frame-diff on every consecutive pair
    bright=[]; contrast=[]; under=[]; over=[]; black=[]; lapv=[]; edged=[]
    skinf=[]; nblob=[]; largest=[]; handblob=[]; together=[]; faces=[]; facesize=[]
    flow=[]; cuts=0
    prev=None; prev_s=None; prev_hist=None; i=0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h0, w0 = fr.shape[:2]                      # optional downscale (0 = full res, used on Kaya)
        if scale and scale > 0 and max(h0, w0) > scale:
            s = scale / max(h0, w0)
            fr = cv2.resize(fr, (max(1, int(w0 * s)), max(1, int(h0 * s))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        area = float(g.shape[0] * g.shape[1])
        if prev is not None:
            fd.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = g
        if i % step == 0:                     # heavy metrics on sampled frames only
            bright.append(float(g.mean())); contrast.append(float(g.std()))
            under.append(float((g < 16).mean())); over.append(float((g > 240).mean()))
            black.append(1.0 if g.mean() < 8 else 0.0)
            lapv.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            edged.append(float((cv2.Canny(g, 80, 160) > 0).mean()))
            sm = skin_mask(fr); skinf.append(float(sm.mean() / 255.0))
            cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = sorted((cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > area * 8e-4), reverse=True)
            nblob.append(len(blobs))
            largest.append(blobs[0] / area if blobs else 0.0)
            hb = blobs[1:3]; handblob.append(sum(hb) / area if hb else 0.0)
            together.append(1.0 if (len(blobs) <= 1 and (blobs[0] / area if blobs else 0) > 0.06) else 0.0)
            fc = FACE.detectMultiScale(g, 1.2, 5, minSize=(30, 30)) if FACE is not None else []
            faces.append(len(fc)); facesize.append(max((w*h for (x,y,w,h) in fc), default=0) / area)
            if prev_s is not None:
                f = cv2.calcOpticalFlowFarneback(prev_s, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow.append(float(np.linalg.norm(f, axis=2).mean()))
            hist = cv2.calcHist([g], [0], None, [64], [0, 256])
            if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.5:
                cuts += 1
            prev_s = g; prev_hist = hist
        i += 1
    cap.release()
    n = i
    if n == 0 or not bright:
        return None
    pct = lambda a, p: float(np.percentile(a, p)) if len(a) else 0.0
    r = dict(
        clip=os.path.basename(path).replace("_signer.mp4", ""),
        width=W, height=H, fps=round(fps, 2), n_frames=n,
        duration_s=round(n / fps, 2) if fps > 0 else 0.0, megapixels=round(W * H / 1e6, 3),
        brightness_mean=round(float(np.mean(bright)), 1), contrast_mean=round(float(np.mean(contrast)), 1),
        underexposed_frac=round(float(np.mean(under)), 3), overexposed_frac=round(float(np.mean(over)), 3),
        lap_var_mean=round(float(np.mean(lapv)), 1), lap_var_p10=round(pct(lapv, 10), 1),
        blurry_frame_frac=round(float(np.mean(np.array(lapv) < 100)), 3),
        edge_density_mean=round(float(np.mean(edged)), 4),
        framediff_mean=round(float(np.mean(fd)), 2) if fd else 0.0,
        framediff_p90=round(pct(fd, 90), 2) if fd else 0.0,
        flow_mag_mean=round(float(np.mean(flow)), 3) if flow else 0.0,
        flow_mag_p95=round(pct(flow, 95), 3) if flow else 0.0,
        skin_frac_mean=round(float(np.mean(skinf)), 4),
        n_skin_blobs_mean=round(float(np.mean(nblob)), 2),
        largest_blob_frac=round(float(np.mean(largest)), 4),
        hand_blob_area_mean=round(float(np.mean(handblob)), 5),
        hand_blobs_frac=round(float(np.mean(np.array(nblob) >= 2)), 3),
        hands_together_frac=round(float(np.mean(together)), 3),
        faces_mean=round(float(np.mean(faces)), 2),
        face_det_frac=round(float(np.mean(np.array(faces) >= 1)), 3),
        multiperson_frac=round(float(np.mean(np.array(faces) >= 2)), 3),
        face_size_frac=round(float(np.mean(facesize)), 4),
        black_frame_frac=round(float(np.mean(black)), 3),
        static_frame_frac=round(float(np.mean(np.array(fd) < 1.0)), 3) if fd else 0.0,
        scene_cuts=cuts,
    )
    r["hand_px_mean"] = int(r["hand_blob_area_mean"] * W * H)     # absolute hand size (native px)
    r["face_px_mean"] = int(r["face_size_frac"] * W * H)
    r["flag_tiny_hands"] = int(r["hand_blob_area_mean"] < 0.004)
    r["flag_high_blur"] = int(r["blurry_frame_frac"] > 0.5)
    r["flag_high_motion"] = int(r["framediff_p90"] > 25)
    r["flag_occlusion"] = int(r["hands_together_frac"] > 0.3)
    r["flag_multiperson"] = int(r["multiperson_frac"] > 0.2)
    r["flag_dark"] = int(r["brightness_mean"] < 40)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--scale", type=int, default=0)   # 0 = full resolution (Kaya); 320 = downscale (Mac)
    a = ap.parse_args()
    RGB = os.environ.get("RGB_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/rgb_clips"))
    OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/Honours/results/dwpose_experiment/classical_cv"))
    os.makedirs(OUT, exist_ok=True)
    mp4s = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
    mine = [v for k, v in enumerate(mp4s) if k % a.nshards == a.shard]
    out_csv = os.path.join(OUT, f"shard_{a.shard}.csv")
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {row["clip"] for row in csv.DictReader(f)}
    fields = None; n = 0
    fh = open(out_csv, "a", newline="")
    writer = None
    for v in mine:
        cid = os.path.basename(v).replace("_signer.mp4", "")
        if cid in done:
            continue
        try:
            r = analyze(v, a.sample, a.scale)
        except Exception as e:
            r = None
        if r is None:
            continue
        if writer is None:
            fields = list(r.keys())
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not done and os.path.getsize(out_csv) == 0:
                writer.writeheader()
        writer.writerow(r); n += 1
        if n % 50 == 0:
            fh.flush(); print(f"[shard {a.shard}] {n}/{len(mine)}", flush=True)
    fh.close()
    print(f"[shard {a.shard}] FINISHED wrote {n} clips", flush=True)


if __name__ == "__main__":
    main()
