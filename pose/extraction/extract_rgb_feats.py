"""RGB fusion — Stage A: per-clip RGB features.
For each clip, sample frames; for each frame make 3 crops localised by the DWPose
keypoints (full signer, left hand, right hand); encode each with a frozen DINOv2-small
backbone; concat -> (T_sampled, 3*384). Shardable + resumable.
Output: rgb_<clip>.npz key FEATURES.
"""
import os, glob, argparse, numpy as np, cv2, torch
os.environ.setdefault("OMP_NUM_THREADS", "4")
from transformers import AutoModel

DINO = "facebook/dinov2-small"          # 384-d CLS
IMN = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def crop_box(xy, cf, thr=0.3, pad=1.4):
    m = cf > thr
    if m.sum() < 3:
        return None
    p = xy[m]
    x0, y0 = p.min(0); x1, y1 = p.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * pad / 2 + 1
    return int(cx - s), int(cy - s), int(cx + s), int(cy + s)


def get_crop(frame, box):
    if box is None:
        return None
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (224, 224))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--frames", type=int, default=32)
    a = ap.parse_args()
    RGB = os.environ.get("RGB_DIR", "/scratch/pmc099/pprasad/dwpose/rgb_clips")
    DW = os.environ.get("DWPOSE_DIR", "/scratch/pmc099/pprasad/dwpose/dwpose_clips")
    OUT = os.environ.get("OUT_DIR", "/scratch/pmc099/pprasad/dwpose/rgb_feats")
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO).to(dev).eval()
    mean = torch.tensor(IMN[0], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMN[1], device=dev).view(1, 3, 1, 1)

    mp4s = sorted(glob.glob(RGB + "/**/*.mp4", recursive=True))
    mine = [v for i, v in enumerate(mp4s) if i % a.nshards == a.shard]
    done = skip = 0
    for v in mine:
        cid = os.path.basename(v).replace("_signer.mp4", "")
        outp = os.path.join(OUT, f"rgb_{cid}.npz")
        if os.path.exists(outp):
            skip += 1; continue
        dwp = os.path.join(DW, cid + ".npz")
        if not os.path.exists(dwp):
            continue
        kps = np.load(dwp)["KPS"].astype(np.float32)       # (T,133,3)
        cap = cv2.VideoCapture(v); frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            continue
        T = min(len(frames), kps.shape[0])
        idx = np.linspace(0, T - 1, min(a.frames, T)).round().astype(int)
        crops, slots = [], []
        for i in idx:
            fr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            k = kps[i]; xy, cf = k[:, :2], k[:, 2]
            boxes = [crop_box(xy, cf), crop_box(xy[91:112], cf[91:112]), crop_box(xy[112:133], cf[112:133])]
            for b in boxes:
                c = get_crop(fr, b)
                slots.append(c is not None)
                crops.append(c if c is not None else np.zeros((224, 224, 3), np.uint8))
        t = torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        t = (t - mean) / std
        embs = []
        with torch.no_grad():
            for j in range(0, t.shape[0], 128):
                embs.append(model(pixel_values=t[j:j + 128]).pooler_output.cpu())
        emb = torch.cat(embs).numpy()
        emb[~np.array(slots)] = 0.0
        emb = emb.reshape(len(idx), -1).astype(np.float32)   # (T_sampled, 3*384)
        tmp = os.path.join(OUT, f"rgb_{cid}.writing")
        np.savez_compressed(tmp, FEATURES=emb)
        os.replace(tmp + ".npz", outp)
        done += 1
        if done % 20 == 0:
            print(f"[shard {a.shard}] done={done} skip={skip} / {len(mine)}", flush=True)
    print(f"[shard {a.shard}] FINISHED done={done} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
