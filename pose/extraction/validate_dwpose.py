"""Cheap validation of the DWPose lever: run a modern RTMPose/RTMW whole-body
estimator (via rtmlib) on the local sample signer video and compare hand-keypoint
confidence against the shipped AlphaPose HALPE tensor for the same clip.

If DWPose hand confidence is materially above the shipped ~0.14, the full
re-extraction (17.4 GB RGB fetch + retrain) is justified; if not, we save hours.
"""
import os, glob, time, numpy as np, cv2

VID = os.path.expanduser("~/Honours/results/pose_layer_sample_videos/video_10_0_signer.mp4")
CLIP = "video_10_0"
OUT = os.path.expanduser("~/Honours/results/dwpose_experiment"); os.makedirs(OUT, exist_ok=True)

# shipped AlphaPose hand confidence for the same clip (HALPE hands = 94..135)
shipped = None
for c in glob.glob(os.path.expanduser(f"~/Honours/*/clips/{CLIP}.npz")) + \
         glob.glob(os.path.expanduser(f"~/Honours/**/{CLIP}.npz"), recursive=True):
    try:
        k = np.load(c)["KPS"].astype(np.float32)
        shipped = float(k[:, 94:136, 2].mean()); shipped_body = float(k[:, 0:26, 2].mean())
        print(f"shipped AlphaPose {CLIP}: frames={k.shape[0]} hand_conf={shipped:.3f} body_conf={shipped_body:.3f}")
        break
    except Exception:
        pass
if shipped is None:
    print("shipped AlphaPose tensor for this clip not found locally (will still report DWPose)")

from rtmlib import Wholebody
print("loading RTMPose/RTMW wholebody (downloads ONNX on first run)...", flush=True)
model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")

cap = cv2.VideoCapture(VID)
hand_conf, body_conf, nframes, t0 = [], [], 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    nframes += 1
    kpts, scores = model(frame)          # (P,133,2), (P,133)
    if scores is None or len(scores) == 0:
        continue
    s = np.asarray(scores[0])            # signer = first/most-confident person
    # COCO-WholeBody: body 0..16, left hand 91..111, right hand 112..132
    hand_conf.append(float(np.mean(s[91:133])))
    body_conf.append(float(np.mean(s[0:17])))
cap.release()
fps = nframes / max(1e-6, time.time() - t0)
dw_hand = float(np.mean(hand_conf)) if hand_conf else 0.0
dw_body = float(np.mean(body_conf)) if body_conf else 0.0
print(f"\n=== RESULT on {CLIP} ({nframes} frames, {fps:.1f} fps CPU) ===")
print(f"  DWPose/RTMW  hand_conf = {dw_hand:.3f}   body_conf = {dw_body:.3f}")
if shipped is not None:
    print(f"  shipped AlphaPose hand_conf = {shipped:.3f}")
    print(f"  --> hand-confidence change: {shipped:.3f} -> {dw_hand:.3f}  "
          f"({'+' if dw_hand>shipped else ''}{(dw_hand-shipped):.3f})")
import json
json.dump({"clip": CLIP, "frames": nframes, "cpu_fps": round(fps, 2),
           "dwpose_hand_conf": round(dw_hand, 3), "dwpose_body_conf": round(dw_body, 3),
           "shipped_alphapose_hand_conf": shipped},
          open(f"{OUT}/validate_result.json", "w"), indent=2)
