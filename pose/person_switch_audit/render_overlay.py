"""Overlay the stored (index-0, red) and person-tracked (green) DWPose skeletons on video_89_193's frames.
Writes PNGs OUTSIDE the repo (dataset frames must not be committed). Usage: render_overlay.py OUT_DIR TRACKED_NPZ"""
import sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
out, tracked = Path(sys.argv[1]), sys.argv[2]
assert ROOT not in out.resolve().parents, "frames are dataset material; write outside the repo"
out.mkdir(parents=True, exist_ok=True)
S = np.load(ROOT / "results/val_clips_dwpose/video_89_193.npz")["KPS"]
T = np.load(tracked); TK, CH, ND = T["KPS"], T["CHOSEN"], T["NDET"]
cap = cv2.VideoCapture(str(ROOT / "results/dwpose_experiment/rgb_clips/Signer/video_89_193_signer.mp4"))
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
assert len(frames) == len(S) == len(TK) == 49, (len(frames), len(S), len(TK))
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
def draw(img, k, col):
    for a, b in EDGES:
        if k[a, 2] > 0.3 and k[b, 2] > 0.3:
            cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), col, 2)
    for i in range(91, 133):
        if k[i, 2] > 0.3: cv2.circle(img, tuple(map(int, k[i, :2])), 2, col, -1)
for t in range(49):
    img = frames[t].copy()
    draw(img, S[t], (0, 0, 255)); draw(img, TK[t], (0, 200, 0))
    same = np.abs(S[t, :, :2] - TK[t, :, :2]).max() < 5.0
    cv2.putText(img, f"frame {t+1}  CHOSEN {CH[t]}  NDET {ND[t]}  {'SAME' if same else 'DIFFERENT'}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"f{t+1:02d}.png"), img)
agree = [bool(np.abs(S[t, :, :2] - TK[t, :, :2]).max() < 5.0) for t in range(49)]
print("frames where stored == tracked:", sum(agree), "/ 49; CHOSEN==0 frames:", int((CH == 0).sum()))
assert agree == [bool(c == 0) for c in CH], "stored pose is not the tracker's detection 0 on some frame"
print("check OK: stored pose equals the tracked run detection 0 (within 5 px, run-to-run numerics) exactly where CHOSEN == 0")
print("frame size", frames[0].shape)
