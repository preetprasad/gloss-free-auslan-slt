#!/usr/bin/env python3
"""DINOv2 WebGPU-vs-WASM harness server (PREREG_DINOV2_WEBGPU_2026-09-30).

Interpreter: ~/.pyenv/versions/3.14.3/bin/python (the only local env with cv2). No large-array arithmetic happens here
(decode, index, cvtColor, tobytes), so the numpy 1.26 elision bug has nothing to bite.

Routes (demo/serve_static.py's handler, headers and path rules, plus):
  /static/engine/dgpu/...   -> web/ (harness page + worker; '../rgb.js' is the SHIPPED demo/static/engine/rgb.js)
  /models/..., /hf/...      -> exactly as serve_static.py (results/live_onnx, local Hugging Face cache)
  /ortdist/...              -> demo/node_modules/onnxruntime-web/dist (byte-identical to config.js RUNTIME; the
                               worker's loader still sha256-checks them, so no other bytes can load)
  GET /dgpu/clips           -> {"clips": [800 validation ids, manifest order], "done": [ids with <id>.json uploaded]}
  GET /dgpu/frames/<clip>   -> one clip, decoded on demand, the exact frames + keypoints that
                               results/live_demo_2026-09-24/scripts/rgb_recompute_val.py feeds demo/rgb_features.py
  POST /__upload/<name>     -> out/<name> (serve_static --upload: needs header X-Dev-Upload, <= 64 MiB)

Frame payload (little-endian):
  bytes 0-3   b'DGF1'
  bytes 4-27  uint32 x 6: n (sampled frames), W, H, T (= min(video frames, KPS frames)), video frames, KPS frames
  then        int32  x n            sampled frame indices (RGBFeatures.sample_indices(T))
  then        float32 x n*133*3     KPS[idx] (KPS.astype(float32), as rgb_recompute_val.py)
  then        uint8  x n*H*W*3      frames[idx], RGB (cv2.cvtColor BGR2RGB, as RGBFeatures.__call__ does), HWC
  total = 28 + 4n + 1596n + 3nHW bytes.
Run from the repo root:  ~/.pyenv/versions/3.14.3/bin/python <scratch>/dino_gpu/serve.py [--port 8131]
"""
import argparse
import http.server
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('.')
sys.path.insert(0, str(ROOT))
from demo import serve_static as ss            # noqa: E402  (handler, header rules, path rules)
from demo.rgb_features import RGBFeatures      # noqa: E402  (sample_indices; imports torch, loads no model)

R = ROOT / 'results'
OUT = HERE / 'out'
SAMPLER = SimpleNamespace(frames=24)            # RGBFeatures() defaults to frames=24; its model is not needed here
CLIPS = [Path(l.split('\t')[0]).stem for l in (R / '_local_val_manifest.tsv').read_text().splitlines() if l.strip()]
assert len(CLIPS) == 800 == len(set(CLIPS)), len(CLIPS)


def payload(cid):
    """rgb_recompute_val.py's inputs for one clip, without holding every decoded frame (two decode passes)."""
    kps = np.load(R / 'val_clips_dwpose' / f'{cid}.npz')['KPS'].astype(np.float32)
    path = str(R / 'dwpose_experiment/rgb_clips/Signer' / f'{cid}_signer.mp4')
    cap, n = cv2.VideoCapture(path), 0
    while cap.grab():
        n += 1
    cap.release()
    T = min(n, len(kps))
    idx = RGBFeatures.sample_indices(SAMPLER, T)
    want, got, j = set(idx.tolist()), {}, 0
    cap = cv2.VideoCapture(path)
    while True:                                  # rgb_recompute_val.py: cap.read() until it fails
        ok, f = cap.read()
        if not ok:
            break
        if j in want:
            got[j] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        j += 1
    cap.release()
    if j != n:
        raise RuntimeError(f'{cid}: read() gave {j} frames, grab() {n}')
    frames = np.stack([got[i] for i in idx.tolist()])
    H, W = frames.shape[1:3]
    head = np.array([len(idx), W, H, T, n, len(kps)], '<u4').tobytes()
    return b'DGF1' + head + idx.astype('<i4').tobytes() + kps[idx].astype('<f4').tobytes() + frames.tobytes()


def make_handler(mounts, upload):
    Base = ss.make_handler(mounts, upload)

    class Handler(Base):
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.url_path
            if p == '/dgpu/clips':
                done = sorted(c for c in CLIPS if (OUT / f'{c}.json').exists())
                return self._send(json.dumps({'clips': CLIPS, 'done': done}).encode(), 'application/json')
            if p.startswith('/dgpu/frames/'):
                cid = p[len('/dgpu/frames/'):]
                if cid not in CLIPS:
                    return self.send_error(404, 'not a validation clip')
                try:
                    body = payload(cid)
                except Exception as e:                   # noqa: BLE001  (reported to the worker, which stops)
                    return self.send_error(500, f'{type(e).__name__}: {e}')
                return self._send(body, 'application/octet-stream')
            return super().do_GET()

    return Handler


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8131)   # not 8123: the studio server may be running there
    ap.add_argument('--out', default=str(OUT), help='upload directory (default: out/ next to this script)')
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    mounts = [('/models', R / 'live_onnx'), ('/static/engine/dgpu', HERE / 'web'),
              ('/ortdist', ROOT / 'demo/node_modules/onnxruntime-web/dist')]

    class Server(http.server.ThreadingHTTPServer):
        request_queue_size = 64
    srv = Server((a.host, a.port), make_handler(mounts, OUT))
    print(f'http://{a.host}:{a.port}/static/engine/dgpu/index.html  (uploads -> {OUT})', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
