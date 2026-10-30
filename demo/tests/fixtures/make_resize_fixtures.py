"""Resize fuzz for the JS cv2.resize port (demo/static/engine/cv.js resize): native cv2 4.13 INTER_LINEAR on uint8
3-channel images, the one call the engine makes (YOLOX letterbox in rtmlib; the 224 x 224 crops of
demo/rgb_features.py). Writes demo/tests/fixtures/resize/fuzz.json: per case the source size, the target size, an
offset into a deterministic byte pool and the first 16 hex digits of sha256(output). The pool (SHA-256 in counter mode
over POOL_TAG) is rebuilt by the test, so no image is stored.

Run with the interpreter the Python service uses (cv2 is only there): ~/.pyenv/versions/3.14.3/bin/python.
numpy is used for views only (frombuffer / reshape), never for arithmetic (numpy 1.26.3 on Python 3.14: see CLAUDE.md).
"""
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / 'resize' / 'fuzz.json'
POOL_TAG = b'auslan-resize-fuzz'
POOL_BYTES = 4 << 20                    # >= 1280 * 960 * 3
SEED = 20260930
N = dict(letterbox=1000, crop=1000, general=500)


def pool() -> bytes:
    return b''.join(hashlib.sha256(POOL_TAG + i.to_bytes(4, 'little')).digest() for i in range(POOL_BYTES // 32))


def loguni(r: random.Random, lo: int, hi: int) -> int:
    return min(hi, max(lo, int(math.exp(r.uniform(math.log(lo), math.log(hi + 1))))))


def edges():
    """Deterministic corner cases: copies, 1 x N / N x 1, exact 2x (INTER_AREA inside cv2), other integer factors,
    2x in one axis only, near-2x, upscales, and the capture sizes the studio sends."""
    c = []
    for s in [(1, 1), (2, 2), (1, 7), (7, 1), (1, 960), (1280, 1), (2, 1), (1, 2), (3, 3), (223, 224), (225, 224),
              (224, 225), (224, 224), (447, 448), (449, 448), (448, 447), (448, 448), (448, 449), (672, 672), (896, 896),
              (448, 100), (100, 448), (448, 224), (112, 112), (113, 97), (50, 70), (1280, 960), (960, 720), (640, 480),
              (835, 720), (468, 720), (1, 448), (448, 1), (225, 1), (1, 225)]:
        c.append(('edge', s, (224, 224)))
    for s, d in [((1280, 960), (640, 480)), ((1280, 960), (640, 640)), ((960, 720), (640, 480)), ((640, 480), (640, 480)),
                 ((320, 240), (640, 480)), ((2, 2), (1, 1)), ((4, 4), (2, 2)), ((6, 4), (3, 2)), ((3, 1), (1, 1)),
                 ((1, 3), (1, 1)), ((5, 5), (2, 2)), ((9, 6), (3, 2)), ((1, 1), (640, 1)), ((1, 1), (1, 640)),
                 ((1, 1), (2, 2)), ((2, 1), (4, 2)), ((1280, 960), (427, 320)), ((1280, 960), (426, 320)),
                 ((1279, 959), (640, 480)), ((1280, 958), (640, 479)), ((639, 479), (640, 480)), ((1280, 2), (640, 1)),
                 ((2, 960), (1, 480)), ((1280, 960), (1, 1)), ((1280, 960), (1280, 1)), ((1280, 960), (1, 960))]:
        c.append(('edge', s, d))
    return c


def main():
    r = random.Random(SEED)
    cases = edges()
    for _ in range(N['letterbox']):             # rtmlib YOLOX.preprocess: ratio = min(640 / h, 640 / w)
        W, H = (loguni(r, 1, 1280), loguni(r, 1, 960)) if r.random() < 0.6 else (r.randint(320, 1280), r.randint(240, 960))
        ratio = min(640 / H, 640 / W)
        w, h = int(W * ratio), int(H * ratio)
        if w and h:
            cases.append(('letterbox', (W, H), (w, h)))
    for _ in range(N['crop']):                  # rgb_features.get_crop: a clamped box of the frame -> 224 x 224
        cases.append(('crop', (loguni(r, 1, 1280), loguni(r, 1, 960)), (224, 224)))
    for _ in range(N['general']):
        cases.append(('general', (loguni(r, 1, 1280), loguni(r, 1, 960)), (loguni(r, 1, 640), loguni(r, 1, 640))))

    buf = pool()
    rows = []
    for kind, (W, H), (w, h) in cases:
        n = W * H * 3
        off = r.randrange(0, len(buf) - n + 1)
        img = np.frombuffer(buf, np.uint8, count=n, offset=off).reshape(H, W, 3)
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        assert out.shape == (h, w, 3) and out.dtype == np.uint8, (W, H, w, h, out.shape)
        rows.append([kind, W, H, w, h, off, hashlib.sha256(out.tobytes()).hexdigest()[:16]])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(
        about='cv2.resize(img, (w, h), interpolation=INTER_LINEAR) on uint8 HxWx3 images cut from a byte pool; '
              'cases: [kind, W, H, w, h, pool offset, sha256(output)[:16]]',
        pool=dict(tag=POOL_TAG.decode(), bytes=POOL_BYTES, sha256=hashlib.sha256(buf).hexdigest()),
        seed=SEED, versions=dict(cv2=cv2.__version__, numpy=np.__version__, python=platform.python_version(),
                                 machine=platform.machine(), interpreter=sys.executable.replace(str(Path.home()), '~')),
        cases=rows)
    OUT.write_text(json.dumps(doc, separators=(',', ':')) + '\n')
    kinds = {k: sum(1 for x in rows if x[0] == k) for k in dict.fromkeys(x[0] for x in rows)}
    print(f'{len(rows)} cases {kinds} -> {OUT} ({OUT.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
