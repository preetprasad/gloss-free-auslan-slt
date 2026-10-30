"""Pose-stage fixtures for the in-browser engine (G4): installed rtmlib + demo/service.py extract() outputs.

Renders browser-like JPEGs (<=960x720, quality 75, cv2.imencode) from validation signer videos, decodes them exactly
as demo.service.decode_frame does (PIL -> RGB -> BGR), and runs ModelService.extract (the real function, with the
Wholebody(mode='balanced') it constructs) while recording every OpenCV and ONNX Runtime call rtmlib makes inside it.
Nothing is re-implemented here: intermediates are captured from the live call.

Also writes the kernel-parity set (native cv2 resize/warpAffine/cvtColor on 20 real frames) and tracker/grid
stress cases generated from a shared 32-bit PRNG so the JS test regenerates identical inputs.

Outputs
  demo/tests/fixtures/pose/manifest.json   small, committed: frame list, hashes, expected extract() outputs
  results/live_onnx/fixtures/pose/          large, gitignored: JPEGs, decoded BGR, intermediates (*.bin, *.jpg)

Run: ~/.pyenv/versions/3.14.3/bin/python demo/tests/fixtures/make_pose_fixtures.py
"""
from __future__ import annotations

import base64
import importlib.metadata
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from demo.service import FRAME_MS, ModelService, _torso, decode_frame, grid_indices, seed_index, track_index  # noqa: E402

import onnxruntime as ort  # noqa: E402
import PIL  # noqa: E402
import rtmlib  # noqa: E402
from rtmlib import Wholebody  # noqa: E402
from rtmlib.tools.object_detection import yolox as yolox_mod  # noqa: E402
from rtmlib.tools.pose_estimation import pre_processings as pre_mod  # noqa: E402
from rtmlib.tools.file import _get_rtmhub_dir  # noqa: E402

SMALL = ROOT / 'demo/tests/fixtures/pose'
LARGE = ROOT / 'results/live_onnx/fixtures/pose'
VIDEOS = ROOT / 'results/dwpose_experiment/rgb_clips/Signer'
VAL = ROOT / 'results/_local_val_manifest.tsv'
NAMED = ['video_1_34', 'video_10_134']      # known multi-person validation clips (identity-switch cases)
MAX_W, MAX_H, QUALITY = 960, 720, 75        # demo/static/app.js CAPTURE_MAX_WIDTH/HEIGHT, JPEG_QUALITY 0.75
KERNEL_FRAMES = 20
RTM = Path(rtmlib.__file__).parent
RTM_SOURCES = ['tools/base.py', 'tools/object_detection/yolox.py', 'tools/object_detection/post_processings.py',
               'tools/pose_estimation/rtmpose.py', 'tools/pose_estimation/pre_processings.py',
               'tools/pose_estimation/post_processings.py', 'tools/solution/wholebody.py']


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def put(name: str, a: np.ndarray) -> str:
    (LARGE / name).write_bytes(np.ascontiguousarray(a).tobytes())
    return name


class Recorder:
    """Wraps rtmlib's cv2 module and ONNX sessions; records calls made inside ModelService.extract."""

    def __init__(self):
        self.log = []

    def cv2_proxy(self):
        rec = self

        class Proxy:
            def __getattr__(self, k):
                return getattr(cv2, k)

            def resize(self, img, dsize, *a, **kw):
                out = cv2.resize(img, dsize, *a, **kw)
                rec.log.append(('resize', dict(dsize=list(dsize), kw={k: int(v) for k, v in kw.items()}), out))
                return out

            def warpAffine(self, img, M, dsize, *a, **kw):
                out = cv2.warpAffine(img, M, dsize, *a, **kw)
                rec.log.append(('warpAffine', dict(M=M.copy(), dsize=list(dsize), kw={k: int(v) for k, v in kw.items()}), out))
                return out

            def getAffineTransform(self, src, dst):
                M = cv2.getAffineTransform(src, dst)
                rec.log.append(('getAffineTransform', dict(src=src.copy(), dst=dst.copy()), M))
                return M
        return Proxy()

    def session(self, real, tag):
        rec = self

        class Session:
            def get_inputs(self):
                return real.get_inputs()

            def get_outputs(self):
                return real.get_outputs()

            def run(self, names, feed):
                out = real.run(names, feed)
                rec.log.append((tag, dict(names=list(names), input=next(iter(feed.values())).copy()), [o.copy() for o in out]))
                return out
        return Session()


def mulberry32(seed: int):
    """Same generator as the JS test (uint32 arithmetic); returns floats in [0, 1)."""
    a = seed & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def f32(x) -> float:
    return float(np.float32(x))


def track_stress(seed=20260925, sequences=40, length=12):
    """Random detections around a moving signer; exercises invalid torsos, the 1.6 gate, ties and re-seeding.
    Input construction is mirrored line for line in demo/tests/engine_track.test.mjs."""
    r = mulberry32(seed)
    first = [r() for _ in range(4)]
    cases = []
    for _ in range(sequences):
        W = 320 + int(r() * 960)
        H = 240 + int(r() * 720)
        cx, cy, w = r() * W, r() * H, 20 + r() * 120
        track, frames = {}, []
        for _ in range(length):
            n = 1 + int(r() * 4)
            kp = np.zeros((n, 133, 2), np.float32)
            sc = np.zeros((n, 133), np.float32)
            for j in range(n):
                for k in range(133):
                    sc[j, k] = f32(r())
                if j == 0:              # the signer drifts; sometimes jumps (a cut)
                    d = r() * (3.2 if r() < 0.25 else 0.8)
                    dx, dy = r() * 2 - 1, r() * 2 - 1     # no trig: libm and V8 may differ in the last ulp
                    px, py = cx + d * w * dx, cy + d * w * dy
                    cx, cy = px, py
                    ww = w * (0.8 + 0.4 * r())
                else:
                    px, py, ww = r() * W, r() * H, 10 + r() * 150
                kp[j, 5] = (f32(px + ww / 2), f32(py + r() * 4 - 2))
                kp[j, 6] = (f32(px - ww / 2), f32(py + r() * 4 - 2))
                u = r()
                if u < 0.08:
                    sc[j, 5] = np.float32(0.3)            # exactly the float32 threshold
                elif u < 0.14:
                    sc[j, 6] = np.nextafter(np.float32(0.3), np.float32(0))
                elif u < 0.18:
                    kp[j, 6] = kp[j, 5]                   # zero-width torso
                elif u < 0.20 and j > 0:
                    kp[j] = kp[0]; sc[j] = sc[0]          # exact duplicate detection (tie)
            order = r()
            if order < 0.5:
                perm = [int(r() * 1e9) for _ in range(n)]
                idx = np.argsort(perm, kind='stable')
                kp, sc = kp[idx], sc[idx]
            torsos = []
            for j in range(n):
                mid, wid, ok = _torso(kp[j], sc[j])
                torsos.append(None if mid is None else [float(mid[0]), float(mid[1]), wid, ok])
            seed_i = seed_index(kp, sc, (W, H))
            i, reseeded = track_index(kp, sc, track.get('mid'), track.get('wid'), (W, H))
            mid, wid, ok = _torso(kp[i], sc[i])     # exactly as ModelService.extract updates its state
            if ok:
                track['mid'], track['wid'] = mid, wid
            frames.append(dict(n=n, torsos=torsos, seed=int(seed_i), index=int(i), reseeded=bool(reseeded),
                               sc_mean=[float(sc[j].mean()) for j in range(n)],
                               track=None if 'mid' not in track else [float(track['mid'][0]), float(track['mid'][1]), track['wid']]))
        cases.append(dict(W=W, H=H, frames=frames))
    return dict(seed=seed, sequences=sequences, length=length, first=first, cases=cases)


def fma_stress(seed=11, n=4000):
    """Triples for the JS exact-FMA emulation, half of them near-cancelling; results hashed (math.fma)."""
    r = mulberry32(seed)
    out = []
    for i in range(n):
        a = (r() - 0.5) * 2.0 ** (int(r() * 60) - 30)
        b = (r() - 0.5) * 2.0 ** (int(r() * 60) - 30)
        c = -(a * b) * (1 + (r() - 0.5) * 2.0 ** -int(r() * 60)) if i % 2 else (r() - 0.5) * 2.0 ** (int(r() * 80) - 40)
        out.append(math.fma(a, b, c))
    return dict(seed=seed, n=n, sha256=sha(np.array(out, np.float64).tobytes()), first=out[:4])


def affine_stress(n=300):
    """getAffineTransform on random float32 point triples (cv2 native) — the solver whose FMA use JS reproduces."""
    rs = np.random.RandomState(3)
    src, dst, M = [], [], []
    while len(M) < n:
        s, d = rs.uniform(-500, 1500, (3, 2)).astype(np.float32), rs.uniform(-300, 300, (3, 2)).astype(np.float32)
        try:
            m = cv2.getAffineTransform(s, d)
        except cv2.error:
            continue
        src.append(s); dst.append(d); M.append(m)
    return dict(src=b64(np.array(src)), dst=b64(np.array(dst)), M=b64(np.array(M, np.float64)))


def grid_stress(seed=7, n_cases=200):
    r = mulberry32(seed)
    cases = []
    for c in range(n_cases):
        n = 1 + int(r() * 90)
        t = [float(int(r() * 1000))]
        for _ in range(n - 1):
            gap = 20 + int(r() * 60) if c % 3 else 20 + int(r() * 6) * 10     # multiples of 10 force ties
            t.append(t[-1] + (gap if c % 5 else gap + 0.5))
        cases.append(dict(n=n, t_last=t[-1], idx=grid_indices(t)))   # t is regenerated by the JS test
    return dict(seed=seed, cases=cases)


# A capture timeline with a fractional start (t0 = 237.7) on which numpy's arange (start + i*delta, compiled to an FMA)
# and the unfused start + i*delta differ by 1 ulp at 22 of 81 grid points, which flips the nearest frame at grid 11.
GRID_T0_CASE = [
    237.70000000000002, 277.70000000000005, 317.70000000000005, 357.70000000000005, 397.70000000000005,
    437.70000000000005, 477.70000000000005, 517.7, 557.7, 597.7, 637.7, 657.7000000000003, 697.7000000000003, 717.7,
    757.7, 797.7, 837.7, 877.7, 917.7, 957.7, 997.7, 1037.7, 1077.7, 1117.7, 1157.7, 1197.7, 1237.7, 1277.7, 1317.7,
    1357.7, 1397.7, 1437.7, 1477.7, 1517.7, 1557.7, 1597.7, 1637.7, 1677.7, 1717.7, 1757.7, 1797.7, 1837.7, 1877.7,
    1917.7, 1957.7, 1997.7, 2037.7, 2077.7, 2117.7, 2157.7, 2197.7, 2237.7, 2277.7, 2317.7, 2357.7, 2397.7, 2437.7,
    2477.7, 2517.7, 2557.7, 2597.7, 2637.7, 2677.7, 2717.7, 2757.7, 2797.7, 2837.7, 2877.7, 2917.7, 2957.7, 2997.7,
    3037.7, 3077.7, 3117.7, 3157.7, 3197.7, 3237.7, 3277.7, 3317.7, 3357.7, 3397.7, 3437.7]


def _grid_unfused(t):
    """grid_indices with the arange fill computed as an unfused start + i*delta (the pre-fix JS), for the report."""
    t = np.asarray(t, float)
    n = math.ceil((t[-1] + 1e-6 - t[0]) / FRAME_MS)
    start, second = t[0], t[0] + FRAME_MS
    grid = np.array([start, second] + [start + i * (second - start) for i in range(2, n)])[:n]
    return np.abs(t[None, :] - grid[:, None]).argmin(1).tolist()


def grid_frac_stress(seed=13, n_random=100, n_ties=40):
    """Fractional capture times (microsecond resolution). `random`: 40 ms steps, 20 ms half-steps and odd gaps,
    regenerated by engine_track.test.mjs from the shared PRNG. `ties`: stored literally; frames sit exactly 20 ms
    either side of numpy's own grid points (as in the constructed t0 = 237.7 case, ties[0]), so the nearest frame is
    decided by the last bit of the grid value, i.e. by whether start + i*delta is computed as one FMA (numpy) or not.
    The random cases never separate the two forms (reported, not asserted); the ties do."""
    r = mulberry32(seed)
    random, random_wrong = [], 0
    for _ in range(n_random):
        n = 2 + int(r() * 80)
        t = [int(r() * 1e8) / 1000]
        for _ in range(n - 1):
            u = r()
            gap = 40.0 if u < 0.6 else 20.0 if u < 0.85 else 20 + int(r() * 60) + int(r() * 10) / 10
            t.append(t[-1] + gap)
        idx = grid_indices(t)
        random_wrong += idx != _grid_unfused(t)
        random.append(dict(n=n, t0=t[0], t_last=t[-1], idx=idx))
    rs = np.random.RandomState(seed)
    ties = [GRID_T0_CASE]
    while len(ties) < n_ties:
        # t0 within 40 ms below a small power of two: t0 + 40 rounds, so delta = (t0 + 40) - t0 is not exactly 40 and
        # has enough significant bits that i*delta rounds too; only then do the fused and unfused forms split (at
        # larger t0 or an exact delta they agree).
        t0, K = 2.0 ** int(rs.randint(6, 11)) - int(rs.randint(1, 40000)) / 1000, int(rs.randint(3, 50))
        g = np.arange(t0, t0 + 40.0 * K, 40.0)          # the grid numpy rebuilds from t[0] (fill ignores the stop)
        delta = (t0 + 40.0) - t0
        split = [k > 1 and g[k] != t0 + k * delta for k in range(K)]   # grid points the unfused form gets wrong
        t, pair = [t0], False
        for k in range(1, K):                          # pairs where the forms split, some elsewhere; never two in a row
            pair = not pair and (split[k] or rs.rand() < 0.3)
            t += [float(g[k] - 20.0), float(g[k] + 20.0)] if pair else [float(g[k])]
        ties.append(t)
    ties = [dict(times=t, idx=grid_indices(t), unfused_wrong=grid_indices(t) != _grid_unfused(t)) for t in ties]
    assert ties[0]['unfused_wrong']
    return dict(seed=seed, random=random, random_unfused_wrong=random_wrong, ties=ties,
                ties_unfused_wrong=sum(c['unfused_wrong'] for c in ties))


def warp_fuzz(wb, rec, frames, per_frame=84):
    """D1 fuzz: rtmlib's own top-down preprocessing (RTMPose.preprocess -> getAffineTransform + cv2.warpAffine, as
    recorded by the proxy) on every real frame: 60 person-like boxes and 24 random, partly outside boxes per frame;
    plus the 409-box degenerate set (400 random + zero/negative/tiny/huge/far/whole-image boxes) on the first 813 px
    frame. The JS test recomputes src, M and the crop from the box and compares the crop by SHA-256."""
    def run(e, bgr, b):
        rec.log.clear()
        wb.pose_model.preprocess(bgr, b)
        g = [x for x in rec.log if x[0] == 'getAffineTransform'][0]
        w = [x for x in rec.log if x[0] == 'warpAffine'][0]
        assert w[1]['kw'] == dict(flags=cv2.INTER_LINEAR) and w[1]['dsize'] == [192, 256], w[1]
        return dict(frame=e['name'], box=[float(v) for v in b], src=[float(v) for v in g[1]['src'].astype(np.float32).ravel()],
                    M=[float(v) for v in w[1]['M'].ravel()], crop_sha256=sha(w[2].tobytes()))

    def load(e):
        return np.frombuffer((LARGE / f"{e['name']}.bgr.bin").read_bytes(), np.uint8).reshape(e['height'], e['width'], 3)

    real, degenerate = [], []
    rs = np.random.RandomState(7)
    for e in frames:
        bgr = load(e)
        H, W = bgr.shape[:2]
        for j in range(per_frame):
            if j < 60:                                          # person-like: 30-100% of the height, near the frame
                bh = rs.uniform(0.3, 1.0) * H; bw = bh * rs.uniform(0.3, 1.2)
                x1 = rs.uniform(-0.1 * W, W - 0.5 * bw); y1 = rs.uniform(-0.1 * H, H - 0.5 * bh)
                box = [x1, y1, x1 + bw, y1 + bh]
            else:                                               # anywhere, any size, often partly outside
                x1, y1 = rs.uniform(-0.5 * W, 1.2 * W), rs.uniform(-0.5 * H, 1.2 * H)
                box = [x1, y1, x1 + rs.uniform(1, 1.5 * W), y1 + rs.uniform(1, 1.5 * H)]
            real.append(run(e, bgr, np.asarray(box, np.float32)))
    e = [x for x in frames if x['width'] == 813][0]
    bgr = load(e)
    H, W = bgr.shape[:2]
    rs = np.random.RandomState(42)
    boxes = []
    for _ in range(400):
        x1, y1 = rs.uniform(-0.5 * W, 1.2 * W), rs.uniform(-0.5 * H, 1.2 * H)
        boxes.append([x1, y1, x1 + rs.uniform(1, 1.5 * W), y1 + rs.uniform(1, 1.5 * H)])
    boxes += [[100, 100, 100, 100],          # zero size: singular -> zero matrix
              [100, 100, 100, 300],          # zero width
              [100, 100, 400, 100],          # zero height
              [400, 300, 100, 100],          # negative width and height
              [0, 0, 1e-3, 1e-3],            # tiny
              [-1e5, -1e5, 1e5, 1e5],        # huge
              [5000, 5000, 5100, 5200],      # far outside
              [0, 0, W, H],                  # whole image (rtmlib's empty-bbox default)
              [12.345678, 7.654321, 700.1, 699.9]]
    for b in np.asarray(boxes, np.float32):
        degenerate.append(run(e, bgr, b))
    body = json.dumps(dict(real=real, degenerate=degenerate), separators=(',', ':')).encode()
    (LARGE / 'warp_fuzz.json').write_bytes(body)
    return dict(file='warp_fuzz.json', sha256=sha(body), real=len(real), degenerate=len(degenerate),
                frames=len(frames), per_frame=per_frame, degenerate_frame=e['name'])


def warp_extreme(frame):
    """cv2.warpAffine exactly as rtmlib calls it, on matrices no detector box produces: a subnormal determinant (Inf
    inverse), NaN and Inf entries, map offsets past int32 and past int64 (cvRound's wrap and saturation)."""
    nan, inf = float('nan'), float('inf')
    Ms = [[1e-160, 0, 0, 0, 1e-160, 0], [1e-7, 0, 5, 0, 1e-7, 3], [1e-17, 0, 0, 0, 1e-17, 0],
          [-1e-17, 0, 0, 0, -1e-17, 0], [nan, 0, 0, 0, 1, 0], [inf, 0, 0, 0, 1, 0], [1, 0, nan, 0, 1, 0],
          [1e-300, 0, 1, 0, 1e-300, 1], [2e-5, 1e-9, -3, 1e-9, 2e-5, 4], [0.5, 0.1, -1e12, 0.2, 0.4, 1e12]]
    bgr = np.frombuffer((LARGE / f"{frame['name']}.bgr.bin").read_bytes(), np.uint8).reshape(frame['height'], frame['width'], 3)
    return dict(frame=frame['name'], cases=[
        dict(M=b64(np.array(M, np.float64)),
             crop_sha256=sha(cv2.warpAffine(bgr, np.array(M, np.float64).reshape(2, 3), (192, 256), flags=cv2.INTER_LINEAR).tobytes()))
        for M in Ms])


def pick_clips(det):
    names = [Path(line.split('\t')[0]).stem for line in VAL.read_text().splitlines()]
    extra_multi, extra_single = [], []
    for name in names:
        if name in NAMED or len(extra_multi) >= 2 and len(extra_single) >= 2:
            continue
        cap = cv2.VideoCapture(str(VIDEOS / f'{name}_signer.mp4'))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read()
        if not ok:
            continue
        k = len(det(fr))
        if k >= 2 and len(extra_multi) < 2:
            extra_multi.append(name)
        elif k == 1 and len(extra_single) < 2:
            extra_single.append(name)
    return [(c, 8) for c in NAMED] + [(c, 5) for c in extra_multi + extra_single]


def browser_jpeg(bgr: np.ndarray) -> bytes:
    h, w = bgr.shape[:2]
    s = min(1.0, MAX_W / w, MAX_H / h)          # app.js: Math.min(1, 960 / width, 720 / height)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    assert ok
    return buf.tobytes()


def main():
    SMALL.mkdir(parents=True, exist_ok=True)
    (LARGE / 'frames').mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)                        # as ModelService.__init__
    wb = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')   # ModelService options, no overrides
    svc = SimpleNamespace(pose=wb)
    hub = Path(_get_rtmhub_dir()) / 'checkpoints'
    models = {k: dict(file=Path(m.onnx_model).name, sha256=sha(Path(m.onnx_model).read_bytes()))
              for k, m in (('detector', wb.det_model), ('pose', wb.pose_model))}
    assert all((hub / v['file']).is_file() for v in models.values())

    rec = Recorder()
    yolox_mod.cv2 = rec.cv2_proxy()
    pre_mod.cv2 = rec.cv2_proxy()
    wb.det_model.session = rec.session(wb.det_model.session, 'det')
    wb.pose_model.session = rec.session(wb.pose_model.session, 'pose')

    clips = pick_clips(lambda fr: wb.det_model(fr))
    rec.log.clear()

    jobs = []
    for clip, k in clips:
        cap = cv2.VideoCapture(str(VIDEOS / f'{clip}_signer.mp4'))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in np.linspace(0, n - 1, k).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            assert ok, (clip, fi)
            jobs.append((clip, int(fi), browser_jpeg(fr)))
    # A frame with nobody in it: extract() must return the zero frame with count 0 and never run pose.
    blank = np.full((480, 640, 3), (60, 140, 90), np.uint8)
    cv2.circle(blank, (320, 240), 90, (200, 200, 200), -1)
    jobs.append(('empty', 0, cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])[1].tobytes()))

    frames, tracks = [], {}
    for fid, (clip, fi, jpg) in enumerate(jobs):
        name = f'{fid:02d}_{clip}_f{fi}'
        (LARGE / 'frames' / f'{name}.jpg').write_bytes(jpg)
        bgr = decode_frame('data:image/jpeg;base64,' + base64.b64encode(jpg).decode())
        put(f'{name}.bgr.bin', bgr)
        track = tracks.setdefault(clip, {})
        prev = None if 'mid' not in track else [float(track['mid'][0]), float(track['mid'][1]), track['wid']]
        rec.log.clear()
        out, count = ModelService.extract(svc, bgr, track)
        log = rec.log[:]
        entry = dict(name=name, clip=clip, frame=fi, jpeg_sha256=sha(jpg), width=bgr.shape[1], height=bgr.shape[0],
                     bgr_sha256=sha(bgr.tobytes()), count=int(count), extract=b64(out.astype(np.float32)),
                     track_before=prev,
                     track_after=None if 'mid' not in track else [float(track['mid'][0]), float(track['mid'][1]), track['wid']])
        kinds = [x[0] for x in log]
        det = [x for x in log if x[0] == 'det']
        assert len(det) == 1, kinds
        resize = [x for x in log if x[0] == 'resize']
        lb = det[0][1]['input'][0].transpose(1, 2, 0)
        assert np.array_equal(lb, lb.round()) and lb.min() >= 0 and lb.max() <= 255
        entry['letterbox'] = put(f'{name}.letterbox.bin', lb.astype(np.uint8))
        entry['resize'] = [dict(r[1]) for r in resize]
        dets = det[0][2][0]
        entry['dets_shape'] = list(dets.shape)
        entry['dets'] = put(f'{name}.dets.bin', dets.astype(np.float32))
        boxes = wb.det_model.postprocess(dets.copy(), min(640 / bgr.shape[0], 640 / bgr.shape[1]))
        assert len(boxes) == count
        entry['boxes'] = b64(np.asarray(boxes, np.float32).reshape(-1, 4))
        warps = [x for x in log if x[0] == 'warpAffine']
        gats = [x for x in log if x[0] == 'getAffineTransform']
        poses = [x for x in log if x[0] == 'pose']
        assert len(warps) == len(gats) == len(poses) == (count if count else 0), kinds
        entry['persons'] = []
        for j, (w, g, p) in enumerate(zip(warps, gats, poses)):
            simx, simy = p[2]
            entry['persons'].append(dict(
                src=b64(g[1]['src'].astype(np.float32)), M=b64(w[1]['M'].astype(np.float64)), dsize=w[1]['dsize'],
                warp_kw=w[1]['kw'], crop=put(f'{name}.p{j}.crop.bin', w[2]),
                input=put(f'{name}.p{j}.input.bin', p[1]['input'].astype(np.float32)),
                simx=put(f'{name}.p{j}.simx.bin', simx.astype(np.float32)), simy=put(f'{name}.p{j}.simy.bin', simy.astype(np.float32)),
                simx_shape=list(simx.shape), simy_shape=list(simy.shape)))
        frames.append(entry)

    # All-detection keypoints: call the pose model on the recorded boxes (same frame, same boxes) and check the
    # tracked row reproduces extract() exactly, which also shows ORT is deterministic run to run.
    for e in frames:
        if not e['count']:
            continue
        bgr = np.frombuffer((LARGE / f"{e['name']}.bgr.bin").read_bytes(), np.uint8).reshape(e['height'], e['width'], 3)
        boxes = np.frombuffer(base64.b64decode(e['boxes']), np.float32).reshape(-1, 4)
        kp, sc = wb.pose_model(bgr, bboxes=boxes)
        kp, sc = np.asarray(kp, np.float32), np.asarray(sc, np.float32)
        out = np.frombuffer(base64.b64decode(e['extract']), np.float32).reshape(133, 3)
        hits = [j for j in range(len(kp)) if np.array_equal(np.concatenate([kp[j], sc[j][:, None]], 1), out)]
        assert hits, e['name']
        e['selected'] = hits[0]
        e['kp'] = put(f"{e['name']}.kp.bin", kp)
        e['sc'] = put(f"{e['name']}.sc.bin", sc)
        e['kp_shape'] = list(kp.shape)

    # Kernel-parity set: native cv2 on the first 20 real frames (inputs are the decoded BGR above).
    kernel = []
    rng = np.random.RandomState(0)
    real = [e for e in frames if e['clip'] != 'empty'][:KERNEL_FRAMES]
    for e in real:
        bgr = np.frombuffer((LARGE / f"{e['name']}.bgr.bin").read_bytes(), np.uint8).reshape(e['height'], e['width'], 3)
        h, w = bgr.shape[:2]
        k = dict(name=e['name'])
        k['r224'] = put(f"{e['name']}.k.r224.bin", cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR))
        y0, x0 = h // 3, w // 3
        k['up_region'] = [x0, y0, 50, 70]
        k['up224'] = put(f"{e['name']}.k.up224.bin",
                         cv2.resize(np.ascontiguousarray(bgr[y0:y0 + 70, x0:x0 + 50]), (224, 224), interpolation=cv2.INTER_LINEAR))
        k['rgb'] = put(f"{e['name']}.k.rgb.bin", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rgba = np.dstack([bgr[:, :, ::-1], np.full((h, w), 255, np.uint8)])
        back = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        assert np.array_equal(back, bgr)        # RGBA->BGR is the permutation decode_frame applies
        # Extra boxes through rtmlib's own top-down preprocessing, including boxes partly outside the frame.
        boxes = [[x1, y1, x1 + bw, y1 + bh] for x1, y1, bw, bh in
                 zip(rng.uniform(-0.3 * w, 0.9 * w, 4), rng.uniform(-0.3 * h, 0.9 * h, 4),
                     rng.uniform(20, 1.2 * w, 4), rng.uniform(20, 1.2 * h, 4))]
        k['boxes'] = []
        for j, b in enumerate(np.asarray(boxes, np.float32)):
            rec.log.clear()
            wb.pose_model.preprocess(bgr, b)
            g = [x for x in rec.log if x[0] == 'getAffineTransform'][0]
            wa = [x for x in rec.log if x[0] == 'warpAffine'][0]
            k['boxes'].append(dict(box=b64(b), src=b64(g[1]['src'].astype(np.float32)), M=b64(wa[1]['M']),
                                   crop=put(f"{e['name']}.k.b{j}.crop.bin", wa[2])))
        kernel.append(k)

    manifest = dict(
        purpose='G4 pose parity fixtures (installed rtmlib + demo.service.extract); regenerate with this script',
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, cv2=cv2.__version__, onnxruntime=ort.__version__,
                      pillow=PIL.__version__, rtmlib=importlib.metadata.version('rtmlib')),
        rtmlib_sources={f: sha((RTM / f).read_bytes()) for f in RTM_SOURCES},  # the patched installed copy that was ported
        models=models, hub=f'~/{hub.relative_to(Path.home())}' if hub.is_relative_to(Path.home()) else str(hub), large_dir=str(LARGE.relative_to(ROOT)),
        mode='balanced', det_input=[640, 640], pose_input=[192, 256],
        frames=frames, kernel=kernel, track=track_stress(), grid=grid_stress(), fma=fma_stress(), affine=affine_stress(),
        grid_frac=grid_frac_stress(), warp_fuzz=warp_fuzz(wb, rec, [e for e in frames if e['clip'] != 'empty']))
    manifest['warp_extreme'] = warp_extreme([e for e in frames if e['name'] == manifest['warp_fuzz']['degenerate_frame']][0])
    (SMALL / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    multi = sum(e['count'] >= 2 for e in frames)
    print(f"{len(frames)} frames ({multi} with >=2 detections, {sum(e['count'] == 0 for e in frames)} empty) "
          f"from {len(clips)} clips; kernel set {len(kernel)}; manifest {(SMALL / 'manifest.json').stat().st_size} bytes")


if __name__ == '__main__':
    main()
