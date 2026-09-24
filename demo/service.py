"""Request-independent inference; no camera images or transcripts are retained here.

Two research operating points, chosen per utterance: the validation-best single model and the six-model
ensemble behind the reported 11.80. Both decode through the research ensemble decoder (one member or six),
so a live answer is produced exactly as the recorded validation and test numbers were.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 1_000_000
MAX_IMAGE_PIXELS = 1280 * 960
FRAME_MS = 40.0  # every Auslan-Daily training clip is 25 fps (REALTIME brief :79, :91)

# Operating points, bound to the exact weights by SHA-256. Candidates are ranked by raw / words**lp
# (tools/ensemble_mbr_wer.py). Confidence is self-CE = -raw / (words + 1), the mean negative log-prob per
# generated token including EOS (tools/ensemble_selective.py::load); it equals the single-model self-CE
# of SELECTIVE_PREDICTION_2026-09-09. Thresholds are quantiles of VALIDATION self-CE for these weights at
# the stated answer rate; the validated range is 50-90% (never below 50%).
MODELS = {
    'single': dict(
        label='Single best model · adopted recipe (pose augmentation + 1.1:1 duration stitch), wave-4 seed 202 · pose only',
        ckpts=[('results/ckpts_live/w4A_s202_best.pt',   # Kaya /group/pmc099/pprasad/wave4/runs_tq/w4A_s202/best.pt
                '0c245b3701d0c80ce0c8ad79a1c81c964afced5f0ac8c3eabe98f87bd35d65b6')],
        # The validation-best of all 21 adopted-recipe single models scored by job 21736 (7 batches x 3 seeds,
        # results/PREREG_SEEDENS_2026-09-24.md:78), at its validation-selected decode: validation 12.643,
        # test 11.652. Reproduced locally; tau from results/live_demo_2026-09-24/dump_w4A_s202_val.
        beam=10, lp=0.7, rgb=False,
        tau={0.9: 0.974299, 0.75: 0.859783, 0.5: 0.706661},
        evidence=dict(validation_bleu4=12.643, test_bleu4=11.652)),
    'ensemble': dict(
        label='Six-model ensemble (S6) · 3 pose-only + 3 pose+RGB members',
        ckpts=[('results/ckpts/comm_dwpose_ctc_vlplong80.pt', '20a6c5e3119417066a4638be324a77b870e36510aae29ca58b3a279bbfc15b09'),
               ('results/ckpts/comm_dwpose_rgb_fusion.pt', 'a3457758a8c9e645e4b4012d5ebbc2171cb2d1c0a2372fcc21078c87ec83fa8d'),
               ('results/ckpts/comm_dwpose_vlp_ctc.pt', '4ca55b0a77868f4146b8c243f428d348d9afe8d266e301c246fc355eb17f23c1'),
               ('results/ckpts/comm_dwpose_ctc_s202.pt', '5e7dec289e5b2b7f062ca0989361979be8b0bf46bacb43f8dfa07b7e4dedd62d'),
               ('results/ckpts/comm_dwpose_fus_s101.pt', 'bed3caeb2a991a6e9dae320f2fb8eaf0319b6470c21b9dc8c3870468393b30e4'),
               ('results/ckpts/comm_dwpose_fus_s202.pt', '6582e4ffedee2796e0b8dc056b9d88cd77229df95f0294f708801ef46b8da2fb')],
        # Recorded order (jobs/slurm/mbr_wer_test2.sbatch); LP 0.3 selected on validation (job 1101909).
        # tau from results/final_ensemble_validation_decodes; at 50% it reproduces the recorded test 19.07 @ 51.9% (enssel_S6.json).
        beam=10, lp=0.3, rgb=True,
        tau={0.9: 1.236622, 0.75: 1.088921, 0.5: 0.904779},
        evidence=dict(validation_bleu4=12.470, test_bleu4=11.802)),
}


def decode_frame(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith('data:image/jpeg;base64,'):
        raise ValueError('Send a JPEG camera frame.')
    encoded = value.split(',', 1)[1]
    if len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ValueError('Camera frame is too large.')
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != 'JPEG' or img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError('Use a camera resolution of 1280 × 960 or lower.')
            return np.asarray(img.convert('RGB'))[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, binascii.Error, Image.DecompressionBombError) as exc:
        raise ValueError('The camera frame could not be read.') from exc


def check_asset(path: str, expected_hash: str = '') -> str:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f'Required model asset is missing: {p.name}')
    if expected_hash:
        with p.open('rb') as asset:
            digest = hashlib.file_digest(asset, 'sha256').hexdigest()
        if digest != expected_hash:
            raise ValueError(f'Model asset checksum mismatch: {p.name}')
    return str(p.resolve())


def grid_indices(times_ms) -> list[int]:
    """Nearest captured frame on the 25 fps training grid (same idiom as slt/data.py's nearest-frame resampling).

    Live capture never runs at exactly 25 fps; feeding the raw sequence compresses time (VLP-40 at 15 fps:
    -1.22 BLEU-4 [-2.24, -0.26]; resampled: -0.25 [-0.87, +0.33]).
    """
    t = np.asarray(times_ms, float)
    grid = np.arange(t[0], t[-1] + 1e-6, FRAME_MS)
    return np.abs(t[None, :] - grid[:, None]).argmin(1).tolist()


# Person tracking, verbatim from results/dwpose_experiment/dwpose_extract_worker_tracked.py:51-96.
# Not imported: that module replaces ort.InferenceSession globally at import time.
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


class Translator:
    """One research operating point: members decoded with the research ensemble beam."""

    def __init__(self, spec, device='cpu'):
        import torch
        sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src' / 'tools')]
        from config import Config
        from slt.data import Vocab
        from slt.preproc_guard import assert_preproc_matches_checkpoints
        from ensemble_decode import Member
        from ensemble_mbr_wer import _align_vocabs, ensemble_nbest
        self.torch, self.nbest = torch, ensemble_nbest
        cks = [torch.load(check_asset(str(ROOT / p), sha), map_location='cpu', weights_only=False) for p, sha in spec['ckpts']]
        self.cfg = Config(num_keypoints=133, keypoint_subset='all', coord_norm='frame', rgb_feature_dim=1152, device=device)
        assert_preproc_matches_checkpoints(self.cfg, cks)
        self.dev = torch.device(device)
        self.vocab = Vocab(cks[0]['vocab'])
        # Vocabulary order differs between recipes; a mismatch is a hard error, never a silent average.
        self.members = _align_vocabs(cks, [Member(c, self.cfg, self.vocab, self.dev) for c in cks],
                                     list(cks[0]['vocab']), self.dev, False)
        self.beam, self.lp, self.rgb = spec['beam'], spec['lp'], spec['rgb']
        self.max_len = self.cfg.decode_max_len or self.cfg.max_tgt_tokens

    def __call__(self, kps: np.ndarray, rgb_feats: np.ndarray | None = None):
        from slt.data import normalize_clip
        from slt.metrics import detokenize
        torch = self.torch
        feats, mask = normalize_clip(kps.astype(np.float32), np.zeros((len(kps), 4), np.float32), self.cfg)
        src = torch.from_numpy(feats).unsqueeze(0).to(self.dev)
        src_pad = torch.from_numpy(~mask).unsqueeze(0).to(self.dev)
        rgb = torch.from_numpy(rgb_feats if rgb_feats is not None else np.zeros((1, 1152), np.float32)).unsqueeze(0).to(self.dev)
        rgb_pad = torch.zeros(1, rgb.size(1), dtype=torch.bool, device=self.dev)
        with torch.no_grad():
            mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in self.members]
            cands = self.nbest(self.members, mems, self.max_len, self.beam, self.dev)[0]
        raw, seq = max(((r, self.vocab.decode(s)) for r, s in cands),
                       key=lambda c: c[0] / (max(1, len(c[1].split())) ** self.lp))
        return SimpleNamespace(text=detokenize(seq), confidence=-raw / (len(seq.split()) + 1), tokenized=seq)


class ModelService:
    mode = 'model'

    def __init__(self, settings):
        import cv2
        import torch
        from rtmlib import Wholebody

        cv2.setNumThreads(1)
        torch.set_num_threads(settings.threads)
        self.translators = {name: Translator(MODELS[name], settings.device) for name in settings.models}
        self.models = {name: dict(label=MODELS[name]['label'], threshold=MODELS[name]['tau'][settings.coverage],
                                  coverage=settings.coverage, beam=MODELS[name]['beam'], length_penalty=MODELS[name]['lp'],
                                  needs_frames=MODELS[name]['rgb'], **MODELS[name]['evidence'])
                       for name in settings.models}
        if any(MODELS[n]['rgb'] for n in settings.models):
            from .rgb_features import RGBFeatures
            self.rgb = RGBFeatures(settings.rgb_device)
        pose_device = 'cuda' if settings.device == 'cuda' else 'cpu'
        options = dict(mode='balanced', backend='onnxruntime', device=pose_device)
        if settings.detector_path:
            options['det'] = check_asset(settings.detector_path, settings.detector_sha256)
            options['det_input_size'] = (640, 640)
        if settings.pose_path:
            options['pose'] = check_asset(settings.pose_path, settings.pose_sha256)
            options['pose_input_size'] = Wholebody.MODE['balanced']['pose_input_size']
        # ONNX Runtime default threading, as in the throughput bench (5.2 fps full pipeline on the M3 Max CPU);
        # the global worker lock in app.py keeps sessions from running concurrently.
        self.pose = Wholebody(**options)

    def extract(self, frame, track=None):
        """Detect on every frame and follow one signer (the tracked extractor's rules)."""
        try:
            boxes = self.pose.det_model(frame)
            count = len(boxes) if boxes is not None else 0
            if count == 0:
                # Masked, unlike training extraction (whole-image pose): empty input decodes to 'one two three .'.
                return np.zeros((133, 3), np.float32), 0
            kp, sc = self.pose.pose_model(frame, bboxes=boxes)
        except Exception:  # as in the research workers: a failed frame is an empty frame
            return np.zeros((133, 3), np.float32), 0
        if sc is None or not len(kp):
            return np.zeros((133, 3), np.float32), 0
        kp, sc = np.asarray(kp, np.float32), np.asarray(sc, np.float32)
        track = {} if track is None else track
        i, _ = track_index(kp, sc, track.get('mid'), track.get('wid'), (frame.shape[1], frame.shape[0]))
        mid, wid, ok = _torso(kp[i], sc[i])
        if ok:
            track['mid'], track['wid'] = mid, wid
        return np.concatenate([kp[i], sc[i][:, None]], axis=1), count

    def translate(self, model, keypoints, times, frames=None):
        idx = grid_indices(times)
        kps = np.stack([keypoints[i] for i in idx]).astype(np.float32)
        visible = np.abs(kps).sum(axis=(1, 2)) > 1e-6
        if visible.mean() < .8:
            raise ValueError('Keep one signer in view for the whole utterance, then try again.')
        rgb_feats, rgb_ms = None, 0.0
        if MODELS[model]['rgb']:
            t0 = time.perf_counter()
            pick = self.rgb.sample_indices(len(idx))
            rgb_feats = self.rgb([decode_frame(frames[idx[j]]) for j in pick], kps[pick])
            rgb_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        out = self.translators[model](kps, rgb_feats)
        out.translate_ms, out.rgb_ms, out.frames_used = (time.perf_counter() - t0) * 1000, rgb_ms, len(idx)
        return out


class PreviewService:
    """Explicit UI development only. Production rejects this mode."""
    mode = 'sample'
    models = {'single': dict(label='Interface preview · sample phrases', threshold=3.0, coverage=None,
                             beam=None, length_penalty=None, needs_frames=False)}

    def extract(self, frame, track=None):
        return np.ones((133, 3), np.float32), 1

    def translate(self, model, keypoints, times, frames=None):
        return SimpleNamespace(text='This is a sample phrase.', confidence=1.5, translate_ms=0.0, rgb_ms=0.0,
                               frames_used=len(grid_indices(times)))
