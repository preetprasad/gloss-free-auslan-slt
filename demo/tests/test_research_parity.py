"""The live service reproduces the recorded validation numbers. Skipped unless RESEARCH_PARITY=1; with it set,
a missing asset is a failure, never a skip.

    cd . && RESEARCH_PARITY=1 ~/.pyenv/versions/3.14.3/bin/python -m pytest demo/tests/test_research_parity.py -q -s

(a) single model and (b) S6 ensemble (stored Kaya RGB features) through ModelService.translate on the 800 validation
clips: BLEU-4, all three thresholds and every hypothesis against the committed dumps; (c) live DINOv2 crop features
against the stored ones; (d) the committed live-feature S6 validation dump; (e) end to end on 8 validation signer
videos: real DWPose + tracker + JPEG frames + live DINOv2, against the stored-pose decode and the research tracker.
Several minutes on CPU.
"""
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo.service import MODELS, ROOT, ModelService, Translator, decode_frame

VAL = ROOT / 'results/_local_val_manifest.tsv'
POSE = ROOT / 'results/val_clips_dwpose'
RGB = ROOT / 'results/rgb_feats_val'
VIDEO = ROOT / 'results/dwpose_experiment/rgb_clips/Signer'
TRACKED = ROOT / 'results/nonsigner/tracked'   # research person-tracked DWPose (dwpose_extract_worker_tracked.py)
LIVE = ROOT / 'results/live_demo_2026-09-24'
DUMPS = {'single': LIVE / 'dump_w4A_s202_val/hyps_LP=0.7.jsonl', 'ensemble': ROOT / 'results/final_ensemble_validation_decodes/hyps_LP=0.3.jsonl'}
LIVE_RGB_DUMP = LIVE / 'dump_S6_val_live_rgb/hyps_LP=0.3.jsonl'
ASSETS = [VAL, POSE, RGB, VIDEO, TRACKED, LIVE_RGB_DUMP, *DUMPS.values()] + \
         [ROOT / p for spec in MODELS.values() for p, _ in spec['ckpts']]
pytestmark = pytest.mark.skipif(os.environ.get('RESEARCH_PARITY') != '1', reason='set RESEARCH_PARITY=1 to run')


@pytest.fixture(autouse=True, scope='module')
def assets_present():
    missing = [str(p.relative_to(ROOT)) for p in ASSETS if not p.exists()]
    assert not missing, f'RESEARCH_PARITY=1 but research assets are missing: {missing}'


def manifest():
    rows = [line.rstrip('\n').split('\t', 1) for line in open(VAL)]
    return [(Path(p).stem, ref) for p, ref in rows]


def dump(path):
    return [json.loads(line) for line in open(path)]


def metrics():
    if str(ROOT / 'src') not in sys.path: sys.path.insert(0, str(ROOT / 'src'))
    import slt.metrics
    return slt.metrics


def bleu(hyps, refs):
    """src/tools/ensemble_selective.py::bleu, exactly."""
    m = metrics()
    return m.corpus_bleu4([m.detokenize(x) for x in hyps], [m.detokenize(x) for x in refs], tokenize='13a')


def check(name, outs, refs, t0):
    # The text a user sees is the scored hypothesis (bleu() detokenises again; detokenize is idempotent here).
    assert all(o.text == metrics().detokenize(o.tokenized) for o in outs)
    recorded = dump(DUMPS[name])
    assert [r['ref'] for r in recorded] == refs
    same = sum(o.tokenized == r['hyp_beam'] for o, r in zip(outs, recorded))
    score = bleu([o.text for o in outs], refs)
    taus = {q: float(np.quantile([o.confidence for o in outs], q)) for q in MODELS[name]['tau']}
    print(f'\n[{name}] n={len(outs)} validation BLEU-4 {score:.3f} (recorded {MODELS[name]["evidence"]["validation_bleu4"]}); '
          f'{same}/{len(outs)} hypotheses identical to {DUMPS[name].parent.name}; self-CE quantiles '
          + ', '.join(f'q{q:g} {t:.6f} (MODELS {MODELS[name]["tau"][q]})' for q, t in sorted(taus.items()))
          + f'; {time.perf_counter() - t0:.0f} s')
    assert abs(score - MODELS[name]['evidence']['validation_bleu4']) <= 0.005
    for q, t in taus.items():
        assert abs(t - MODELS[name]['tau'][q]) <= 1e-4, q
    return same


def jpeg_url(img_bgr, quality=75):
    import cv2
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


def test_single_model_through_the_live_path():
    t0 = time.perf_counter()
    service = ModelService.__new__(ModelService)
    service.translators = {'single': Translator(MODELS['single'])}
    outs, refs = [], []
    for clip, ref in manifest():
        kps = np.load(POSE / f'{clip}.npz')['KPS']
        out = service.translate('single', list(kps), np.arange(len(kps)) * 40)
        assert out.frames_used == len(kps)
        outs.append(out); refs.append(ref)
    assert check('single', outs, refs, t0) == len(outs)


def test_ensemble_through_the_live_path_with_stored_rgb_features():
    from PIL import Image
    from demo.rgb_features import RGBFeatures
    buf = io.BytesIO(); Image.new('RGB', (8, 8)).save(buf, 'JPEG')
    blank = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

    class StoredRGB(SimpleNamespace):  # ModelService.rgb stand-in: the live sampling, then the stored Kaya features
        sample_indices = RGBFeatures.sample_indices
        def __call__(self, frames, kps): return self.features

    t0 = time.perf_counter()
    service = ModelService.__new__(ModelService)
    service.translators, service.rgb = {'ensemble': Translator(MODELS['ensemble'])}, StoredRGB(frames=24)
    outs, refs = [], []
    for clip, ref in manifest():
        kps = np.load(POSE / f'{clip}.npz')['KPS']
        service.rgb.features = np.load(RGB / f'rgb_{clip}.npz')['FEATURES']
        outs.append(service.translate('ensemble', list(kps), np.arange(len(kps)) * 40, [blank] * len(kps))); refs.append(ref)
    assert check('ensemble', outs, refs, t0) == len(outs)


def test_live_rgb_features_match_stored():
    import cv2
    from demo.rgb_features import RGBFeatures
    t0 = time.perf_counter()
    rgb = RGBFeatures('cpu')
    low = {}
    for clip, _ in manifest()[:5]:
        kps = np.load(POSE / f'{clip}.npz')['KPS'].astype(np.float32)
        cap, frames = cv2.VideoCapture(str(VIDEO / f'{clip}_signer.mp4')), []
        while (read := cap.read())[0]: frames.append(read[1])
        cap.release()
        pick = rgb.sample_indices(min(len(frames), len(kps)))
        live = rgb([frames[i] for i in pick], kps[pick]).reshape(len(pick), 3, 384)
        stored = np.load(RGB / f'rgb_{clip}.npz')['FEATURES'].reshape(-1, 3, 384)
        assert live.shape == stored.shape
        zero_live, zero_stored = ~live.any(-1), ~stored.any(-1)
        assert (zero_live == zero_stored).all(), clip
        cos = (live * stored).sum(-1) / np.maximum(np.linalg.norm(live, axis=-1) * np.linalg.norm(stored, axis=-1), 1e-12)
        per_slot = [f'{name} {cos[~zero_live[:, j], j].min():.4f}/{cos[~zero_live[:, j], j].mean():.4f}'
                    for j, name in enumerate(('signer', 'left', 'right'))]
        print(f'\n[rgb] {clip}: {len(pick)} frames, {int(zero_live.sum())} empty crops; min/mean cosine ' + ', '.join(per_slot))
        # The published claim (docs/DEMO_DEPLOYMENT.md, evidence.json): slot means >= 0.994, every crop >= 0.9827 (asserted at 0.982).
        slot_means = [cos[~zero_live[:, j], j].mean() for j in range(3)]
        if cos[~zero_live].min() < 0.982 or min(slot_means) < 0.994: low[clip] = round(float(cos[~zero_live].min()), 4)
    print(f'[rgb] {time.perf_counter() - t0:.0f} s')
    assert not low, f'live features weaker than the published agreement: {low}'


def test_live_feature_s6_gate_committed_dump():
    """S6 on validation with DINOv2 features recomputed through demo/rgb_features.py (scripts/rgb_recompute_val.py)."""
    live, stored = dump(LIVE_RGB_DUMP), dump(DUMPS['ensemble'])
    refs = [r['ref'] for r in stored]
    assert [r['ref'] for r in live] == refs
    m = metrics()
    scores = [bleu([r['hyp_beam'] for r in rows], refs) for rows in (live, stored)]
    wers = [m.corpus_wer([m.detokenize(r['hyp_beam']) for r in rows], [m.detokenize(x) for x in refs]) for rows in (live, stored)]
    same = sum(a['hyp_beam'] == b['hyp_beam'] for a, b in zip(live, stored))
    print(f'\n[S6 live features] BLEU-4 {scores[0]:.3f} vs stored {scores[1]:.3f}; WER {wers[0]:.4f} vs {wers[1]:.4f}; '
          f'{same}/{len(refs)} identical')
    assert abs(scores[0] - 12.291) <= 0.005 and abs(scores[1] - 12.470) <= 0.005
    assert abs(wers[0] - 0.9157) <= 5e-5 and abs(wers[1] - 0.9164) <= 5e-5
    assert same == 772


def test_end_to_end_on_validation_signer_videos():
    """Real ModelService: browser-like JPEG frames (960x720 max, quality 0.75, as app.js) through decode_frame, DWPose,
    the tracker, the 25 fps grid and live DINOv2. Recorded: the ensemble matches the stored-pose decode on 8 of 8;
    the single model's count (2 of 8 for w4A_s202 on 2026-09-24) is printed, not asserted."""
    import cv2
    from demo.app import Settings
    service = ModelService(Settings(mode='model'))
    agree, far, compared, pose_ms = dict.fromkeys(service.translators, 0), 0, 0, []
    clips = manifest()[:8]
    for clip, ref in clips:
        cap, data = cv2.VideoCapture(str(VIDEO / f'{clip}_signer.mp4')), []
        while (read := cap.read())[0]:
            h, w = read[1].shape[:2]; scale = min(1, 960 / w, 720 / h)
            img = cv2.resize(read[1], (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else read[1]
            data.append(jpeg_url(img))
        cap.release()
        track, kps, t0 = {}, [], time.perf_counter()
        for d in data: kps.append(service.extract(decode_frame(d), track)[0])
        pose_ms.append((time.perf_counter() - t0) * 1000 / len(data))
        # Tracker parity: the followed person matches the research tracked extractor (shoulder midpoint within 0.5 widths).
        research = np.load(TRACKED / f'{clip}.npz')['KPS']
        n = min(len(kps), len(research))
        live = np.stack(kps[:n]); live[:, :, :2] /= scale
        both = (live[:, 5:7, 2] > .3).all(1) & (research[:n, 5:7, 2] > .3).all(1)
        width = np.maximum(np.linalg.norm(research[:n, 5, :2] - research[:n, 6, :2], axis=1), 1e-3)
        dist = np.linalg.norm(live[:, 5:7, :2].mean(1) - research[:n, 5:7, :2].mean(1), axis=1) / width
        far += int((dist[both] > .5).sum()); compared += int(both.sum())
        stored = np.load(POSE / f'{clip}.npz')['KPS'].astype(np.float32)
        for name, translator in service.translators.items():
            out = service.translate(name, kps, np.arange(len(kps)) * 40, data)
            offline = translator(stored, np.load(RGB / f'rgb_{clip}.npz')['FEATURES'] if MODELS[name]['rgb'] else None)
            agree[name] += out.text == offline.text
            print(f'\n[e2e] {clip} {name}: live {out.text!r} | stored-pose {offline.text!r} | ref {ref!r}')
    print(f'[e2e] identical to the stored-pose decode: {agree} of {len(clips)}; tracker off by > 0.5 shoulder widths on '
          f'{far}/{compared} frames; pose {np.mean(pose_ms):.0f} ms/frame')
    assert agree['ensemble'] == len(clips)
    assert far <= 0.01 * compared
