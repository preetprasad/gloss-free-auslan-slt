"""Export a recorded, reproducible single-model walkthrough for LOCAL inspection.

Dataset media, coordinates and activations stay outside the repository and the
Vercel deployment directory. This is a stored-pose replay, not webcam inference.
Run: python demo/export_walkthrough.py --clip video_75_27
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', default='video_75_27')
    parser.add_argument('--output', type=Path, default=Path.home() / '.cache/auslan-walkthrough')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/tools')]
    from config import Config
    from slt.data import Vocab, normalize_clip
    from slt.preproc_guard import assert_preproc_matches_checkpoints
    from slt.metrics import detokenize
    from ensemble_decode import Member
    from ensemble_mbr_wer import ensemble_nbest
    torch.set_num_threads(2)
    rows = (ROOT / 'results/_local_val_manifest.tsv').read_text().splitlines()
    refs = {Path(row.split('\t')[0]).stem: row.split('\t')[1] for row in rows}
    if args.clip not in refs:
        raise ValueError('Choose an official validation clip from the local manifest.')
    out = args.output.expanduser().resolve()
    # Even an explicit output argument cannot put dataset material in this repo.
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Output must be outside the repository and its deployment root.')
    video = ROOT / f'results/dwpose_experiment/rgb_clips/Signer/{args.clip}_signer.mp4'
    pose_path = ROOT / f'results/val_clips_dwpose/{args.clip}.npz'
    cap = cv2.VideoCapture(str(video))
    fps, width, height, count = (cap.get(key) for key in (5, 3, 4, 7))
    ok, poster = cap.read()
    cap.release()
    kps = np.load(pose_path)['KPS'].astype(np.float32)
    if not ok or len(kps) != int(count):
        raise ValueError('Video and pose must have exactly matching frames.')
    # A frozen, explicitly identified research snapshot, independent of changes
    # to the live service's operating-point selection.
    checkpoint = 'results/ckpts_live/w4A_s202_best.pt'
    checkpoint_sha = '0c245b3701d0c80ce0c8ad79a1c81c964afced5f0ac8c3eabe98f87bd35d65b6'
    with (ROOT / checkpoint).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != checkpoint_sha:
            raise ValueError('Recorded-example checkpoint checksum mismatch.')
    ckpt = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)
    cfg = Config(num_keypoints=133, keypoint_subset='all', coord_norm='frame', device='cpu')
    assert_preproc_matches_checkpoints(cfg, [ckpt])
    vocab = Vocab(ckpt['vocab'])
    member = Member(ckpt, cfg, vocab, torch.device('cpu'))
    feats, mask = normalize_clip(kps, np.zeros((len(kps), 4), np.float32), cfg)
    src, pad = torch.from_numpy(feats)[None], torch.from_numpy(~mask)[None]
    model = member.model
    # Observe the real self-attention inputs, not an approximation made from
    # encoder outputs. Hooks also disable the fused encoder fast path.
    layers, attention_inputs, hooks = [], [], []
    for block in model.transformer.encoder.layers:
        hooks.append(block.register_forward_hook(
            lambda m, i, o: layers.append(o.detach().cpu()[0].numpy())))
        hooks.append(block.self_attn.register_forward_pre_hook(
            lambda m, i: attention_inputs.append(i[0].detach().clone())))
    try:
        with torch.no_grad():
            projected = model.input_proj(src)[0].numpy()
            observed_memory = model.encode(src, pad)
    finally:
        for hook in hooks:
            hook.remove()
    with torch.no_grad():
        memory = model.encode(src, pad)
        torch.testing.assert_close(observed_memory, memory, atol=2e-5, rtol=2e-5)
        attention = []
        for block, q in zip(model.transformer.encoder.layers, attention_inputs, strict=True):
            output, weights = block.self_attn(q, q, q, key_padding_mask=pad,
                                              need_weights=True, average_attn_weights=True)
            original, _ = block.self_attn(q, q, q, key_padding_mask=pad, need_weights=False)
            torch.testing.assert_close(output, original, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Invalid self-attention weights.')
            attention.append(np.round(weights[0].numpy(), 7).tolist())
        candidates = ensemble_nbest([member], [(memory, pad)], cfg.decode_max_len or cfg.max_tgt_tokens,
                                    10, torch.device('cpu'))[0]
    raw, sequence = max(((r, vocab.decode(s)) for r, s in candidates),
                        key=lambda c: c[0] / max(1, len(c[1].split())) ** 0.7)
    confidence = -raw / (len(sequence.split()) + 1)
    matrices = [feats, projected, *layers]
    features = []
    for matrix in matrices:
        # Preserve individual channels and numeric units. Colour scaling is a
        # display choice; it never changes exported values or averages channels.
        scale = max(float(np.abs(matrix).max()), 1e-6)
        features.append({'width': matrix.shape[1], 'scale': scale,
                         'values': np.round(matrix, 5).tolist()})
    def sha(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    trace = {
        'version': 2, 'kind': 'recorded', 'clip': args.clip, 'split': 'official validation',
        'fps': fps, 'frames': len(kps), 'width': width, 'height': height,
        'duration': len(kps) / fps, 'video': 'clip.mp4', 'poster': 'poster.jpg',
        'points': np.round(kps, 3).tolist(), 'features': features,
        'attention': attention, 'attention_heads': model.transformer.encoder.layers[0].self_attn.num_heads,
        'featureLabels': ['Normalised input', 'Learned projection', *[f'Encoder layer {i+1}' for i in range(len(layers))]],
        'reference': refs[args.clip], 'prediction': detokenize(sequence),
        'confidence': confidence, 'threshold': 0.706661,
        'accepted': confidence <= 0.706661,
        'model': 'Pose-only Transformer · wave-4 A, seed 202',
        'checkpoint': checkpoint, 'checkpoint_sha256': checkpoint_sha,
        'beam': 10, 'length_penalty': 0.7,
        'video_sha256': sha(video), 'pose_sha256': sha(pose_path),
        'provenance': 'Stored Auslan-Daily signer video and DWPose. Actual checkpoint decode and encoder outputs. Selected illustrative example; not representative performance. No sign-to-word alignment is inferred.',
        'feature_note': 'All frames and individual channels, without channel averaging. Each feature panel uses its own maximum absolute value for colour scaling. Projection is Linear → LayerNorm → ReLU; scaling and positional encoding precede encoder layer 1. Layer outputs precede the final encoder norm. Attention is the mean over 8 heads, recomputed from each layer’s actual inputs and checked for output parity. It is not causal importance, correctness or a sign-to-word alignment.',
    }
    out.mkdir(parents=True, exist_ok=True)
    # The release uses FMP4, which browsers do not consistently decode. Keep all
    # frames/timestamps while creating a browser-compatible, silent H.264 copy.
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                    '-i', str(video), '-an', '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(out / 'clip.mp4')], check=True)
    trace['display_video_sha256'] = sha(out / 'clip.mp4')
    trace['video_processing'] = 'Silent H.264 display transcode, CRF 18; same frame count and time grid. Poses and model features use original stored data.'
    cv2.imwrite(str(out / 'poster.jpg'), poster)
    (out / 'manifest.json').write_text(json.dumps(trace, separators=(',', ':')))
    print(json.dumps({k: trace[k] for k in ('clip','frames','duration','prediction','reference','confidence','accepted')}))
    print(f'Local-only assets: {out}')


if __name__ == '__main__':
    main()
