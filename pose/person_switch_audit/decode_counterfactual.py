"""Decode video_89_193 with the walkthrough's exact pipeline (w4A_s202, beam 10, LP 0.7, the exporter's selection and
self-CE) under three pose inputs: the stored index-0 pose (must reproduce the recorded 'sally look.' / 0.90282), the
person-tracked re-extraction (job 6590), and the tracked pose cut to frames 1-27, the span where the tracker is on the
signer (verified by eye, see README). Usage: decode_counterfactual.py TRACKED_NPZ"""
import hashlib, sys, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/tools"))
from config import Config
from slt.data import Vocab, normalize_clip
from slt.metrics import detokenize
from ensemble_decode import Member
from ensemble_mbr_wer import ensemble_nbest
torch.set_num_threads(2)
CK = ROOT / "results/ckpts_live/w4A_s202_best.pt"
assert hashlib.sha256(CK.read_bytes()).hexdigest() == "0c245b3701d0c80ce0c8ad79a1c81c964afced5f0ac8c3eabe98f87bd35d65b6"
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame", device="cpu")
vocab = Vocab(ckpt["vocab"]); member = Member(ckpt, cfg, vocab, torch.device("cpu"))
def decode(kps):
    feats, mask = normalize_clip(kps.astype(np.float32), np.zeros((len(kps), 4), np.float32), cfg)
    src, pad = torch.from_numpy(feats)[None], torch.from_numpy(~mask)[None]
    with torch.no_grad():
        mem = member.model.encode(src, pad)
        cands = ensemble_nbest([member], [(mem, pad)], cfg.decode_max_len or cfg.max_tgt_tokens, 10, torch.device("cpu"))[0]
    raw, seq = max(((r, vocab.decode(s)) for r, s in cands), key=lambda c: c[0] / max(1, len(c[1].split())) ** 0.7)
    return detokenize(seq), -raw / (len(seq.split()) + 1), [(vocab.decode(s), round(-r / (len(vocab.decode(s).split()) + 1), 3)) for r, s in sorted(cands, key=lambda c: -c[0] / max(1, len(vocab.decode(c[1]).split())) ** 0.7)[:5]]
stored = np.load(ROOT / "results/val_clips_dwpose/video_89_193.npz")["KPS"]
tracked = np.load(sys.argv[1])["KPS"]
p, c, _ = decode(stored)
assert p == "sally look." and abs(c - 0.9028197526931763) < 1e-4, (p, c)
print(f"GATE OK: stored pose reproduces the walkthrough: {p!r} self-CE {c:.4f}")
for name, k in (("stored (index 0)", stored), ("tracked", tracked), ("tracked, frames 1-27 only", tracked[:27])):
    p, c, top = decode(k)
    print(f"{name:26s} -> {p!r:28s} self-CE {c:.3f} ({'answer' if c <= 0.706661 else 'declined'}) | top-5: {top}")
