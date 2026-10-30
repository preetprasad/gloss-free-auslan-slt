"""DINOv2 crop features for the RGB-fusion ensemble members.

Same computation as results/dwpose_experiment/extract_rgb_feats.py (the features the S6 fusion
members were trained and evaluated on): up to 24 frames sampled uniformly over the utterance,
three pose-localised crops per frame (signer, left hand, right hand), frozen DINOv2-small
pooler output, concatenated to (T_sampled, 1152). Crop helpers are imported, not copied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'results' / 'dwpose_experiment'))
from extract_rgb_feats import DINO, IMN, crop_box, get_crop  # noqa: E402

DINO_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'  # snapshot verified against the stored Kaya features


class RGBFeatures:
    def __init__(self, device: str = 'cpu', frames: int = 24):
        import cv2
        from transformers import AutoModel
        self.cv2 = cv2
        self.frames = frames
        self.device = torch.device(device)
        # Weights must already be cached; the service never downloads at startup.
        self.model = AutoModel.from_pretrained(DINO, revision=DINO_REVISION, local_files_only=True).to(self.device).eval()
        self.mean = torch.tensor(IMN[0], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMN[1], device=self.device).view(1, 3, 1, 1)

    def sample_indices(self, n_frames: int) -> np.ndarray:
        return np.linspace(0, n_frames - 1, min(self.frames, n_frames)).round().astype(int)

    @torch.no_grad()
    def __call__(self, frames_bgr, kps: np.ndarray) -> np.ndarray:
        """frames_bgr: sequence of BGR uint8 images for the sampled indices only; kps: matching (n,133,3)."""
        crops, slots = [], []
        for fr, k in zip(frames_bgr, kps):
            fr = self.cv2.cvtColor(fr, self.cv2.COLOR_BGR2RGB)
            xy, cf = k[:, :2], k[:, 2]
            for b in (crop_box(xy, cf), crop_box(xy[91:112], cf[91:112]), crop_box(xy[112:133], cf[112:133])):
                c = get_crop(fr, b)
                slots.append(c is not None)
                crops.append(c if c is not None else np.zeros((224, 224, 3), np.uint8))
        t = torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(self.device)
        emb = self.model(pixel_values=(t - self.mean) / self.std).pooler_output.float().cpu().numpy()
        emb[~np.array(slots)] = 0.0
        return emb.reshape(len(kps), -1).astype(np.float32)
