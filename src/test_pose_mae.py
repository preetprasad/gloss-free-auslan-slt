"""Structural, masking and device tests for the local Auslan pose MAE."""
from __future__ import annotations

import torch

from slt.pose_mae import MultiStreamPoseMAE, make_diff_masks


def run(device):
    valid = torch.tensor([[True, True, True], [True, True, False]], device=device)
    masks = make_diff_masks(valid, streams=3, ratio=0.5, seed=11)
    assert masks.shape == (2, 3, 3)
    assert not bool((masks & ~valid[:, None]).any())
    # DiffMask must not collapse into one identical mask for every stream.
    assert not torch.equal(masks[:, 0], masks[:, 1]) or not torch.equal(
        masks[:, 1], masks[:, 2])

    model = MultiStreamPoseMAE(
        hidden=64, encoder_layers=2, heads=4, ff_size=128,
        decoder_hidden=64, decoder_layers=1, patch_size=4,
        output_dim=128, min_confidence=0.1).to(device)
    source = torch.randn(2, 11, 204, device=device)
    pose = source.reshape(2, 11, 68, 3)
    pose[..., 2] = 0.9
    # Natural missing hand joints must not become reconstruction targets.
    pose[0, :, 26:47, 2] = 0.0
    pose[0, :, 26:47, :2] = 0.0
    pad = torch.zeros(2, 11, dtype=torch.bool, device=device)
    pad[1, 8:] = True
    source[1, 8:] = 0.0
    result = model(source, pad, mask_ratio=0.5, mask_seed=7)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    features, feature_pad = model.extract_features(source, pad)
    assert features.shape == (2, 3, 128)
    assert feature_pad.shape == (2, 3)
    assert result.observed_targets["left_hand"] > 0  # second sample remains observed
    print(f"[ok] pose MAE masks/reconstruction/features/backward on {device}")


if __name__ == "__main__":
    run(torch.device("cpu"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        run(torch.device("mps"))
    print("POSE MAE TESTS PASSED")
