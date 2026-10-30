"""The safety guards must fire.

Each of these exists because its absence once produced a plausible wrong number rather than a
crash. A guard that has stopped firing is worse than no guard, so each one is tested in both
directions: it raises on the bad input AND stays out of the way on the good one.
"""
import argparse
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from config import Config                                          # noqa: E402
from slt.cli import strict_bool                                    # noqa: E402
from slt.preproc_guard import assert_preproc_matches_checkpoints   # noqa: E402


# ── strict_bool ────────────────────────────────────────────────────────────
# `type=lambda x: x.lower() == "true"` maps everything that is not "true" to False, so
# `--pose_aug 1` silently ran the CONTROL while the log reported the treatment
# (found 2026-09-16). A flag that selects a treatment must fail on input it cannot read.

@pytest.mark.parametrize("s, want", [("true", True), ("TRUE", True), ("True", True),
                                     ("  false ", False), ("FALSE", False)])
def test_strict_bool_accepts_only_true_and_false(s, want):
    assert strict_bool(s) is want


@pytest.mark.parametrize("s", ["1", "0", "yes", "no", "y", "n", "", "t", "f", "on", "off", "None"])
def test_strict_bool_rejects_everything_else(s):
    with pytest.raises(argparse.ArgumentTypeError):
        strict_bool(s)


def test_strict_bool_rejection_is_an_argparse_error_not_a_silent_false():
    """The failure mode being guarded against: quietly returning False."""
    try:
        strict_bool("1")
    except argparse.ArgumentTypeError:
        return
    pytest.fail("strict_bool('1') returned instead of raising")


# ── preprocessing agreement ────────────────────────────────────────────────
# Both ensemble tools build their decode Config from ARGPARSE DEFAULTS, not from the
# checkpoints. A preprocessing field the caller forgets decodes members on input they were
# never trained on and still prints a plausible BLEU (the general form of the 2026-09-10
# RGB bug, where two of five members silently ran on a single zero frame).

def _ckpt(**cfg):
    base = {"coord_norm": "frame", "num_keypoints": 133, "keypoint_subset": "all",
            "bbox_format": "xywh", "use_confidence": True, "max_src_frames": 256}
    base.update(cfg)
    return {"cfg": base}


def test_preproc_guard_passes_when_the_decode_config_matches(capsys):
    cfg = Config(coord_norm="frame", num_keypoints=133)
    assert_preproc_matches_checkpoints(cfg, [_ckpt(), _ckpt()])
    assert "[preproc] OK" in capsys.readouterr().out


def test_preproc_guard_raises_on_a_mismatched_checkpoint():
    cfg = Config(coord_norm="frame", num_keypoints=133)
    with pytest.raises(SystemExit) as e:
        assert_preproc_matches_checkpoints(cfg, [_ckpt(), _ckpt(coord_norm="bbox")])
    msg = str(e.value)
    assert "preprocessing mismatch" in msg
    assert "member 1" in msg and "coord_norm" in msg


def test_preproc_guard_names_every_offending_field_and_member():
    cfg = Config(coord_norm="frame", num_keypoints=133)
    with pytest.raises(SystemExit) as e:
        assert_preproc_matches_checkpoints(
            cfg, [_ckpt(num_keypoints=136), _ckpt(coord_norm="bbox", max_src_frames=512)])
    msg = str(e.value)
    for expected in ("member 0: num_keypoints", "member 1: coord_norm", "member 1: max_src_frames"):
        assert expected in msg, msg


def test_preproc_guard_override_warns_instead_of_raising(capsys):
    cfg = Config(coord_norm="frame", num_keypoints=133)
    assert_preproc_matches_checkpoints(cfg, [_ckpt(coord_norm="bbox")], override=True)
    assert "WARNING (overridden)" in capsys.readouterr().out


def test_preproc_guard_ignores_fields_absent_from_an_old_checkpoint():
    """Checkpoints predating a field must not be rejected for lacking it."""
    cfg = Config(coord_norm="frame", num_keypoints=133)
    assert_preproc_matches_checkpoints(cfg, [{"cfg": {"coord_norm": "frame"}}])


# ── normalize_clip on a clip with no valid frames ──────────────────────────
# Before the 2026-09-24 audit this returned a single zero frame, so a whole run could train
# and decode on no input at all and still print a plausible BLEU. The canonical trigger is
# coord_norm="bbox" on DWPose keypoints, whose BBOX is all zeros by construction.
# Synthetic data only — no dataset content is needed or used.

def _synthetic_clip(T=8, K=133, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    kps = np.empty((T, K, 3), dtype=np.float32)
    kps[..., :2] = rng.uniform(50, 400, size=(T, K, 2))   # plausible pixel coordinates
    kps[..., 2] = 0.9                                     # confident
    return kps


def test_normalize_clip_raises_when_every_frame_is_invalid():
    from slt.data import normalize_clip
    kps = _synthetic_clip()
    bbox = np.zeros((kps.shape[0], 4), dtype=np.float32)   # the all-zero DWPose BBOX
    cfg = Config(coord_norm="bbox", bbox_format="xywh", num_keypoints=133)
    with pytest.raises(ValueError, match="all frames invalid"):
        normalize_clip(kps, bbox, cfg)


def test_normalize_clip_error_names_the_cause_and_the_fix():
    from slt.data import normalize_clip
    kps = _synthetic_clip()
    bbox = np.zeros((kps.shape[0], 4), dtype=np.float32)
    cfg = Config(coord_norm="bbox", bbox_format="xywh", num_keypoints=133)
    with pytest.raises(ValueError) as e:
        normalize_clip(kps, bbox, cfg)
    assert "--coord_norm frame" in str(e.value)


def test_normalize_clip_succeeds_on_the_same_clip_under_frame_normalisation():
    """The guard must not fire on the configuration the project actually trains with."""
    from slt.data import normalize_clip
    kps = _synthetic_clip()
    bbox = np.zeros((kps.shape[0], 4), dtype=np.float32)
    cfg = Config(coord_norm="frame", num_keypoints=133)
    feats, mask = normalize_clip(kps, bbox, cfg)
    assert feats.shape == (kps.shape[0], 133 * 3)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    assert mask.all()


def test_normalize_clip_succeeds_on_a_valid_bbox():
    from slt.data import normalize_clip
    kps = _synthetic_clip()
    bbox = np.tile(np.array([[40.0, 40.0, 400.0, 400.0]], dtype=np.float32), (kps.shape[0], 1))
    cfg = Config(coord_norm="bbox", bbox_format="xywh", num_keypoints=133)
    feats, mask = normalize_clip(kps, bbox, cfg)
    assert feats.shape[0] == kps.shape[0] and np.isfinite(feats).all()


def test_normalize_clip_rejects_a_keypoint_count_mismatch():
    from slt.data import normalize_clip
    kps = _synthetic_clip(K=136)
    cfg = Config(coord_norm="frame", num_keypoints=133)
    with pytest.raises(ValueError, match="num_keypoints"):
        normalize_clip(kps, np.zeros((kps.shape[0], 4), dtype=np.float32), cfg)
