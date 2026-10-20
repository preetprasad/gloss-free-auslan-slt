"""Preprocessing agreement guard for decode-time tools."""
# ── preprocessing agreement guard (audit 2026-09-24) ──────────────────────────
# Both ensemble tools build their Config from ARGPARSE DEFAULTS, not from the checkpoints. That is the
# general form of the 2026-09-10 RGB bug: a preprocessing field the caller forgets silently decodes the
# members on input they were never trained on, and still prints a plausible BLEU. Only `rgb_feat_dir`
# was ever guarded. These are the fields that change what the model SEES.
_PREPROC_FIELDS = ("coord_norm", "num_keypoints", "keypoint_subset", "bbox_format", "use_confidence",
                   "use_velocity", "conf_threshold", "max_src_frames", "temporal_pool", "hand_local_norm",
                   "w13_input", "w13_frontend")
# Wave 13 (21 Oct 2026): a C133 checkpoint has the same 399-D shape as a frame-normalised model, so decoding it under a Config that
# does not carry its w13_input printed a plausible BLEU on the wrong input. A checkpoint saved before the fields existed has no such
# key and means "" (the default): it is compared with the decode Config like any other field, so it still passes under the default Config.
_DEFAULT_WHEN_ABSENT = {"w13_input": "", "w13_frontend": ""}


def assert_preproc_matches_checkpoints(cfg, cks, override=False):
    """Every member was trained under some preprocessing; decoding must use the same.

    Raises unless the decode Config agrees with every checkpoint's saved cfg on _PREPROC_FIELDS.
    `override=True` (--allow_preproc_mismatch) downgrades it to a printed warning for the rare
    deliberate case (e.g. probing a checkpoint under a different normalisation).
    """
    bad = []
    for i, c in enumerate(cks):
        saved = c.get("cfg", {}) or {}
        for f in _PREPROC_FIELDS:
            if f not in saved and f not in _DEFAULT_WHEN_ABSENT:
                continue
            want = saved[f] if f in saved else _DEFAULT_WHEN_ABSENT[f]
            got = getattr(cfg, f, _DEFAULT_WHEN_ABSENT.get(f))
            if want != got:
                bad.append(f"  member {i}: {f} checkpoint={want!r} decode={got!r}")
    if not bad:
        print(f"[preproc] OK: decode Config matches all {len(cks)} checkpoints on "
              f"{len(_PREPROC_FIELDS)} preprocessing fields", flush=True)
        return
    msg = ("preprocessing mismatch between the decode Config and the checkpoints:\n" + "\n".join(bad) +
           "\n\nThe decode Config is built from argparse DEFAULTS, so a field you did not pass on the "
           "command line silently differs from training. Pass the missing flags (most often "
           "--coord_norm frame --num_keypoints 133), or --allow_preproc_mismatch if this is deliberate.")
    if override:
        print("[preproc] WARNING (overridden): " + msg, flush=True)
    else:
        raise SystemExit("[preproc] " + msg)
