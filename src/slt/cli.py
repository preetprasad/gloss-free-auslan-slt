"""Shared argparse helpers.

`strict_bool` exists because the idiom `type=lambda x: x.lower() == "true"` silently maps every
value that is not literally "true" to False. `--pose_aug 1` therefore turned the treatment OFF while
the run logged success (found 2026-09-16). A flag that selects a treatment must fail on input it
does not understand, not quietly choose the control.
"""
import argparse


def strict_bool(x):
    """Accept only true/false (any case). Anything else is an argparse error."""
    v = str(x).strip().lower()
    if v not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected true or false, got {x!r}")
    return v == "true"
