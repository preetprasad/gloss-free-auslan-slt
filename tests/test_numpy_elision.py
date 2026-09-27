"""Guard against numpy issue #28681: on Python 3.14 with numpy 1.x, `z = x + y` inside a function can write into a
named local `x` of >= 256 KiB (results/NUMPY_ELISION_AUDIT_2026-09-26.md). An interpreter that fails this test must not
be used for analysis."""
import numpy as np


def _add(n):
    x = np.ones(n)
    y = np.ones(n)
    z = x + y
    return x, z


def test_binary_op_leaves_named_local_intact():
    x, z = _add(40_000)                     # 320 KB, above numpy's 256 KiB elision threshold
    assert not np.shares_memory(x, z)
    assert np.all(x == 1.0), "numpy temporary elision overwrote a named array (numpy #28681)"
