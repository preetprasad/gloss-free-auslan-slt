"""Minimal repro: does numpy's temporary elision modify a NAMED local array in place under this interpreter?

Found while verifying the pose-cleaning lens: inside a function, `(tail & ~n).sum() / tail.sum()` returned 100% and a
later expression on `tail` returned nan, while the same code at module level gave 18.65%. numpy 1.x decides a binary
operand is a disposable temporary from its reference count (>= 256 KiB arrays only); Python 3.14 loads locals as borrowed
references, so a named local can look like a temporary. Prints whether a named local changed after `a & ~b`.
Run: ~/.pyenv/versions/3.14.3/bin/python verify_numpy_elision_hazard.py
"""
import sys

import numpy as np


def f(size):
    a = np.ones(size, bool)
    b = np.zeros(size, bool)
    b[: size // 2] = True
    before = int(a.sum())
    _ = (a & ~b).sum()
    return before, int(a.sum())


def g(size):
    x = np.ones(size)
    y = np.ones(size)
    before = float(x.sum())
    _ = x + y          # a plain named-operand add
    return before, float(x.sum())


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    for size in (1000, 10_000_000):
        b0, b1 = f(size)
        c0, c1 = g(size // 8)
        print(f"size {size}: bool a.sum() {b0} -> {b1} ({'CHANGED' if b0 != b1 else 'unchanged'}); "
              f"float x.sum() {c0} -> {c1} ({'CHANGED' if c0 != c1 else 'unchanged'})")
