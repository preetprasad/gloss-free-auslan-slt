"""Fixtures for demo/static/engine/activity.js: the 8 kinematic signing-activity features and the frozen model score.

The reference is the research code itself, never a retyping: results/signing_activity/signing_activity.py::features on
float64 copies of the float32 keypoints (the JS reads the same float32 values), and the arithmetic of the model in
results/person_labels/activity_model.json.

Writes
  results/live_onnx/fixtures/activity/<case>.f32      float32 LE keypoints, T x 133 x 3 (gitignored with the other binaries)
  demo/tests/fixtures/activity/expected.json          per case: T, the 8 features (null = NaN), the score, a sha256

Cases: 24 signer clips and 24 Non_Signer crops from the primary population (seeded), each as the whole clip, its first
25 frames, and a copy with every 5th frame lost (all-zero rows, as a person who is briefly not detected); plus the
degenerate clips (no confident shoulders, one frame).

Run: ~/.venvs/honours_py312_twin/bin/python demo/tests/fixtures/make_activity_fixtures.py
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/live_onnx/fixtures/activity"
HERE = Path(__file__).resolve().parent / "activity"
spec = importlib.util.spec_from_file_location("signing_activity", ROOT / "results/signing_activity/signing_activity.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
MODEL = json.load(open(ROOT / "results/person_labels/activity_model.json"))
KIN = MODEL["features"]
assert KIN == sa.KIN, "the shipped model's features must be the research KIN list"


def score(feat):
    x = np.array([feat[k] for k in KIN], float)
    x = np.where(np.isnan(x), MODEL["median"], x)
    return float(MODEL["intercept"] + np.dot((x - MODEL["mean"]) / MODEL["scale"], MODEL["coef"]))


def case(name, k32):
    assert k32.dtype == np.float32 and k32.ndim == 3 and k32.shape[1:] == (133, 3)
    f = sa.features(k32.astype(np.float64))
    feat = {k: (None if not np.isfinite(f[k]) else float(f[k])) for k in KIN}
    (OUT / f"{name}.f32").write_bytes(k32.astype("<f4").tobytes())
    return {"name": name, "T": int(len(k32)), "features": feat, "score": score({k: (np.nan if v is None else v) for k, v in feat.items()}),
            "sha256": hashlib.sha256(k32.astype("<f4").tobytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True); HERE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results/signing_activity/features_oof.csv")
    df = df[df.split == "test"]
    rng = np.random.default_rng(0)
    NS = ROOT / "results/nonsigner"
    picks = []
    for y, folder in ((1, "tracked"), (0, "npz")):
        rows = df[df.y == y].sample(24, random_state=int(rng.integers(1 << 30)))
        picks += [(f"{'sig' if y else 'non'}{i:02d}", NS / folder / f"{r.cid}.npz") for i, r in enumerate(rows.itertuples())]
    cases = []
    for name, path in picks:
        k = np.load(path)["KPS"].astype(np.float32)
        cases.append(case(name, k))
        cases.append(case(name + "_w25", k[:25].copy()))
        g = k.copy(); g[::5] = 0                                   # every 5th frame lost
        cases.append(case(name + "_gap", g))
    dead = np.zeros((40, 133, 3), np.float32); dead[..., :2] = 100.0            # no confident keypoint at all
    cases.append(case("degenerate_noconf", dead))
    one = np.load(picks[0][1])["KPS"][:1].astype(np.float32)
    cases.append(case("degenerate_oneframe", one))
    json.dump({"model": {"features": KIN, "intercept": MODEL["intercept"]}, "cases": cases}, open(HERE / "expected.json", "w"), indent=1)
    nan = sum(v is None for c in cases for v in c["features"].values())
    print(f"{len(cases)} cases, {nan} NaN feature values; wrote {HERE / 'expected.json'} and binaries under {OUT}")


if __name__ == "__main__":
    main()
