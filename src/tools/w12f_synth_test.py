"""Wave 12f: end-to-end test of tools/w12f_probe.py on an INVENTED data set (no real clip, no real number).

It builds a small world in a temporary folder (6 signs, 10 conversational spans each, isolated clips cut from the
same frames, citation recordings, 3-D poses), points the probe's constants at it and runs both measurements through
the same functions the Kaya job calls.  It checks the plumbing and the direction of every arm:
  * the facts gate passes, and fails when the located first frames are off by one
  * gates H, E and T pass; gate E fails when the isolated poses are not the production poses
  * A4 (self-match allowed) is near 100%, far above A2 (own clip excluded): the leak the exclusion removes
  * depth that carries the sign gives reading C3, depth that is noise gives C1
Run:  python tools/w12f_synth_test.py
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w12f_probe as P  # noqa: E402

G, PER, T, B0, E0 = 6, 10, 40, 10, 26
COLS = ["Video_ID", "Query_Gloss", "islr_split", "frames", "width", "height", "Video_Name_Clip", "Begin_Frame",
        "End_Frame", "d_min", "d_other", "n_candidates", "status", "comm_split"]


def pose(rng, proto, frames, noise):
    """(frames, 133, 3): a per-sign trajectory of the hands plus noise, body fixed, confidence 0.9."""
    k = np.zeros((frames, 133, 3), np.float32)
    k[..., :2] = rng.normal(300, 60, (1, 133, 2)) * 0 + np.linspace(100, 500, 133)[None, :, None]
    t = np.linspace(0, 1, frames)[:, None, None]
    k[:, 91:133, :2] += proto[None] * (0.5 + t) + rng.normal(0, noise, (frames, 42, 2))
    k[..., :2] += rng.normal(0, 1.0, (frames, 133, 2))
    k[..., 2] = 0.9
    return k


def build(root, z_signal, bank_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d = {n: os.path.join(root, n) for n in ("prod", "bank", "d3", "mm_train", "mm_valid")}
    for p in d.values():
        os.makedirs(p)
    protos = rng.normal(0, 40, (G, 42, 2)).astype(np.float32)
    zprot = rng.normal(0, 1, (G, 133)).astype(np.float32)
    spans, loc, vid = [], [], 0
    for g in range(G):
        for j in range(PER + 2):                       # 10 spans, then 2 further isolated clips of the sign
            clip = f"video_{g * 20 + j}_{j}"
            k = pose(rng, protos[g], T, 25.0)
            np.savez(os.path.join(d["prod"], clip + ".npz"), KPS=k)
            z = (zprot[g][None] * (1 + np.linspace(0, 1, T)[:, None]) if z_signal else 0) + rng.normal(0, 1.0, (T, 133))
            np.savez(os.path.join(d["d3"], clip + ".npz"), KPS=k + rng.normal(0, 0.5, k.shape).astype(np.float32) * [1, 1, 0],
                     Z=z.astype(np.float32))
            if j < PER:
                spans.append([clip + "_signer.mp4", f"G{g}".title(), B0, E0])
            vid += 1
            np.savez(os.path.join(d["bank"], f"{vid}.npz"), KPS=k[B0:E0] + np.float32([bank_shift, bank_shift, 0]))
            loc.append([vid, f"g{g}", "train", E0 - B0, 640, 480, clip, B0, E0, "0.0000", "", 1, "located", "train"])
        for i, sub in enumerate(("mm_train",) * 12 + ("mm_valid",) * 2):       # 14 citation recordings: slower, cleaner
            np.savez(os.path.join(d[sub], f"{g:02d}{i:03d}.npz"), KPS=pose(rng, protos[g], 80, 3.0))
    vid += 1; loc.append([vid, "g0", "dev", 12, 640, 480, "video_900_1", 3, 15, "0.0000", "", 1, "located", "val"])      # not a TRAIN source
    np.savez(os.path.join(d["bank"], f"{vid}.npz"), KPS=pose(rng, protos[0], 12, 25.0))
    vid += 1; loc.append([vid, "g1", "train", 9, 640, 480, "", -1, -1, "7.1000", "", 1, "unlocated", ""])
    np.savez(os.path.join(d["bank"], f"{vid}.npz"), KPS=pose(rng, protos[1], 9, 25.0))
    vid += 1; loc.append([vid, "g2", "train", 0, 640, 480, "", -1, -1, "", "", 0, "empty", ""])
    np.savez(os.path.join(d["bank"], f"{vid}.npz"), KPS=np.zeros((1, 133, 3), np.float32))
    for name, head, rows in (("spans.csv", ["Video_Name_Clip", "Query_Gloss", "Begin_Frame", "End_Frame"], spans), ("located.csv", COLS, loc)):
        with open(os.path.join(root, name), "w", newline="") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(rows)
    for sub, name in (("mm_train", "Train.json"), ("mm_valid", "Valid.json")):
        json.dump({f[:-4]: f"G{int(f[:2])}" for f in os.listdir(d[sub])}, open(os.path.join(root, name), "w"))
    return d


def run(fn, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(argparse.Namespace(**kw))
    return buf.getvalue()


def main():
    P.GLOSSES45 = tuple(f"g{g}" for g in range(G))
    P.SPAN_SET, P.MMWL_CLIPS = (G * PER, G * PER, G), G * 14
    with tempfile.TemporaryDirectory() as root:
        d = build(root, z_signal=True)
        spans = P.read_spans(os.path.join(root, "spans.csv"), set(P.GLOSSES45))
        P.RECORDED = {}
        for name, grp, hloc in P.VARIANTS:             # the invented world's own A0, so that gate H tests the plumbing
            Xq, yq, clq, _, _ = P.query_set(spans, P.Clips(d["prod"]), P.GROUPS[grp], hloc)
            per = {}
            _, _, n = P.clip_disjoint_eval(Xq, yq, clq, np.random.default_rng(P.SEED), P.FOLDS, per)
            P.RECORDED[name], P.N_RECORDED = int(sum(per.values())), n
        kw = dict(spans=os.path.join(root, "spans.csv"), prod_dir=d["prod"], bank_dir=d["bank"], located=os.path.join(root, "located.csv"),
                  mmwl_labels=[os.path.join(root, "Train.json"), os.path.join(root, "Valid.json")], mmwl_dirs=[d["mm_train"], d["mm_valid"]])
        # the located table alone: the isolated clips are the spans; a table whose first frames are off by one must stop the chain
        txt = run(P.cmd_facts, located=kw["located"], spans=kw["spans"], out=os.path.join(root, "facts.json"), prod_dir=d["prod"])
        assert "FACTS OK" in txt and "GATE H PASSED" in txt, txt
        fj = json.load(open(os.path.join(root, "facts.json")))
        assert fj["safe_spans_with_twin"] == fj["safe_spans_with_twin_of_equal_length"] == G * PER, fj
        rows = list(csv.reader(open(kw["located"])))
        with open(os.path.join(root, "shifted.csv"), "w", newline="") as f:
            csv.writer(f).writerows([rows[0]] + [r[:7] + [int(r[7]) + 1] + r[8:] for r in rows[1:]])
        try:
            run(P.cmd_facts, located=os.path.join(root, "shifted.csv"), spans=kw["spans"], out="", prod_dir="")
        except AssertionError as e:
            assert "not the spotting spans" in str(e), e
        else:
            raise SystemExit("the facts gate did not fire")
        txt = run(P.cmd_iso, out="", gate_only=True, **kw)
        assert "GATES PASSED" in txt, txt
        txt = run(P.cmd_iso, out=os.path.join(root, "iso.json"), gate_only=False, **kw)
        r = json.load(open(os.path.join(root, "iso.json")))
        v = r["variants"][P.PRIMARY]
        a2, a4 = v["A2 isolated, own clip excluded [PRIMARY]"], v["A4 isolated, nothing excluded (self-match; leak size only)"]
        assert r["gate_E"]["median_px"] < 1e-3 and v["queries_with_twin"] == v["queries"] and v["A1_agreement_with_A0"] == 1.0
        assert v["reference_candidates"] == v["reference_rows"] == v["reference_rows_safe"] == G * (PER + 2), v   # the validation-sourced, unlocated and empty clips are never read
        assert a4["top1"] > 0.95 and a4["n"] == a2["n"] == v["queries"], (a2, a4)
        vb = r["variants"]["body"]                     # the body carries no sign here: only a self-match can score
        assert vb["A4 isolated, nothing excluded (self-match; leak size only)"]["top1"] > 0.8 > 0.4 > vb["A2 isolated, own clip excluded [PRIMARY]"]["top1"], vb
        assert vb["tests"]["A4"]["p"] < 1e-6 and vb["tests"]["A2"]["p"] > 0.05 and "A5" in v and v["A5"]["n"] == v["queries"]
        assert set(v["tests"]) == {"A2", "A3", "A2m", "A2_vs_A2m", "A4", "A1", "A5", "A5_vs_A2"} and r["reading"]["same_footage"] is True
        assert r["reading"]["reading"][:2] in ("R1", "R2", "R3") and set(r["within"]) == {"120 spotting glosses", "all glosses of the isolated set"}
        assert "READING" in txt and "length only" in txt and set(v["seed_sensitivity"]) == {str(x) for x in P.SENS_SEEDS}
        assert all(t["n"] == v["queries"] for t in v["seed_sensitivity"].values()), v["seed_sensitivity"]

        txt = run(P.cmd_depth, spans=os.path.join(root, "spans.csv"), prod_dir=d["prod"], d3_dir=d["d3"], out=os.path.join(root, "depth.json"), n_perm=60)
        pr = json.load(open(os.path.join(root, "depth.json")))["45 glosses of the recorded run"][P.PRIMARY]["primary"]
        assert pr["rows_are_the_recorded_run"] and pr["xy_comparable"] and pr["depth_not_constant"] and pr["z_only_p"] < 0.05, pr
        assert pr["reading"].startswith(("C2", "C3")), pr["reading"]
        assert set(pr["seed_sensitivity"]) == {str(x) for x in P.SENS_SEEDS} and all(t["z_only_top1"] > 0.5 for t in pr["seed_sensitivity"].values())
        sec = json.load(open(os.path.join(root, "depth.json")))["all 120 spotting glosses"][P.PRIMARY]["secondary"]
        assert sec["z_only_p"] < 0.05 and "reading" not in sec

    with tempfile.TemporaryDirectory() as root:        # depth that is noise: no identity
        d = build(root, z_signal=False)
        run(P.cmd_depth, spans=os.path.join(root, "spans.csv"), prod_dir=d["prod"], d3_dir=d["d3"], out=os.path.join(root, "depth.json"), n_perm=60)
        pr = json.load(open(os.path.join(root, "depth.json")))["45 glosses of the recorded run"][P.PRIMARY]["primary"]
        assert pr["reading"].startswith("C1") and pr["z_only_p"] >= 0.05, pr

    with tempfile.TemporaryDirectory() as root:        # isolated poses that are NOT the production poses: gate E must stop the run
        d = build(root, z_signal=True, bank_shift=5.0)
        try:
            run(P.cmd_iso, out="", gate_only=True, spans=os.path.join(root, "spans.csv"), prod_dir=d["prod"], bank_dir=d["bank"],
                located=os.path.join(root, "located.csv"), mmwl_labels=[], mmwl_dirs=[])
        except AssertionError as e:
            assert "gate E FAILED" in str(e), e
        else:
            raise SystemExit("gate E did not fire")

    with tempfile.TemporaryDirectory() as root:        # reference clips that silently give no descriptor: the run must stop
        d = build(root, z_signal=True)
        for vid in range(1, 9):                        # 8 of the 72 reference clips lose their pose file (gate E cannot see them)
            os.remove(os.path.join(d["bank"], f"{vid}.npz"))
        try:
            run(P.cmd_iso, out="", gate_only=True, spans=os.path.join(root, "spans.csv"), prod_dir=d["prod"], bank_dir=d["bank"],
                located=os.path.join(root, "located.csv"), mmwl_labels=[], mmwl_dirs=[])
        except AssertionError as e:
            assert "reference set FAILED" in str(e), e
        else:
            raise SystemExit("the reference-set guard did not fire")
    print("w12f_synth_test passed (invented data: facts, gates, arms, readings, depth)")


if __name__ == "__main__":
    main()
