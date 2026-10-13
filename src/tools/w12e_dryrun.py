"""Wave 12e dry run: the whole local half of the pipeline (manifest -> gates -> build -> analyse) on a stand-in for
the Kaya output that contains NO ensemble decode of a non-signer crop. The stand-in gives every negative the n-best
list of its own signer clip (from the saved S6 dumps), optionally shifted, so the expected reading is known:

  same   negatives exactly as confident as their signers     -> R near 1
  low    every negative far below every threshold            -> FV = 0, WORKS (strong)
  high   every negative above every threshold                -> FV = 1, INVERTED

Real inputs: the pose files, the saved dumps, the manifests, the dataset tables and (if --redecode is given) local
re-decodes of the validation and test signers, which the build stage compares row by row with the saved dumps.
Feature files of the negatives and of the tracked signers are random stand-ins of the right shape.

  python tools/w12e_dryrun.py [--redecode DIR_WITH local_dump_val local_dump_test] [--scenarios same,low,high]
"""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import w12e_nonsigner_s6 as w          # noqa: E402
import nonsigner_gate as g             # noqa: E402

RES = w.RES
TOOL = [sys.executable, str(HERE / "w12e_nonsigner_s6.py")]


def run(*args):
    r = subprocess.run(TOOL + [str(x) for x in args], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-600:])
    assert r.returncode == 0, args
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redecode", default="")
    ap.add_argument("--scenarios", default="same,low,high")
    a = ap.parse_args()
    # the job scripts pin this tool and the extractor by md5: a stale pin must fail here, not on the cluster
    kaya = HERE.parent.parent / "jobs" / "slurm"
    for sb in ("w12e_extract.sbatch", "w12e_decode.sbatch"):
        assert f"TOOL_MD5={w.md5(HERE / 'w12e_nonsigner_s6.py')} " in (kaya / sb).read_text(), f"{sb}: stale TOOL_MD5"
    assert w.md5(w.ROOT / "results/dwpose_experiment/extract_rgb_feats.py") in (kaya / "w12e_extract.sbatch").read_text()
    print("job-script pins match the local tool and extractor")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); k = t / "kaya"; k.mkdir()
        # ---- inputs in the Kaya layout: the registered table (with train rows, which must be ignored) and manifests
        neg = g.negatives()
        with open(t / "expected.tsv", "w") as f:
            f.writelines(f"{r.Person_Video_Clip}\t{r.Video_Clip_Name}\t{r.Split}\n" for r in neg.itertuples())
            f.write("video_17_2_2_nonsigner.mp4\tvideo_17_2\ttrain\n")
        val, test = RES / "_local_val_manifest.tsv", RES / "TEST_manifest_official.tsv"
        for src, name in ((val, "trk_val.tsv"), (test, "trk_test.tsv")):
            with open(t / name, "w") as f:
                f.writelines(f"{g.NS / 'tracked' / (Path(p).stem + '.npz')}\t{x}\n" for p, x in w.read_tsv(src))
        nm, sm = t / "neg_mp4" / "Non_Signer", t / "sig_mp4" / "Signer"
        nm.mkdir(parents=True); sm.mkdir(parents=True)
        for c in neg.Person_Video_Clip:
            (nm / c).touch()
        for p, _ in w.read_tsv(val) + w.read_tsv(test):
            (sm / f"{Path(p).stem}_signer.mp4").touch()
        run("manifest", "--expected", t / "expected.tsv", "--neg_dir", g.NS / "npz", "--val_manifest", val, "--test_manifest", test,
            "--trk_val", t / "trk_val.tsv", "--trk_test", t / "trk_test.tsv", "--neg_mp4_dir", t / "neg_mp4",
            "--sig_mp4_dir", t / "sig_mp4", "--out_dir", k, "--require_prefix", "")
        links = {s: w.read_tsv(k / f"ext_{s}.tsv") for s in ("neg", "trk", "gate")}
        assert [len(links[s]) for s in ("neg", "trk", "gate")] == [1678, 1600, 100]
        assert all(Path(src).exists() and name.endswith("_signer.mp4") for s in links for src, name in links[s])
        meta = json.load(open(k / "negatives_meta.json"))
        order = [r[1] for r in w.read_tsv(k / "negatives.tsv")]
        assert len(order) == 1678 and order == sorted(order) and "video_17_2_2_nonsigner" not in meta
        # ---- feature gates: the real stored features of the signers; random stand-ins elsewhere
        rng = np.random.default_rng(0)
        P = json.load(open(k / "pose_facts.json"))
        for d, sets in (("rgb_neg", ("neg",)), ("rgb_trk", ("trk_val", "trk_test"))):
            (t / d).mkdir()
            for s in sets:
                for c, f in P["facts"][s].items():
                    np.savez(t / d / f"rgb_{c}.npz", FEATURES=rng.normal(size=(min(24, f["T"]), 1152)).astype(np.float32))
        G = k / "gates"
        run("featcheck", "--manifest", val, "--rgb_dir", RES / "rgb_feats_val", "--n", 800, "--name", "feat_sig_val", "--gates", G)
        run("featcheck", "--manifest", test, "--rgb_dir", RES / "rgb_feats_test", "--n", 800, "--name", "feat_sig_test", "--gates", G)
        run("featcheck", "--manifest", k / "negatives.tsv", "--rgb_dir", t / "rgb_neg", "--n", 1678, "--name", "feat_neg", "--gates", G)
        run("featcheck", "--manifest", t / "trk_val.tsv", "--rgb_dir", t / "rgb_trk", "--n", 800, "--name", "feat_trk_val", "--gates", G)
        run("featcheck", "--manifest", t / "trk_test.tsv", "--rgb_dir", t / "rgb_trk", "--n", 800, "--name", "feat_trk_test", "--gates", G)
        test_gate = [(s, n) for s, n in links["gate"] if (RES / "rgb_feats_test" / f"rgb_{n.replace('_signer.mp4', '')}.npz").exists()]
        with open(t / "gate_test_only.tsv", "w") as f:
            f.writelines(f"{s}\t{n}\n" for s, n in test_gate)
        assert len(test_gate) == 50
        run("rgbgate", "--list", t / "gate_test_only.tsv", "--new", RES / "rgb_feats_test", "--stored", RES / "rgb_feats_test", "--gates", G)
        # ---- dumps: signers = local re-decodes (or the saved dumps), tracked = the saved dumps, negatives = stand-in
        saved = {"val": RES / "final_ensemble_validation_decodes", "test": RES / "final_ensemble_test_decodes"}
        for sp in ("val", "test"):
            src = Path(a.redecode) / f"local_dump_{sp}" if a.redecode else saved[sp]
            for name in (f"sig_{sp}", f"trk_{sp}"):
                (k / "dumps" / name).mkdir(parents=True)
                shutil.copy((src if name.startswith("sig") else saved[sp]) / "nbest.jsonl", k / "dumps" / name / "nbest.jsonl")
                (k / "dumps" / name / "DONE").touch()
            run("dumpgate", "--new", k / "dumps" / f"sig_{sp}", "--split", sp, "--saved", saved[sp], "--name", f"dump_sig_{sp}", "--gates", G)
        row = {}
        for sp, man in (("val", val), ("test", test)):
            for (p, _), l in zip(w.read_tsv(man), open(saved[sp] / "nbest.jsonl")):
                row[Path(p).stem] = json.loads(l)["nbest"]
        (k / "dumps" / "neg").mkdir()
        (k / "dumps" / "neg" / "DONE").touch()
        # "same": numerator and denominator move together, so the interval is narrow and may or may not hold 1
        expect = {"same": (None, lambda r: 0.85 < r < 1.15), "low": ("WORKS (strong)", lambda r: r == 0.0),
                  "high": ("INVERTED", lambda r: r > 1.5)}
        for sc in a.scenarios.split(","):
            shift = {"same": lambda raw: raw, "low": lambda raw: raw - 50.0, "high": lambda raw: -1e-6}[sc]
            with open(k / "dumps" / "neg" / "nbest.jsonl", "w") as f:
                for i, c in enumerate(order):
                    f.write(json.dumps({"idx": i, "ref": c, "nbest": [[shift(raw), h] for raw, h in row[meta[c]["parent"]]]}) + "\n")
            out = t / f"out_{sc}"
            w.build(k, out)
            rep, ex = w.analyse(out, skip_single=True)
            r = rep["i_production"]["q0.5"]["test"]
            assert expect[sc][0] in (None, rep["final_reading"]) and expect[sc][1](r["R"]), (sc, rep["final_reading"], r)
            print(f"DRYRUN scenario {sc}: reading {rep['final_reading']}, R {r['R']:.3f} [{r['R_lo']:.3f}, {r['R_hi']:.3f}] -- as expected\n")
        # ---- the build stage must refuse a failed gate, a missing DONE marker and a re-decode that differs
        gate = json.load(open(G / "rgbgate.json")); gate["ok"] = False
        json.dump(gate, open(G / "rgbgate.json", "w"))
        try:
            w.build(k, t / "x"); raise SystemExit("build accepted a failed gate")
        except AssertionError as e:
            print("refused, as it must:", e)
        gate["ok"] = True; json.dump(gate, open(G / "rgbgate.json", "w"))
        (k / "dumps/neg/DONE").unlink()
        try:
            w.build(k, t / "x"); raise SystemExit("build accepted a decode without its DONE marker")
        except AssertionError as e:
            print("refused, as it must:", e)
        (k / "dumps/neg/DONE").touch()
        lines = open(k / "dumps/sig_val/nbest.jsonl").read().splitlines()
        for i in range(5):                                   # five validation rows decoded differently
            r = json.loads(lines[i]); r["nbest"] = [[-0.01, "zzz ."]]; lines[i] = json.dumps(r)
        open(k / "dumps/sig_val/nbest.jsonl", "w").write("\n".join(lines) + "\n")
        try:
            w.build(k, t / "x"); raise SystemExit("build accepted a re-decode that differs from the saved dump")
        except AssertionError as e:
            print("refused, as it must:", str(e)[:160])
    print("DRYRUN PASSED")


if __name__ == "__main__":
    main()
