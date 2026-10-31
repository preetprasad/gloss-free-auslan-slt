"""Wave G read-out: signer-guided DWPose (G) against C0 (detection 0) and V2 (tracker v2). Pre-registration:
results/PREREG_WAVEG_SIGNERGUIDED_2026-10-26.md section 3 (decision rules = round 2, the project log section 9).

Reuses the wave-12i statistics module unchanged (bleu, tstats, line, did, load_bins, ...). Runs in $G/waveG/runs: <arm>_s<seed> with arm in
{c0, v2, g}; c0 and v2 read the wave-9 manifests, g reads waveG/manifests; every arm is decoded on its OWN pose.

  python tools/wG_readout.py selftest
  python tools/wG_readout.py report --runs RUNS --w9man W9_MANIFESTS --gman G_MANIFESTS [--bins B] [--arm_stats ARM_STATS_G] [--readouts_dir D]
                                    [--marker STAGE1_PASS_FILE] [--boot N]

Stage 1 (seeds 42 101 202, all three arms): prints the per-seed table, G - C0 (the criterion), G - V2, V2 - C0, the heavy-bin
difference-in-differences and the futility verdict; writes the --marker file ONLY if the 3-seed mean beam-5 validation delta G - C0 > 0.
Stage 2 (seeds 303 404, arms C0 and G; only if the marker exists): the 5-seed adoption rule. Nothing is printed for a missing run.
"""
from __future__ import annotations

import argparse, json, math, os, statistics as st, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import w12i_heavy_did as H  # noqa: E402
from w12i_heavy_did import bleu, rows, sha, did, line, tstats, sub, KEY, BINS  # noqa: E402

ARMS = ("c0", "v2", "g")
S1, S2 = (42, 101, 202), (303, 404)
ALPHA = 0.05
G_SHARE = 0.3                      # exploratory G-specific bin: clips on which G != C0 on more than 30% of frames


def man(w9, gm, arm, split):
    return os.path.join(gm if arm == "g" else w9, f"{arm}_{split}.tsv")


def done(runs, arm, s):
    return os.path.exists(os.path.join(runs, f"{arm}_s{s}", "DONE"))


def load(runs, w9, gm, arms, seeds):
    """Hypotheses after the gates (own pose, manifest and checkpoint named in the dump, references = text, exact re-score)."""
    D = {}
    for split in ("val", "test"):
        order, texts = rows(man(w9, gm, "c0", split))
        for a in ARMS:
            assert rows(man(w9, gm, a, split)) == (order, texts), f"{split}: {a} manifest differs from c0 in clip order or text"
        D[split, "order"], D[split, "refs"] = order, texts
        for a in arms:
            for s in seeds[a]:
                j = json.load(open(os.path.join(runs, f"{a}_s{s}", f"beam5_{split}_best.json")))
                assert j["manifest"].endswith(f"/{a}_{split}.tsv"), (a, s, split, j["manifest"])
                assert j["checkpoint"].endswith(f"/{a}_s{s}/best.pt"), j["checkpoint"]
                assert j["beam"] == 5 and j["refs"] == texts and len(j["beam_hyps"]) == len(texts), (a, s, split)
                assert abs(bleu(j["beam_hyps"], texts) - j["beam_result"]["metrics"][KEY]) < 1e-6, f"re-score {a} s{s} {split}"
                D[split, a, s] = j["beam_hyps"]
    return D


def pair_seeds(runs, a, b, pool):
    return [s for s in pool if done(runs, a, s) and done(runs, b, s)]


def whole(D, split, a, s):
    return bleu(D[split, a, s], D[split, "refs"])


def g_bin(arm_stats, order):
    S = {r["clip"]: r for r in json.load(open(arm_stats))}
    hv = [i for i, c in enumerate(order) if S[c]["c0_ne_g"] / S[c]["T"] > G_SHARE]
    rest = [i for i in range(len(order)) if i not in set(hv)]
    return hv, rest


def stage1_verdict(val_deltas):
    m = st.mean(val_deltas)
    return m > 0, m


def adopt_verdict(test_d, val_d, fresh_test_d):
    p = tstats(test_d)["p_one_sided"]; vm = st.mean(val_d)
    ok = p < ALPHA and vm > 0 and all(x > 0 for x in fresh_test_d)
    return dict(p_test=p, val_mean=vm, fresh_positive=all(x > 0 for x in fresh_test_d), adopt=ok)


def report(runs, w9, gm, bins, arm_stats, readouts_dir, marker, n_boot=0):
    s1 = {a: [s for s in S1 if done(runs, a, s)] for a in ARMS}
    if any(tuple(s1[a]) != S1 for a in ARMS):
        print("INCOMPLETE: no score printed. Stage 1 needs DONE for c0, v2 and g at seeds 42 101 202; missing: " +
              str([f"{a}_s{s}" for a in ARMS for s in S1 if not done(runs, a, s)]))
        return 2
    seeds = {a: list(S1) for a in ARMS}
    stage2 = [s for s in S2 if done(runs, "c0", s) and done(runs, "g", s)]
    if stage2:
        if not os.path.exists(marker):
            print(f"REFUSED: Stage-2 runs exist ({stage2}) but {marker} does not (Stage 1 did not pass, or was not read out): no Stage-2 score printed")
            stage2 = []
        else:
            seeds["c0"] = list(S1) + stage2; seeds["g"] = list(S1) + stage2
    D = load(runs, w9, gm, ARMS, seeds)
    print(f"GATES PASSED: {sum(len(v) for v in seeds.values())} runs DONE, every beam-5 BLEU re-scored exactly from its dump, references = manifest text, "
          f"each arm decoded on its own pose, c0/v2/g manifests identical in order and English text")
    for split in ("val", "test"):
        print(f"\n== {split}: beam-5 BLEU-4 (best.pt), whole split")
        for a in ARMS:
            print(f"  {a:3s} " + " ".join(f"s{s} {whole(D, split, a, s):.3f}" for s in seeds[a]))
    print("\n== Stage 1 (seeds 42 101 202): paired deltas, whole split")
    for (a, b, lab) in (("g", "c0", "G - C0  (PRIMARY: validation)"), ("g", "v2", "G - V2  (secondary)"), ("v2", "c0", "V2 - C0 (replication of wave 9)")):
        for split in ("val", "test"):
            print(f"  {lab:34s} {split:4s} {line([whole(D, split, a, s) - whole(D, split, b, s) for s in S1])}")
    ok, m = stage1_verdict([whole(D, "val", "g", s) - whole(D, "val", "c0", s) for s in S1])
    print(f"\nSTAGE 1 (futility): 3-seed mean beam-5 validation G - C0 = {m:+.3f} -> " + ("PASSES futility (Stage 2, seeds 303 404, C0 and G, may be run)" if ok else "STOP: G does not pass futility"))
    if ok:
        open(marker, "w").write(f"STAGE1_PASS mean val delta G-C0 {m:+.4f} seeds 42 101 202\n")
    elif os.path.exists(marker):
        os.remove(marker)
    # heavy-bin DiD (frozen bins), every pair, own pose
    B = {}
    for split in ("val", "test"):
        B[split], dg = H.load_bins(bins, split, D[split, "order"])
        print(f"\n== {split}: heavy-switch bins (frozen from v1; n " + ", ".join(f"{k} {len(v)}" for k, v in B[split].items()) + ")")
        for (a, b) in (("g", "c0"), ("g", "v2"), ("v2", "c0")):
            sd = S1
            print(f"  {a}-{b} none-bin delta        {line(did(D, split, sd, a, b, B[split]['none'], None))}")
            for name in ("light", "medium", "heavy"):
                print(f"  {a}-{b} {name:6s} DiD vs none  {line(did(D, split, sd, a, b, B[split][name], B[split]['none']))}")
        if arm_stats:
            hv, rest = g_bin(arm_stats, D[split, "order"])
            print(f"  [exploratory] G-specific bin: clips with G != C0 on > {int(100 * G_SHARE)}% of frames: n={len(hv)} (rest {len(rest)})")
            for (a, b) in (("g", "c0"), ("g", "v2")):
                print(f"  {a}-{b} G-bin DiD vs rest  {line(did(D, split, S1, a, b, hv, rest))}")
    if readouts_dir:
        print("\n== whole-split secondaries paired per seed (tools/arm_readouts.py): G - C0 and G - V2")
        for split in ("val", "test"):
            rd = dict((l.split(" ", 1)[0], json.loads(l.split(" ", 1)[1])) for l in open(os.path.join(readouts_dir, f"readouts_{split}.txt")) if " {" in l)
            if split == "test":
                assert all(rd[f"{a}_s{s}"]["unseen_n"] == 570 for a in ARMS for s in seeds[a]), "unseen-target n is not 570"
            for k in ("bleu4_unseen", "content_precision", "content_recall", "len_ratio", "wer"):
                for (a, b) in (("g", "c0"), ("g", "v2")):
                    print(f"  {split:4s} {k:18s} {a}-{b} {line([rd[f'{a}_s{s}'][k] - rd[f'{b}_s{s}'][k] for s in S1])}")
    if stage2:
        S = list(S1) + stage2
        print(f"\n== Stage 2 (seeds {stage2}): the adoption rule on G - C0 over the {len(S)} seeds")
        tt = [whole(D, "test", "g", s) - whole(D, "test", "c0", s) for s in S]
        vv = [whole(D, "val", "g", s) - whole(D, "val", "c0", s) for s in S]
        fresh = [whole(D, "test", "g", s) - whole(D, "test", "c0", s) for s in stage2]
        print(f"  test       {line(tt)}\n  validation {line(vv)}\n  fresh seeds, test: " + " ".join(f"{x:+.3f}" for x in fresh))
        v = adopt_verdict(tt, vv, fresh)
        print(f"  (a) one-sided paired t on test, p {v['p_test']:.4f} < {ALPHA}: {'MET' if v['p_test'] < ALPHA else 'NOT MET'} | (b) 5-seed validation mean {v['val_mean']:+.3f} > 0: "
              f"{'MET' if v['val_mean'] > 0 else 'NOT MET'} | (c) both fresh seeds positive on test: {'MET' if v['fresh_positive'] else 'NOT MET'}")
        print(f"  VERDICT: {'ADOPT G' if v['adopt'] else 'DO NOT ADOPT G'}")
        for split in ("val", "test"):
            print(f"  heavy DiD G-C0, {len(S)} seeds, {split:4s} {line(did(D, split, S, 'g', 'c0', B[split]['heavy'], B[split]['none']))}")
    return 0


def selftest():
    refs = [f"this is sentence number {w} of the set ." for w in "one two three four five six seven eight".split()]
    wrong = "that was great ."
    with tempfile.TemporaryDirectory() as tmp:
        w9, gm, runs = (os.path.join(tmp, x) for x in ("w9", "g", "runs")); os.makedirs(w9); os.makedirs(gm)
        for a in ARMS:
            for split in ("val", "test"):
                open(os.path.join(gm if a == "g" else w9, f"{a}_{split}.tsv"), "w").write("".join(f"/x/{a}/clip{i}.npz\t{r}\n" for i, r in enumerate(refs)))
        order = [f"clip{i}" for i in range(8)]
        bins = os.path.join(tmp, "bins.json")
        json.dump({sp: {"order_sha256": sha(order), "degenerate": [], "bins": {"none": order[:4], "light": order[4:5], "medium": order[5:6], "heavy": order[6:]}}
                   for sp in ("val", "test")}, open(bins, "w"))
        stats = os.path.join(tmp, "arm_stats_g.json")
        json.dump([{"clip": c, "T": 10, "c0_ne_g": 8 if i >= 6 else 0} for i, c in enumerate(order)], open(stats, "w"))
        keep = H.N_BIN; H.N_BIN = {sp: {"none": 4, "light": 1, "medium": 1, "heavy": 2} for sp in ("val", "test")}
        marker = os.path.join(tmp, "STAGE1_PASS")
        try:
            def dump(a, s, hyps):
                d = os.path.join(runs, f"{a}_s{s}"); os.makedirs(d, exist_ok=True)
                for split in ("val", "test"):
                    json.dump({"manifest": f"/m/{a}_{split}.tsv", "checkpoint": f"/r/{a}_s{s}/best.pt", "beam": 5, "beam_hyps": hyps, "refs": refs,
                               "beam_result": {"metrics": {KEY: bleu(hyps, refs)}}}, open(os.path.join(d, f"beam5_{split}_best.json"), "w"))
            def fin(a, s): open(os.path.join(runs, f"{a}_s{s}", "DONE"), "w").close()
            for s in S1 + S2:
                dump("c0", s, refs[:6] + [wrong, wrong]); dump("v2", s, refs[:7] + [wrong]); dump("g", s, refs)
            assert report(runs, w9, gm, bins, stats, "", marker) == 2                           # no DONE: nothing printed
            for s in S1:
                for a in ARMS: fin(a, s)
            assert report(runs, w9, gm, bins, stats, "", marker) == 0 and os.path.exists(marker)  # G better everywhere -> passes futility, marker
            for s in S2: fin("c0", s); fin("g", s)
            assert report(runs, w9, gm, bins, stats, "", marker) == 0                          # stage 2 runs with marker
            os.remove(marker)
            for s in S1: dump("g", s, refs[:6] + [wrong, wrong]); dump("c0", s, refs[:6] + [wrong, wrong])  # G == C0: delta 0, not > 0
            assert report(runs, w9, gm, bins, stats, "", marker) == 0 and not os.path.exists(marker)  # fails futility, marker absent -> stage 2 refused
            D = load(runs, w9, gm, ARMS, {a: list(S1) for a in ARMS})
            idx, _ = H.load_bins(bins, "test", D["test", "order"]); hv, rest = g_bin(stats, D["test", "order"])
            assert hv == [6, 7] and len(rest) == 6
            assert adopt_verdict([1, 2, 1, 2, 1.5], [1] * 5, [1, 2])["adopt"] and not adopt_verdict([1, 2, 1, 2, 1.5], [1] * 5, [1, -2])["adopt"]
            assert not adopt_verdict([1, -2, 1, -2, 1.5], [1] * 5, [1, 2])["adopt"] and not adopt_verdict([1, 2, 1, 2, 1.5], [-1] * 5, [1, 2])["adopt"]
            j = json.load(open(os.path.join(runs, "g_s42", "beam5_val_best.json"))); j["beam_result"]["metrics"][KEY] += 0.01
            json.dump(j, open(os.path.join(runs, "g_s42", "beam5_val_best.json"), "w"))
            try:
                load(runs, w9, gm, ARMS, {a: list(S1) for a in ARMS}); raise SystemExit("selftest: re-score gate did not fire")
            except AssertionError:
                pass
        finally:
            H.N_BIN = keep
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "report"))
    ap.add_argument("--runs"); ap.add_argument("--w9man"); ap.add_argument("--gman")
    ap.add_argument("--bins", default=str(HERE / "w12i_bins.json")); ap.add_argument("--arm_stats", default="")
    ap.add_argument("--readouts_dir", default=""); ap.add_argument("--marker", default=""); ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest()
    else:
        sys.exit(report(a.runs, a.w9man, a.gman, a.bins, a.arm_stats, a.readouts_dir, a.marker or os.path.join(a.runs, "STAGE1_PASS"), a.boot))


if __name__ == "__main__":
    main()
