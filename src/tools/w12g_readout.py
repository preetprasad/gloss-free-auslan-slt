#!/usr/bin/env python3
"""Wave-12g read-out: BLEU-4 on signers absent from training, and the same clips with the signer in training.
Pre-registration: results/PREREG_WAVE12G_SIGNERSPLIT_2026-10-13.md

Per seed it reads the beam-5 decodes of best.pt written by w4run (beam5_{val,test}_best.json) for the arms
SD, EXPa, EXPb, and the manifests written by tools/w12g_signer_split.py. Every decode is tied to its manifest
(same file name, references equal row for row, recorded BLEU-4 reproduced) before anything is scored.
  SD  = one model, scored on all 1,087 clips of the held-out signers
  EXP = EXPa's output on fold B + EXPb's output on fold A: the same 1,087 clips, each decoded by the model that
        had the other half of that signer's clips in training
Scores come from tools/arm_readouts.readouts (project metrics: detokenise, sacreBLEU 13a); intervals over seeds from
tools/seed_decision.interval. It prints tables and decides nothing; the readings are fixed in the pre-registration.

  python tools/w12g_readout.py readout --runs RUNS --manifests DIR [--seeds 42 101 202] [--boot 1000]
  python tools/w12g_readout.py validate --xlsx X --test_manifest T --dump NAME=hyps.jsonl:ID1,ID2,OTHERS ...
  python tools/w12g_readout.py selftest
"""
import argparse, collections, hashlib, json, os, random, statistics as st, sys, tempfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from slt.metrics import corpus_bleu4, detokenize          # noqa: E402
from arm_readouts import KEY, readouts                    # noqa: E402
from seed_decision import interval                        # noqa: E402
from stitch_lexicon import STOP                           # noqa: E402
import w12g_signer_split as sp                            # noqa: E402

# The stop list defines "content words". The file stitch_lexicon.py is edited by other experiments, so it is the list itself
# that is pinned (71 words, as on record in CLAUDE.md 2026-09-22k), not the file.
assert len(STOP) == 71 and hashlib.md5(" ".join(sorted(STOP)).encode()).hexdigest() == "a73c33b6e39e7145a87f9c639b5c0a10", \
    "stitch_lexicon.STOP is not the verified 71-word stop list"

ARMS = ("SD", "EXPa", "EXPb")


def load(runs, mandir, arm, seed, split):
    """{clip: hypothesis}, {clip: reference}, recorded BLEU-4 -- after tying the decode to its manifest."""
    name = f"w12g_{arm}_test.tsv" if split == "test" else "w12g_val.tsv"
    man = sp.read_manifest(os.path.join(mandir, name))
    path = os.path.join(runs, f"{arm}_s{seed}", f"beam5_{split}_best.json")
    j = json.load(open(path))
    refs = [r.split("\t", 1)[1] for r in man]
    assert os.path.basename(j["manifest"]) == name, f"{path}: decoded {j['manifest']}, expected {name}"
    assert j["refs"] == refs, f"{path}: references differ from {name} in content or order"
    got = corpus_bleu4([detokenize(h) for h in j["beam_hyps"]], [detokenize(r) for r in refs], "13a")
    assert abs(got - j["beam_result"]["metrics"][KEY]) < 1e-6, f"{path}: BLEU-4 {got} != recorded"
    clips = [sp.clip_of(r) for r in man]
    return dict(zip(clips, j["beam_hyps"])), dict(zip(clips, refs)), got


def subsets(mandir):
    """Fixed in the pre-registration. Each is a list of held-out clips in SD-test order."""
    side = sp.read_side(mandir)
    text = lambda n: [r.split("\t", 1)[1] for r in sp.read_manifest(os.path.join(mandir, f"w12g_{n}.tsv"))]
    T = {n: {sp.norm(t) for t in text(f"{n}_train")} for n in ARMS}
    test = sp.read_manifest(os.path.join(mandir, "w12g_SD_test.tsv"))
    clips, ref = [sp.clip_of(r) for r in test], {sp.clip_of(r): sp.norm(r.split("\t", 1)[1]) for r in test}
    exp_train = lambda c: T["EXPa" if side[c]["part"] == "B" else "EXPb"]      # the model that decodes c in EXP
    sub = collections.OrderedDict()
    sub["all held-out signers"] = clips
    sub["signer 10"] = [c for c in clips if side[c]["signer"] == "10"]
    sub["signer 8"] = [c for c in clips if side[c]["signer"] == "8"]
    sub["other 45 signers"] = [c for c in clips if side[c]["signer"] not in ("10", "8")]
    sub["target unseen (in neither arm's training text)"] = [c for c in clips if ref[c] not in T["SD"] and ref[c] not in exp_train(c)]
    sub["target seen in SD training text"] = [c for c in clips if ref[c] in T["SD"]]
    sub["signer exposed in the same video (EXP)"] = [c for c in clips if side[c]["exposed"] == "1"]
    sub["fold A (EXP = model EXPb)"] = [c for c in clips if side[c]["part"] == "A"]
    sub["fold B (EXP = model EXPa)"] = [c for c in clips if side[c]["part"] == "B"]
    return sub, side


def one_seed(runs, mandir, seed):
    sd, ref, _ = load(runs, mandir, "SD", seed, "test")
    a, ra, _ = load(runs, mandir, "EXPa", seed, "test")
    b, rb, _ = load(runs, mandir, "EXPb", seed, "test")
    assert not set(a) & set(b) and set(a) | set(b) == set(sd), "EXPa and EXPb tests do not partition the SD test"
    assert all(ref[c] == r for c, r in {**ra, **rb}.items()), "references differ between arms"
    val = {arm: load(runs, mandir, arm, seed, "val")[2] for arm in ARMS}
    return {"SD": sd, "EXP": {**a, **b}}, ref, val


def score(hyp, ref, clips):
    return readouts([hyp[c] for c in clips], [ref[c] for c in clips], STOP)


def pct(xs, q):
    xs = sorted(xs)
    return xs[round(q * (len(xs) - 1))]


def bootstrap(H, ref, clips, seeds, n_boot, groups=None):
    """Seed-mean BLEU-4 of SD and seed-mean EXP - SD on resampled clips (or resampled whole signers)."""
    D = {arm: {s: {c: detokenize(H[s][arm][c]) for c in clips} for s in seeds} for arm in ("SD", "EXP")}
    R = {c: detokenize(ref[c]) for c in clips}
    rng, units, out = random.Random(0), sorted(groups) if groups else None, []
    for _ in range(n_boot):
        pick = ([c for g in rng.choices(units, k=len(units)) for c in groups[g]] if groups
                else rng.choices(clips, k=len(clips)))
        r = [R[c] for c in pick]
        m = {arm: st.mean(corpus_bleu4([D[arm][s][c] for c in pick], r, "13a") for s in seeds) for arm in D}
        out.append((m["SD"], m["EXP"] - m["SD"]))
    return [(pct([o[i] for o in out], 0.025), pct([o[i] for o in out], 0.975)) for i in (0, 1)]


def floor(mandir, k=60):
    """Input-blind reference: one training sentence output for every clip. Among the k most frequent SD-training
    targets the one with the highest BLEU-4 on VALIDATION is taken; its scores on the held-out clips are returned."""
    text = lambda n: [r.split("\t", 1)[1] for r in sp.read_manifest(os.path.join(mandir, f"w12g_{n}.tsv"))]
    val, test = text("val"), text("SD_test")
    cands = [t for t, _ in collections.Counter(text("SD_train")).most_common(k)]
    best = max(cands, key=lambda t: readouts([t] * len(val), val, STOP)["bleu4"])
    return best, readouts([best] * len(val), val, STOP), readouts([best] * len(test), test, STOP)


def reading(d, ci, other45):
    """Pre-registered reading of the primary contrast (PREREG_WAVE12G section 5).
    d = per-seed EXP - SD BLEU-4 on all held-out clips; ci = clip-bootstrap 95% interval of its seed mean ("test
    sampling only"); other45 = the same per-seed differences on the 45 signers other than 10 and 8."""
    if len(d) < 3 or ci is None:
        return "NO READING: the rule needs three complete seeds and the bootstrap"
    m, (lo, hi), pos = st.mean(d), ci, sum(x > 0 for x in d)
    if m >= 1.0 and pos == len(d) and lo > 0:
        r = "R1: holding the signers out costs BLEU-4 (EXP - SD >= +1.0, 3/3 seeds positive, bootstrap interval above 0)"
    elif m <= -1.0 and pos == 0 and hi < 0:
        r = "R4: EXP scores BELOW SD (<= -1.0, 3/3 seeds negative, bootstrap interval below 0); reported as unexplained"
    elif abs(m) < 0.5 and lo > -1.5 and hi < 1.5:
        r = "R2: no resolvable cost (|EXP - SD| < 0.5 and the bootstrap interval inside -1.5 .. +1.5)"
    else:
        r = "R3: unresolved (the estimate and its intervals are reported, nothing is claimed)"
    if other45 is not None and len(other45):
        r += f" | scope: the 45 signers other than 10 and 8 give a positive difference in {sum(x > 0 for x in other45)}/{len(other45)} seeds"
    return r


def per_signer(H, ref, clips, side, seeds):
    """BLEU-4 per held-out signer (seed means), largest signer first. Signers with fewer than 20 clips are listed, not read."""
    by = collections.defaultdict(list)
    for c in clips:
        by[int(side[c]["signer"])].append(c)
    rows = []
    for sg, cs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        b = {arm: st.mean(score(H[s][arm], ref, cs)["bleu4"] for s in seeds) for arm in ("SD", "EXP")}
        rows.append(dict(signer=sg, n=len(cs), SD=b["SD"], EXP=b["EXP"], diff=b["EXP"] - b["SD"]))
    return rows


def readout(runs, mandir, seeds, n_boot):
    sub, side = subsets(mandir)
    H, ref, val = {}, None, {}
    for s in seeds:
        try:
            H[s], ref, val[s] = one_seed(runs, mandir, s)
        except FileNotFoundError as e:
            print(f"seed {s}: INCOMPLETE ({e.filename})")
    wanted, seeds = len(seeds), sorted(H)
    if not seeds:
        raise SystemExit("no complete seed")
    if len(seeds) < wanted:
        print(f"NOT ALL SEEDS COMPLETE ({len(seeds)} of {wanted}): the pre-registered readings need every seed.")
    print(f"seeds read: {seeds} | held-out clips {len(sub['all held-out signers'])} | BLEU-4, beam 5, best.pt "
          f"(chosen on greedy validation)")
    res = {}
    for name, clips in sub.items():
        if not clips:
            print(f"\n== {name} (n = 0) =="); continue
        r = {arm: [score(H[s][arm], ref, clips) for s in seeds] for arm in ("SD", "EXP")}
        d = [e["bleu4"] - x["bleu4"] for e, x in zip(r["EXP"], r["SD"])]
        (m_sd, lo_sd, hi_sd), (m_d, lo_d, hi_d) = interval([x["bleu4"] for x in r["SD"]]), interval(d)
        res[name] = dict(n=len(clips), SD=r["SD"], EXP=r["EXP"], diff=d)
        print(f"\n== {name} (n = {len(clips)}) ==")
        for arm in ("SD", "EXP"):
            print(f"  {arm:3s} per seed " + " ".join(f"{x['bleu4']:.3f}" for x in r[arm])
                  + f" | mean {st.mean(x['bleu4'] for x in r[arm]):.3f}")
        print(f"  SD mean {m_sd:.3f}, 95% t over seeds [{lo_sd:.3f}, {hi_sd:.3f}]")
        print(f"  EXP - SD per seed " + " ".join(f"{x:+.3f}" for x in d) + f" | positive {sum(x > 0 for x in d)}/{len(d)}"
              f" | mean {m_d:+.3f} | 95% t over seeds [{lo_d:+.3f}, {hi_d:+.3f}]")
        for k in ("wer", "len_ratio", "content_precision", "content_recall", "distinct"):
            dk = [e[k] - x[k] for e, x in zip(r["EXP"], r["SD"])]
            m, lo, hi = interval(dk)
            print(f"    {k:18s} SD {st.mean(x[k] for x in r['SD']):8.3f}   EXP {st.mean(x[k] for x in r['EXP']):8.3f}"
                  f"   EXP - SD {m:+.3f} [{lo:+.3f}, {hi:+.3f}], positive {sum(x > 0 for x in dk)}/{len(dk)}")
    const, fv, ft = floor(mandir)
    print(f"\n== input-blind reference: the constant output {const!r} (chosen on validation) ==\n"
          f"  held-out clips BLEU-4 {ft['bleu4']:.3f}, WER {ft['wer']:.3f} | validation BLEU-4 {fv['bleu4']:.3f}, WER {fv['wer']:.3f}")
    print("\n== validation: signers 1 and 2, six videos absent from training (selection set; descriptive) ==")
    for arm in ARMS:
        print(f"  {arm:5s} " + " ".join(f"{val[s][arm]:.3f}" for s in seeds) + f" | mean {st.mean(val[s][arm] for s in seeds):.3f}")
    clips = sub["all held-out signers"]
    for s in seeds:              # w4summary layout, so tools/seed_decision.py reads the primary contrast
        i = seeds.index(s)
        json.dump({"SD": dict(beam5_val=val[s]["SD"], beam5_test=res["all held-out signers"]["SD"][i]["bleu4"]),
                   "EXP": dict(beam5_val=(val[s]["EXPa"] + val[s]["EXPb"]) / 2,
                               beam5_test=res["all held-out signers"]["EXP"][i]["bleu4"])},
                  open(os.path.join(runs, f"summary_w12g_s{s}.json"), "w"), indent=1)
    rows = per_signer(H, ref, clips, side, seeds)
    res["per_signer"] = rows
    print("\n== per held-out signer, seed means (signers with fewer than 20 clips are listed, not interpreted) ==\n"
          "  signer    n     SD    EXP   EXP - SD")
    for r in rows:
        print(f"  {r['signer']:6d} {r['n']:4d} {r['SD']:6.2f} {r['EXP']:6.2f} {r['diff']:+8.2f}" + ("   (n < 20)" if r["n"] < 20 else ""))
    boot = None
    if n_boot:
        by_signer = collections.defaultdict(list)
        for c in clips:
            by_signer[int(side[c]["signer"])].append(c)
        for label, groups in (("clips resampled (test sampling only)", None),
                              (f"whole signers resampled ({len(by_signer)} signers)", by_signer)):
            (lo, hi), (dlo, dhi) = bootstrap(H, ref, clips, seeds, n_boot, groups)
            res[f"bootstrap: {label}"] = dict(SD=[lo, hi], EXP_minus_SD=[dlo, dhi])
            if groups is None:
                boot = (dlo, dhi)
            print(f"\nbootstrap, {n_boot} draws, {label}: seed-mean SD 95% [{lo:.3f}, {hi:.3f}] | "
                  f"seed-mean EXP - SD 95% [{dlo:+.3f}, {dhi:+.3f}]")
    verdict = reading(res["all held-out signers"]["diff"], boot, res.get("other 45 signers", {}).get("diff"))
    if len(seeds) < wanted:
        verdict = "NO READING: not all seeds are complete"
    print("\n== PRE-REGISTERED READING (PREREG_WAVE12G section 5) ==\n  " + verdict)
    res["reading"] = verdict
    json.dump(dict(seeds=seeds, validation={str(s): val[s] for s in seeds}, subsets=res),
              open(os.path.join(runs, "w12g_readout.json"), "w"), indent=1)
    return res


def validate(xlsx, test_manifest, dumps):
    """Per-signer BLEU-4 of saved official-test decodes, against numbers already on record (SIGNER_BREAKDOWN)."""
    info = sp.read_xlsx(xlsx)
    man = sp.read_manifest(test_manifest)
    clips, refs = [sp.clip_of(r) for r in man], [r.split("\t", 1)[1] for r in man]
    for spec in dumps:
        name, rest = spec.split("=", 1)
        path, want = rest.rsplit(":", 1)
        if path.endswith(".json"):                      # w4run layout: beam5_test_best.json, rows in manifest order
            j = json.load(open(path, encoding="utf-8"))
            rows = [dict(ref=r, hyp_beam=h) for r, h in zip(j["refs"], j["beam_hyps"])]
        else:
            rows = sorted((json.loads(l) for l in open(path, encoding="utf-8")), key=lambda r: r["idx"])
        assert [r["ref"] for r in rows] == refs, f"{name}: references differ from the test manifest"
        hyp, ref = dict(zip(clips, (r["hyp_beam"] for r in rows))), dict(zip(clips, refs))
        groups = [[c for c in clips if info[c]["signer"] == "1"], [c for c in clips if info[c]["signer"] == "2"],
                  [c for c in clips if info[c]["signer"] not in sp.PRINCIPAL]]
        got = [score(hyp, ref, g)["bleu4"] for g in groups]
        print(f"{name}: signer 1 {got[0]:.2f} (n {len(groups[0])}) | signer 2 {got[1]:.2f} (n {len(groups[1])}) | "
              f"other signers {got[2]:.2f} (n {len(groups[2])}) | on record {want}")
        if want != "-":                                  # "-" = no recorded value to check against, print only
            assert all(abs(g - float(w)) < 0.006 for g, w in zip(got, want.split(","))), f"{name}: does not reproduce {want}"
    print("validate OK: the per-signer scores that have a recorded value are reproduced")


def selftest():
    """Synthetic split and decodes with a known answer; the gates must reject a decode that does not fit."""
    rows, info = sp.synthetic()
    with tempfile.TemporaryDirectory() as d:
        sp.write(d, *sp.build(rows, info, val_min=50))
        man = {n: sp.read_manifest(os.path.join(d, f"w12g_{n}.tsv")) for n in sp.MANIFESTS}

        def dump(arm, seed, split, hyp_of):
            name = f"w12g_{arm}_test.tsv" if split == "test" else "w12g_val.tsv"
            m = man[f"{arm}_test"] if split == "test" else man["val"]
            refs = [r.split("\t", 1)[1] for r in m]
            hyps = [hyp_of(sp.clip_of(r), t) for r, t in zip(m, refs)]
            bleu = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs], "13a")
            os.makedirs(os.path.join(d, f"{arm}_s{seed}"), exist_ok=True)
            json.dump(dict(manifest=f"/group/somewhere/{name}", refs=refs, beam_hyps=hyps, beam_result=dict(metrics={KEY: bleu})),
                      open(os.path.join(d, f"{arm}_s{seed}", f"beam5_{split}_best.json"), "w"))

        wrong, right = (lambda c, t: "zzz yyy"), (lambda c, t: t)
        only10 = lambda c, t: t if info[c]["signer"] == "10" else "zzz yyy"
        for s in (1, 2):
            for arm, f in (("SD", wrong), ("EXPa", right), ("EXPb", only10)):
                dump(arm, s, "test", f)
                dump(arm, s, "val", right)
        res = readout(d, d, [1, 2, 3], n_boot=20)                    # seed 3 is absent: reported, not fatal
        r = lambda name, arm: res[name][arm][0]["bleu4"]
        assert res["all held-out signers"]["n"] == 26 and r("all held-out signers", "SD") == 0
        assert abs(r("signer 10", "EXP") - 100) < 1e-6, "signer 10 is right in both EXP models"
        assert abs(r("fold B (EXP = model EXPa)", "EXP") - 100) < 1e-6 and r("fold B (EXP = model EXPa)", "SD") == 0
        a_other = [c for c in map(sp.clip_of, man["EXPb_test"]) if info[c]["signer"] != "10"]
        assert a_other and 0 < r("fold A (EXP = model EXPb)", "EXP") < 100, "fold A must mix right and wrong outputs"
        assert all(x > 0 for x in res["all held-out signers"]["diff"])
        ps = res["per_signer"]
        assert sum(r["n"] for r in ps) == 26 and [r["n"] for r in ps] == sorted((r["n"] for r in ps), reverse=True)
        assert next(r for r in ps if r["signer"] == 10)["EXP"] > 99, "signer 10 is right in both EXP models"
        assert res["reading"].startswith("NO READING"), "seed 3 is missing: no reading may be given"
        s1 = json.load(open(os.path.join(d, "summary_w12g_s1.json")))
        assert s1["SD"]["beam5_test"] == 0 and s1["EXP"]["beam5_test"] == r("all held-out signers", "EXP")
        lo, hi = res["bootstrap: clips resampled (test sampling only)"]["EXP_minus_SD"]
        assert 0 < lo <= hi
        # gates: a decode of the wrong manifest, and a decode whose rows are in another order, are refused
        p = os.path.join(d, "EXPa_s1", "beam5_test_best.json")
        good = json.load(open(p))
        for gate, bad in (("expected w12g_EXPa_test.tsv", dict(good, manifest="/x/w12g_EXPb_test.tsv")),
                          ("references differ", dict(good, refs=good["refs"][::-1], beam_hyps=good["beam_hyps"][::-1])),
                          ("!= recorded", dict(good, beam_result=dict(metrics={KEY: 1.0})))):
            json.dump(bad, open(p, "w"))
            try:
                one_seed(d, d, 1)
            except AssertionError as e:
                assert gate in str(e), f"selftest FAILED: wrong gate fired for '{gate}': {e}"
            else:
                raise SystemExit(f"selftest FAILED: gate did not fire: {gate}")
        json.dump(good, open(p, "w"))
        one_seed(d, d, 1)
    R = lambda d3, ci, o=None: reading(d3, ci, o).split(":")[0]
    for what, args, want in (("clear cost", ([1.5, 2.0, 1.2], (0.4, 2.6)), "R1"),
                             ("mean big but one seed negative", ([2.5, 2.0, -0.1], (0.4, 2.6)), "R3"),
                             ("mean big but bootstrap includes 0", ([1.5, 2.0, 1.2], (-0.2, 3.0)), "R3"),
                             ("mean just below +1.0", ([1.2, 0.9, 0.8], (0.1, 1.9)), "R3"),
                             ("no cost", ([0.2, -0.3, 0.1], (-0.9, 1.0)), "R2"),
                             ("small mean, wide bootstrap", ([0.2, -0.3, 0.1], (-1.8, 2.0)), "R3"),
                             ("reverse", ([-1.5, -2.0, -1.2], (-2.6, -0.4)), "R4"),
                             ("two seeds only", ([1.5, 2.0], (0.4, 2.6)), "NO READING"),
                             ("no bootstrap", ([1.5, 2.0, 1.2], None), "NO READING")):
        assert R(*args) == want, f"selftest FAILED: reading of '{what}' is {R(*args)}, expected {want}"
    assert reading([1.5, 2.0, 1.2], (0.4, 2.6), [0.3, -0.1, 0.2]).endswith("positive difference in 2/3 seeds")
    print("selftest OK: pooled EXP scores, subsets, summary files and three gates behave as specified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["readout", "validate", "selftest"])
    ap.add_argument("--runs"); ap.add_argument("--manifests")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--xlsx"); ap.add_argument("--test_manifest"); ap.add_argument("--dump", action="append", default=[])
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "validate":
        return validate(a.xlsx, a.test_manifest, a.dump)
    readout(a.runs, a.manifests, a.seeds, a.boot)


if __name__ == "__main__":
    sys.exit(main())
