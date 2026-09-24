"""t3-speech: does the demonstrator's voice deliver what the translator wrote? (synthesis -> recognition round trip)

Pre-registration (binding, amendments A1-A11 win): results/PREREG_T3_SPEECH_2026-09-22.md.
Stages, run in order:  build | synth | transcribe | g5sheet (a human labels it) | score.   `python tools/speech_roundtrip.py selftest` runs G3.
"""
import argparse, hashlib, json, os, random, re, subprocess, sys, wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, corpus_wer, detokenize  # noqa: E402
from slt.speech import Speaker, _duration_seconds  # noqa: E402

RES = ROOT / "results"
OUT = RES / "speech_roundtrip"
ARMS = ("A", "A6", "B", "C")
FINGERSPELLED = {621, 771, 772, 773}          # A5: reported with and without
WHISPER_FLAGS = ["-l", "en", "-tp", "0.0", "-nf", "-bs", "5", "-bo", "5", "-mc", "0", "-nt", "-np", "-otxt",
                 "-nth", "0.60", "-lpt", "-1.00", "-et", "2.40", "-ac", "0", "-fa", "-t", "8"]   # A6: every threshold pinned


# ---------------------------------------------------------------- normalisation (§6, A1)
def n0(s):
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s.lower()).replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9]", " ", s).split())


_CONTR = [("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("'m", " am"), ("'re", " are"),
          ("'ll", " will"), ("'ve", " have"), ("'d", " would")]
_NUM = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen twenty".split()
_TENS = {30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_TOK = {"cannot": "can not", "okay": "ok", "alright": "all right"}


def n1(s):
    s = s.lower().replace("’", "'")
    for a, b in _CONTR:
        s = s.replace(a, b)
    out = []
    for t in n0(s).split():
        if t.isdigit() and int(t) <= 20:
            t = _NUM[int(t)]
        elif t.isdigit() and int(t) in _TENS:
            t = _TENS[int(t)]
        out.append(_TOK.get(t, t))
    return " ".join(" ".join(out).split())


def F(s, norm=n0):
    """BLEU scoring form: normalised text plus the sentence-final period every text line carries (§6, A4)."""
    x = norm(s)
    return x + " ." if x else ""


def scramble(ref, idx):
    toks = ref.split()
    assert toks[-1] == ".", ref
    body = toks[:-1]
    if len(set(body)) < 2:
        return ref, False
    rng, new = random.Random(f"t3-speech:{idx}"), list(body)
    for _ in range(1000):
        rng.shuffle(new)
        if new != body:
            return " ".join(new + ["."]), True
    raise RuntimeError(f"could not scramble idx {idx}")


def stratum(text):
    n = len(n0(text).split())
    return "1" if n <= 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"


# ---------------------------------------------------------------- build (G1)
def build():
    bench = json.load(open(RES / "speech_pipeline_bench.json"))["records"]
    man = [l.rstrip("\n").split("\t") for l in open(RES / "TEST_manifest_official.tsv")]
    s6 = sorted((json.loads(l) for l in open(RES / "final_ensemble_test_decodes" / "hyps_LP=0.3.jsonl")), key=lambda r: r["idx"])
    assert len(bench) == len(man) == len(s6) == 800
    assert [Path(p).stem for p, _ in man] == [r["clip"] for r in bench], "G1: bench clip order != manifest"
    refs = [r["ref"] for r in bench]
    assert refs == [t for _, t in man] == [r["ref"] for r in s6], "G1: references differ"
    arms = {"A": [r["hyp"] for r in bench], "A6": [r["hyp_beam"] for r in s6], "B": refs, "C": [], }
    flags = []
    for i, r in enumerate(refs):
        c, ok = scramble(r, i)
        arms["C"].append(c)
        flags.append(ok)
    for arm, xs in arms.items():
        assert all(F(x) == x for x in xs + refs if x), f"G1: F(x) != x in arm {arm}"
    for arm, want in (("A", 10.2691), ("A6", 11.8017)):
        got = corpus_bleu4([detokenize(h) for h in arms[arm]], [detokenize(r) for r in refs])
        assert abs(got - want) < 0.0015, f"G1: {arm} BLEU {got} != {want}"
        print(f"G1 {arm} BLEU-4 {got:.4f} (recorded {want})")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"refs": refs, "arms": arms, "scrambled": flags, "audio_s": [r["audio_s"] for r in bench],
               "clips": [r["clip"] for r in bench]}, open(OUT / "arms.json", "w"))
    print(f"G1 PASSED | C scrambled {sum(flags)} unchanged {800 - sum(flags)}")


def load_arms():
    return json.load(open(OUT / "arms.json"))


# ---------------------------------------------------------------- synth (G2)
def _synth_one(job):
    arm, i, text = job
    aiff, wav = OUT / arm / f"{i:03d}.aiff", OUT / arm / f"{i:03d}.wav"
    if not wav.exists():
        Speaker().synthesise(text, out_path=str(aiff))           # voice Karen, rate 180, exactly as the chain
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)], check=True)
    return arm, i, _duration_seconds(str(aiff)), _duration_seconds(str(wav))


def synth(workers):
    d = load_arms()
    voice = [l for l in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
             if l.startswith("Karen ") and "en_AU" in l]
    assert voice, "G0/G2: Karen en_AU voice missing"
    for a in ARMS:
        (OUT / a).mkdir(exist_ok=True)
    jobs = [(a, i, t) for a in ARMS for i, t in enumerate(d["arms"][a])]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_synth_one, jobs))
    bad = [(a, i, da, d["audio_s"][i]) for a, i, da, _ in res if a == "A" and abs(da - d["audio_s"][i]) > 1e-6]
    assert not bad, f"G2 FAILED (voice changed?): {bad[:5]}"
    drift = [(a, i, da, dw) for a, i, da, dw in res if abs(da - dw) > 1e-3]
    assert not drift, f"G2 FAILED: wav/aiff duration differ {drift[:5]}"
    for a in ARMS:
        for i in range(800):
            with wave.open(str(OUT / a / f"{i:03d}.wav")) as w:
                assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), (a, i)
    json.dump({"voice": voice[0].strip(), "rate": 180, "utterances": len(res), "armA_duration_match": 800,
               "wav_format": "16000 Hz mono Int16, all 3200"}, open(OUT / "g2_pass.json", "w"), indent=1)
    print(f"G2 PASSED | {len(res)} utterances | arm-A durations equal the recorded chain for 800/800 | {voice[0].strip()}")


# ---------------------------------------------------------------- transcribe (G0, G4)
def _whisper(model, wavs, backend):
    for w in wavs:                                     # whisper-cli exits 0 and writes nothing for an unreadable
        Path(f"{w}.txt").unlink(missing_ok=True)      # input, so a stale .txt must never survive a re-run
    cmd = ["whisper-cli", "-m", model, *WHISPER_FLAGS] + (["-ng"] if backend == "cpu" else []) + [str(w) for w in wavs]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe(model, backend):
    size = os.path.getsize(model)
    assert size == 487_614_201, f"G0: model size {size}"
    sha = hashlib.sha256(open(model, "rb").read()).hexdigest()
    v = subprocess.run(["whisper-cli", "--version"], capture_output=True, text=True)
    assert "1.9.4" in v.stdout + v.stderr, "G0: whisper-cli version"
    jfk = OUT / "g0" / "jfk.wav"
    jfk.parent.mkdir(exist_ok=True)
    jfk.write_bytes(Path("/opt/homebrew/share/whisper.cpp/jfk.wav").read_bytes())
    shortest = min(Path(OUT / "B").glob("*.wav"), key=lambda p: _duration_seconds(str(p)))
    short_copy = OUT / "g0" / "shortest.wav"
    short_copy.write_bytes(shortest.read_bytes())
    _whisper(model, [jfk, short_copy], backend)
    assert "ask not what your country can do for you" in n0((OUT / "g0" / "jfk.wav.txt").read_text()), "G0: JFK"
    assert n0((OUT / "g0" / "shortest.wav.txt").read_text()), f"G0 (A6): shortest clip {shortest.name} is empty"
    for a in ARMS:
        _whisper(model, sorted((OUT / a).glob("*.wav")), backend)
        assert len(list((OUT / a).glob("*.wav.txt"))) == 800, f"G0: missing transcripts in {a}"
    # G4a on symlinked copies, reversed order, separate directory (A7)
    rep = OUT / "g4a"
    rep.mkdir(exist_ok=True)
    links = []
    for w in sorted((OUT / "B").glob("*.wav"), reverse=True):
        l = rep / w.name
        if not l.exists():
            l.symlink_to(w)
        links.append(l)
    _whisper(model, links, backend)
    diff = [w.name for w in links if (rep / f"{w.name}.txt").read_bytes() != (OUT / "B" / f"{w.name}.txt").read_bytes()]
    # G4b: identical spoken text -> identical wav and transcript
    d, seen, g4b = load_arms(), {}, []
    for a in ARMS:
        for i, t in enumerate(d["arms"][a]):
            w = OUT / a / f"{i:03d}.wav"
            key = (hashlib.sha1(w.read_bytes()).hexdigest(), (OUT / a / f"{i:03d}.wav.txt").read_bytes())
            if t in seen and seen[t] != key:
                g4b.append((a, i, t))
            seen.setdefault(t, key)
    json.dump({"whisper": "1.9.4", "model": "ggml-small.en.bin", "model_sha256": sha, "backend": backend,
               "flags": WHISPER_FLAGS, "g4a_differ": diff, "g4b_differ": g4b[:50]},
              open(OUT / f"transcribe_{backend}.json", "w"), indent=1)
    assert not diff and not g4b, f"G4 FAILED ({len(diff)} reorder, {len(g4b)} duplicate): re-run with --backend cpu"
    print(f"G0+G4 PASSED ({backend}) sha256 {sha[:12]}")


def transcripts(arm):
    return [(OUT / arm / f"{i:03d}.wav.txt").read_text().strip() for i in range(800)]


# ---------------------------------------------------------------- score
def boot_ci(stat, n, B=2000, seed=0):
    import numpy as np
    rng, full = np.random.default_rng(seed), stat(list(range(n)))
    draws = sorted(stat(list(rng.integers(0, n, n))) for _ in range(B))
    return full, draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


def rt_wer(hyp, tgt, idx, norm):
    return corpus_wer([norm(hyp[i]) for i in idx], [norm(tgt[i]) for i in idx])


def rt_wer_std(hyp, tgt, idx, norm, weights, strata_of):
    """Direct standardisation (A2): stratum rates reweighted to arm A's target-word mass per stratum."""
    tot = 0.0
    for s, w in weights.items():
        sub = [i for i in idx if strata_of[i] == s]
        if w:
            assert sub, f"stratum {s} has weight {w:.4f} but no clips in the arm being standardised"
            tot += w * rt_wer(hyp, tgt, sub, norm)
    return tot


def g5_order():
    return [i for i in random.Random(0).sample(range(800), 40) if i not in FINGERSPELLED]


def g5sheet():
    """The labelling sheet for G5 (A5): idx, file, raw transcript, reference, for a human to read and label."""
    d, rows = load_arms(), []
    for i in g5_order():
        f = OUT / "B" / f"{i:03d}.wav.txt"
        assert int(f.name[:3]) == i
        rows.append({"idx": i, "file": f"B/{f.name}", "transcript": f.read_text().strip(), "reference": d["arms"]["B"][i],
                     "label": ""})
    json.dump(rows, open(OUT / "g5_sheet.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows to {OUT / 'g5_sheet.json'}; label each match / mismatch / listen")


def subs_pairs(hyp, tgt, idx, norm, k=20):
    import jiwer
    H = [norm(hyp[i]) or "<empty>" for i in idx]
    R = [norm(tgt[i]) or "<empty>" for i in idx]
    out, c = jiwer.process_words(R, H), Counter()
    for a, r, h in zip(out.alignments, R, H):
        rw, hw = r.split(), h.split()
        for ch in a:
            if ch.type == "substitute":
                for j in range(ch.ref_end_idx - ch.ref_start_idx):
                    c[(rw[ch.ref_start_idx + j], hw[ch.hyp_start_idx + j])] += 1
    return c.most_common(k)


def score(labels_path, backend):
    d, T = load_arms(), {a: transcripts(a) for a in ARMS}
    spoken, n = d["arms"], 800
    g2 = json.load(open(OUT / "g2_pass.json"))                   # written only when G2 passed
    tj = json.load(open(OUT / f"transcribe_{backend}.json"))
    assert not tj["g4a_differ"] and not tj["g4b_differ"] and tj["model_sha256"], "G4 did not pass for these transcripts"
    # G5: hand-verified harness gate (A5)
    lab = json.load(open(labels_path))           # {"idx": "match"|"mismatch"|"listen"}
    order = g5_order()
    bad = [i for i in order if lab.get(str(i)) not in ("match", "mismatch", "listen")]
    assert not bad, f"G5: items drawn but not labelled (or mislabelled): {bad}"
    print("G5 labels:", Counter(lab[str(i)] for i in order))
    match = [i for i in order if lab.get(str(i)) == "match"][:20]
    assert len(match) == 20, f"G5 FAILED: only {len(match)} matches in the first 40 draws"
    assert rt_wer(T["B"], spoken["B"], match, n0) == 0.0, "G5 FAILED: a hand-verified match scores > 0 -- fix the code"
    for i in order:
        if lab.get(str(i)) == "mismatch":
            assert rt_wer(T["B"], spoken["B"], [i], n0) > 0, f"G5 FAILED: mismatch {i} scores 0"
    shifted = T["B"][1:] + T["B"][:1]
    assert corpus_wer([n0(x) for x in shifted], [n0(x) for x in spoken["B"]]) > 0.8, "G5: alignment control"
    print("G5 PASSED")

    out = {"meta": {"whisper": tj["whisper"], "model": tj["model"], "model_sha256": tj["model_sha256"],
                    "backend": tj["backend"], "flags": tj["flags"], "voice": g2["voice"]}}
    for tag, norm in (("N0", n0), ("N1", n1)):
        for sub_tag, idx in (("all800", list(range(n))), ("no_fingerspelled", [i for i in range(n) if i not in FINGERSPELLED]),
                             ("scramblable757", [i for i in range(n) if d["scrambled"][i]])):
            r = {"empty": {a: sum(1 for i in idx if not norm(T[a][i])) for a in ARMS},
                 "empty_by_stratum": {a: dict(Counter(stratum(spoken[a][i]) for i in idx if not norm(T[a][i])))
                                      for a in ARMS}}
            for a in ARMS:
                r[f"RT-WER_{a}"] = boot_ci(lambda s, a=a: rt_wer(T[a], spoken[a], [idx[j] for j in s], norm), len(idx))
                assert r[f"RT-WER_{a}"][0] == corpus_wer([norm(T[a][i]) for i in idx], [norm(spoken[a][i]) for i in idx])
            two_plus = {"2", "3-5", "6+"}
            std_strata = two_plus if sub_tag == "scramblable757" else None     # B has no 1-word clips there
            for x in ("A", "A6"):
                r[f"d{x}"] = boot_ci(lambda s, x=x: rt_wer(T[x], spoken[x], [idx[j] for j in s], norm)
                                     - rt_wer(T["B"], spoken["B"], [idx[j] for j in s], norm), len(idx))
                r[f"d{x}-dC"] = boot_ci(lambda s, x=x: rt_wer(T[x], spoken[x], [idx[j] for j in s], norm)
                                        - rt_wer(T["C"], spoken["C"], [idx[j] for j in s], norm), len(idx))
                # A2: standardise B and C to arm x's target-word mass per stratum, inside every resample
                st = {a: [stratum(spoken[a][i]) for i in range(n)] for a in ARMS}

                def wts(sel, x=x):
                    m = {}
                    for i in sel:
                        m[st[x][i]] = m.get(st[x][i], 0) + len(norm(spoken[x][i]).split())
                    tot = sum(m.values())
                    return {k: v / tot for k, v in m.items()}

                def dstd(s, ref_arm, x=x, strata=None):
                    sel = [idx[j] for j in s]
                    w = wts(sel)
                    if strata:
                        w = {k: v for k, v in w.items() if k in strata}
                        z = sum(w.values())
                        w = {k: v / z for k, v in w.items()}
                    ax = rt_wer_std(T[x], spoken[x], sel, norm, w, st[x])
                    return ax - rt_wer_std(T[ref_arm], spoken[ref_arm], sel, norm, w, st[ref_arm])
                r[f"d{x}_std" + ("(2+)" if std_strata else "")] = boot_ci(lambda s: dstd(s, "B", strata=std_strata), len(idx))
                r[f"d{x}-dC_std(2+)"] = boot_ci(lambda s: dstd(s, "C", strata=two_plus), len(idx))

                def dc_std(s, x=x):                       # A2: dC standardised to arm x's 2+ mass
                    sel = [idx[j] for j in s]
                    w = {k: v for k, v in wts(sel).items() if k in two_plus}
                    z = sum(w.values())
                    w = {k: v / z for k, v in w.items()}
                    return (rt_wer_std(T["C"], spoken["C"], sel, norm, w, st["C"])
                            - rt_wer_std(T["B"], spoken["B"], sel, norm, w, st["B"]))
                r[f"dC_std(2+)|w={x}"] = boot_ci(dc_std, len(idx))
            r["dC"] = boot_ci(lambda s: rt_wer(T["C"], spoken["C"], [idx[j] for j in s], norm)
                              - rt_wer(T["B"], spoken["B"], [idx[j] for j in s], norm), len(idx))
            r["C_vs_original_refs"] = rt_wer(T["C"], spoken["B"], idx, norm)       # A8 repair diagnostic
            r["subs_top20"] = {a: subs_pairs(T[a], spoken[a], idx, norm) for a in ("B", "A", "A6")}
            r["strata"] = {a: {s: rt_wer(T[a], spoken[a], [i for i in idx if stratum(spoken[a][i]) == s], norm)
                               for s in ("1", "2", "3-5", "6+") if any(stratum(spoken[a][i]) == s for i in idx)}
                           for a in ARMS}
            out[f"{tag}/{sub_tag}"] = r
            print(tag, sub_tag, {k: (round(v[0], 4), round(v[1], 4), round(v[2], 4)) if isinstance(v, tuple) else None
                                 for k, v in r.items() if isinstance(v, tuple)})
    # Q2 heard-BLEU dumps for tools/paired_bootstrap.py, both scoring forms (A4)
    for tag, norm in (("F0", n0), ("F1", n1)):
        for name, arm in (("VLP", "A"), ("S6", "A6")):
            for kind, xs in (("TXT", spoken[arm]), ("HEARD", T[arm])):
                with open(OUT / f"{kind}_{name}_{tag}.jsonl", "w") as f:
                    for i in range(n):
                        f.write(json.dumps({"idx": i, "ref": F(d["refs"][i], norm), "hyp_beam": F(xs[i], norm)}) + "\n")
    json.dump(out, open(OUT / "roundtrip_results.json", "w"), indent=1)


def selftest():
    assert F("I am possum.") == F("i am possum .") == "i am possum ."
    assert F("Well, never mind.") == "well never mind ."
    assert n0("[BLANK_AUDIO]") == "" and F("[BLANK_AUDIO]") == ""
    for s in ("It's OK, isn't it?", "i am possum .", "  (music) hello  "):
        assert n0(n0(s)) == n0(s) and n1(n1(s)) == n1(s)
    assert n1("I can't, okay? 2 cats, alright") == "i can not ok two cats all right"
    assert n1("cannot") == "can not" and n0("can't") == "cant"
    assert scramble("a b c .", 5) == scramble("a b c .", 5) and scramble("a b c .", 5)[0] != "a b c ."
    assert scramble("hello .", 1) == ("hello .", False) and scramble("no no .", 1) == ("no no .", False)
    assert scramble("a b .", 3) == ("b a .", True)
    assert stratum("hello .") == "1" and stratum("a b c .") == "3-5" and stratum("a b c d e f .") == "6+"
    # score arithmetic: perfect transcripts give 0, deleting everything gives 1
    tg = ["a b .", "c ."]
    assert rt_wer(["A B.", "c"], tg, [0, 1], n0) == 0.0 and rt_wer(["", ""], tg, [0, 1], n0) == 1.0
    w = {"1": 0.5, "2": 0.5}
    assert rt_wer_std(["a b", "x"], tg, [0, 1], n0, w, ["2", "1"]) == 0.5
    if (OUT / "arms.json").exists():
        d = load_arms()
        for i, (b, c, ok) in enumerate(zip(d["arms"]["B"], d["arms"]["C"], d["scrambled"])):
            assert (c != b) == ok and sorted(c.split()) == sorted(b.split()), i
            assert (c, ok) == scramble(b, i)
    print("G3 selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("selftest", "build", "synth", "transcribe", "g5sheet", "score"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=str(Path.home() / "models/whisper/ggml-small.en.bin"))
    ap.add_argument("--backend", choices=("metal", "cpu"), default="metal")
    ap.add_argument("--labels", default=str(OUT / "g5_labels.json"))
    a = ap.parse_args()
    {"selftest": selftest, "build": build, "synth": lambda: synth(a.workers),
     "transcribe": lambda: transcribe(a.model, a.backend), "g5sheet": g5sheet,
     "score": lambda: score(a.labels, a.backend)}[a.stage]()
