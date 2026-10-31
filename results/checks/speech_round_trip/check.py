"""Speech round trip (Whisper small.en): round-trip WER of voiced output vs human and scrambled references.

Rebuilds the four spoken arms (A model output, A6 final ensemble, B human references, C scrambled references),
the saved Whisper transcripts (B and C stored as edits against the spoken text), and re-runs the original
scoring functions of tools/speech_roundtrip.py (bootstrap: 2000 resamples, seed 0).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
import speech_roundtrip as S  # noqa: E402
from slt.metrics import corpus_wer  # noqa: E402

man = ROOT / "comm_run" / "manifests" / "test.tsv"
if not man.exists():
    sys.exit("needs the test references: run `make data` first")
M = {}
for l in open(man):
    if "\t" in l:
        a, b = l.rstrip("\n").split("\t", 1)
        M[Path(a).name[:-4]] = b
items = [json.loads(l) for l in open(HERE / "items.jsonl")]
n = len(items)
refs = [M[it["id"]] for it in items]
scr = [S.scramble(r, i) for i, r in enumerate(refs)]
spoken = {"A": [it["hyp_A"] for it in items], "A6": [it["hyp_A6"] for it in items], "B": refs, "C": [s for s, _ in scr]}
scrambled = [ok for _, ok in scr]
edits = json.load(open(HERE / "transcript_edits.json"))


def apply(sp, ops):
    return "".join(sp[o[1]:o[2]] if o[0] == "equal" else (o[3] if o[0] != "delete" else "") for o in ops)


T = {"A": [it["tr_A"] for it in items], "A6": [it["tr_A6"] for it in items]}
for a in ("B", "C"):
    T[a] = [apply(spoken[a][i], edits[a][i]) for i in range(n)]
rec = json.load(open(HERE / "recorded_results.json"))
meta, tj = rec["meta"], json.load(open(HERE / "transcribe_metal.json"))
print(f"Transcriber: whisper.cpp {tj['whisper']}, model {tj['model']} (sha256 {tj['model_sha256'][:12]}), backend {tj['backend']}  [c07-072]")
assert tj["model"] == "ggml-small.en.bin"

# GATE: strict all-800 round-trip WER of the human references equals the value the original job recorded
g = corpus_wer([S.n0(T["B"][i]) for i in range(n)], [S.n0(spoken["B"][i]) for i in range(n)])
print(f"GATE human references strict RT-WER {g:.6f} (recorded {rec['N0/all800']['RT-WER_B'][0]:.6f})")
if abs(g - rec["N0/all800"]["RT-WER_B"][0]) > 1e-12 or sum(scrambled) != 757:
    sys.exit("GATE FAILED")

idx = list(range(n))
import numpy as np, jiwer  # noqa: E402
P = lambda v: f"{100*v[0]:.1f}% [{100*v[1]:.1f}, {100*v[2]:.1f}]"
NORMS = {"strict": S.n0, "normalised": S.n1}


def per_item(h, r, norm):
    """edit errors and reference words per clip; corpus WER = sum(err)/sum(words), as jiwer on the whole set."""
    e, w = [], []
    for x, y in zip(h, r):
        x, y = norm(x), norm(y)
        o = jiwer.process_words([y or "<empty>"], [x or "<empty>"])
        e.append(o.substitutions + o.deletions + o.insertions); w.append(o.hits + o.substitutions + o.deletions)
    return np.array(e, float), np.array(w, float)


E_ = {(t, a): per_item(T[a], spoken[a], nm) for t, nm in NORMS.items() for a in S.ARMS}
assert abs(E_["strict", "B"][0].sum() / E_["strict", "B"][1].sum() - g) < 1e-12
wer = lambda t, a, s: E_[t, a][0][s].sum() / E_[t, a][1][s].sum()
R = {}
for t in NORMS:
    for a in S.ARMS:
        R[t, a] = S.boot_ci(lambda s, t=t, a=a: wer(t, a, np.array(s)), n)
print("\nRound-trip WER, 800 official test clips, 95% bootstrap interval (2000 resamples, seed 0)")
lab = {"A": "Output of the 40-epoch pre-trained model (voiced)", "A6": "Final ensemble output",
       "B": "Human reference sentences", "C": "References with scrambled word order"}
ids = {("strict", "A"): "c20-012 c20-016 c14-095 c20-015", ("strict", "A6"): "c20-012 c20-017 c14-095 c20-015",
       ("strict", "B"): "c20-012 c20-018 c14-095 c20-013 c20-015", ("strict", "C"): "c20-013 c20-019 c20-015",
       ("normalised", "A"): "c20-016", ("normalised", "A6"): "c20-017", ("normalised", "B"): "c20-018",
       ("normalised", "C"): "c20-019"}
for a in ("A", "A6", "B", "C"):
    print(f"{lab[a]:52s} strict {P(R['strict', a])}  normalised {P(R['normalised', a])}  [{' '.join(sorted(set((ids['strict', a] + ' ' + ids['normalised', a]).split())))}]")
print(f"scrambling the references raised strict RT-WER from {100*R['strict','B'][0]:.1f}% to {100*R['strict','C'][0]:.1f}%  [c20-013]")
print(f"ensemble strict upper bound unrounded {100*R['strict','A6'][2]:.3f}% (rounds to 4.4, not 4.5)  [c20-017]")

# paired difference, B standardised to A's target-word mass per stratum (tool conventions, speech_roundtrip.py)
st = {a: np.array([S.stratum(spoken[a][i]) for i in range(n)]) for a in S.ARMS}
NW = {(t, a): np.array([len(nm(spoken[a][i]).split()) for i in range(n)]) for t, nm in NORMS.items() for a in S.ARMS}


def dstd(x, t, sel_idx, strata):
    def f(s):
        sel = np.array([sel_idx[j] for j in s])
        w = {k: NW[t, x][sel][st[x][sel] == k].sum() for k in set(st[x][sel])}
        tot = sum(w.values()); w = {k: v / tot for k, v in w.items()}
        if strata:
            w = {k: v for k, v in w.items() if k in strata}
            z = sum(w.values()); w = {k: v / z for k, v in w.items()}
        r = 0.0
        for arm, sign in ((x, 1), ("B", -1)):
            for k, wk in w.items():
                sub = sel[st[arm][sel] == k]
                assert len(sub), "empty stratum"
                r += sign * wk * wer(t, arm, sub)
        return r
    return f


sub757 = [i for i in range(n) if scrambled[i]]
D = {}
print("\nPaired difference, voiced output minus human references (B standardised to the output's length mix), percentage points")
for tag in NORMS:
    for x in ("A", "A6"):
        for sub, si, stt in (("all 800", idx, None), ("757 scramblable", sub757, {"2", "3-5", "6+"})):
            D[tag, x, sub] = S.boot_ci(dstd(x, tag, si, stt), len(si))
            v = D[tag, x, sub]
            tagid = " c18-054" if (tag, x, sub) == ("strict", "A", "all 800") else ""
            print(f"{tag:10s} {x:3s} {sub:16s} {100*v[0]:+.1f} ({100*v[1]:+.1f} to {100*v[2]:+.1f})  [c20-014{tagid}]")
v = D["strict", "A", "all 800"]
print(f"Voiced 40-epoch output, strict, all 800: {100*v[0]:+.1f} percentage points ({100*v[1]:+.1f} to {100*v[2]:+.1f})  [c18-054]")
mxA = max(100 * v[2] for k, v in D.items() if k[1] == "A")
mxA6 = max(100 * v[2] for k, v in D.items() if k[1] == "A6")
print(f"largest upper bound, 40-epoch output (A), all variants: {mxA:+.1f} points, i.e. at most half a point  [c20-014]")
print(f"largest upper bound, final ensemble (A6), all variants: {mxA6:+.1f} points (strict, 757 clips)  [c20-014]")
print("DISAGREES for the ensemble: the appendix says 'every variant ... at most half a percentage point'; it holds for the 40-epoch output only")
print(f"setting: {n} official test clips, 95% bootstrap intervals  [c20-015]")
