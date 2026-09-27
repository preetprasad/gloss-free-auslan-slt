"""Does fixing person switches pay off where switches are dense? Exploratory dose-response on wave 3 (job 6592/9444).

w3C = trained and decoded on index-0 pose; w3T = trained and decoded on person-tracked pose; seeds 42/101/202/303/404,
in-batch pairs. Clips are binned by frac_chosen_nonzero from results/ndet_per_clip.csv (fraction of frames where the
tracker overrode detection 0). On unaffected clips (frac 0) both arms see the same input, so their delta is a pure
training effect; the DiD (heavy - unaffected) isolates the effect of the corrected INPUT on the clips that need it.

RULE, written before any binned number was computed (2026-09-25): invest in a better tracker only if the heavy-bin
(frac > 0.3) DiD is >= +1.0 BLEU-4 on BOTH validation and test, with >= 4/5 seeds positive. Otherwise the switch defect
is not where BLEU is. Exploratory: the official test has been inspected many times; this decides engineering effort,
not a claim.
"""
import csv, json, sys, statistics as st
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize
import sacrebleu
D = ROOT / "results/switch_video_89_193/w3_val"
SEEDS = [42, 101, 202, 303, 404]; T975 = {4: 2.776}; KEY = "tokenize=13a;detok=True"
FRAC = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(open(ROOT / "results/ndet_per_clip.csv"))}
BINS = {"none (0)": lambda f: f == 0, "light (0,0.1]": lambda f: 0 < f <= 0.1,
        "medium (0.1,0.3]": lambda f: 0.1 < f <= 0.3, "heavy (>0.3)": lambda f: f > 0.3}
def clips(p): return [l.split("\t")[0].rsplit("/", 1)[1][:-4] for l in open(p) if "\t" in l]
def bleu(h, r): return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")
def chrf(h, r): return st.mean(sacrebleu.sentence_chrf(detokenize(a), [detokenize(b)]).score for a, b in zip(h, r))
def ci(d): m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5; return m, m - hw, m + hw
for split, fn, off, trk in (("validation", "{a}_s{s}.json", "real_data_learning_curve/val.tsv", "val_tracked.tsv"),
                            ("test", "{a}_s{s}_test.json", "real_data_learning_curve/test.tsv", "test_tracked.tsv")):
    order = clips(ROOT / "results" / off); assert order == clips(D / trk), "tracked manifest order differs"
    frac = [FRAC[c] for c in order]
    print(f"\n==== {split} (n per bin: " + ", ".join(f"{b} {sum(map(f, frac))}" for b, f in BINS.items()) + ")")
    per = {b: {"T": [], "C": [], "chT": [], "chC": []} for b in BINS}
    for s in SEEDS:
        R = {}
        for a in ("w3C", "w3T"):
            d = json.load(open(D / fn.format(a=a, s=s)))
            h, r = d["beam_hyps"], d["refs"]; assert len(h) == 800
            assert abs(bleu(h, r) - d["beam_result"]["metrics"][KEY]) < 1e-6, f"gate: {a} s{s} does not re-score"
            R[a] = (h, r)
        assert R["w3C"][1] == R["w3T"][1]
        for b, f in BINS.items():
            idx = [i for i, x in enumerate(frac) if f(x)]
            for a, k in (("w3T", "T"), ("w3C", "C")):
                h, r = R[a]; per[b][k].append(bleu([h[i] for i in idx], [r[i] for i in idx]))
                per[b]["ch" + k].append(chrf([h[i] for i in idx], [r[i] for i in idx]))
    base = [t - c for t, c in zip(per["none (0)"]["T"], per["none (0)"]["C"])]
    for b in BINS:
        dlt = [t - c for t, c in zip(per[b]["T"], per[b]["C"])]
        dch = [t - c for t, c in zip(per[b]["chT"], per[b]["chC"])]
        m, lo, hi = ci(dlt); cm, clo, chi = ci(dch)
        line = (f"{b:17s} BLEU T {st.mean(per[b]['T']):6.2f} C {st.mean(per[b]['C']):6.2f} | T-C {' '.join(f'{x:+.2f}' for x in dlt)}"
                f" mean {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | chrF T-C {cm:+.2f} [{clo:+.2f}, {chi:+.2f}]")
        if b != "none (0)":
            did = [x - y for x, y in zip(dlt, base)]; dm, dlo, dhi = ci(did)
            line += f" | DiD vs none {dm:+.2f} [{dlo:+.2f}, {dhi:+.2f}] ({sum(x > 0 for x in did)}/5)"
        print(line)
print("\nGATE OK: every dump re-scored to its recorded BLEU; tracked and official manifests share clip order.")
