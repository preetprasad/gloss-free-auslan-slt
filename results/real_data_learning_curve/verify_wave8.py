#!/usr/bin/env python3
"""Independent adversarial re-computation of the wave-8 (job 21734) readout.

Does NOT import tools/learning_curve.py or the readout agent's scripts. Reads the raw eval_beam JSON dumps,
progress.json files and manifests directly; every BLEU goes through slt/metrics.py (corpus_bleu4 on detokenize()d
text, 13a). Subset integrity is checked by re-running the pinned builder's sampling functions (tools/real_subsets.py,
md5 c338e5f9) on the pinned train manifest and comparing to the local manifest copies row for row.

  cd src && ~/.pyenv/versions/3.14.3/bin/python ../results/real_data_learning_curve/verify_wave8.py
"""
import json, math, os, re, sys, statistics as st
from collections import Counter
import numpy as np

W8 = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(W8, "..", "..", "src")
sys.path.insert(0, PB); sys.path.insert(0, os.path.join(PB, "tools"))
from slt.metrics import corpus_bleu4, corpus_wer, detokenize   # noqa: E402
import real_subsets as RS                                        # noqa: E402  (pinned builder, used only to regenerate)
from stitch_lexicon import STOP, content_words                  # noqa: E402
from sacrebleu.metrics import BLEU                               # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
ARMS = ["R025", "R050", "R100", "V050"]
N = {"R025": 3141, "R050": 6241, "R100": 12439}
T975_DF4 = 2.776
KEY = "tokenize=13a;detok=True"


def B(h, r, keep=None):
    if keep is not None:
        h = [x for x, k in zip(h, keep) if k]; r = [x for x, k in zip(r, keep) if k]
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")


def ci(d, t=T975_DF4):
    m = sum(d) / len(d); sd = st.stdev(d); hw = t * sd / math.sqrt(len(d))
    return m, m - hw, m + hw


def line(name, d):
    m, lo, hi = ci(d)
    return f"{name:38s} {' '.join(f'{x:+.3f}' for x in d)} | mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | +{sum(x > 0 for x in d)}/5"


def rows(p):
    return [l.rstrip("\n").split("\t", 1) for l in open(p) if "\t" in l]


def vid(path):
    m = re.match(r"video_(\d+)_", os.path.basename(path))
    assert m, path
    return int(m.group(1))


# ---------------------------------------------------------------- self-check first (recorded numbers)
j = json.load(open(os.path.join(W8, "..", "pose_augmentation_decodes", "pa_aug_s42", "beam5_test.json")))
b0 = B(j["beam_hyps"], j["refs"]); assert abs(b0 - 11.443) < 0.0005, b0
tsv = rows(os.path.join(W8, "test.tsv"))
ref_raw = [t for _, t in tsv]
wnull = corpus_wer([detokenize("it is sally .")] * 800, [detokenize(r) for r in ref_raw])
assert abs(wnull - 0.9504) < 0.00005, wnull
print(f"SELF-CHECK OK: pa_aug_s42 beam-5 test {b0:.3f} (recorded 11.443); WER null floor 'it is sally .' {wnull:.4f} (recorded 0.9504)")

# ---------------------------------------------------------------- manifests and subsets
train = rows(os.path.join(W8, "train.tsv")); val = rows(os.path.join(W8, "val.tsv"))
assert len(train) == 12439 and len(tsv) == 800 and len(val) == 800
train_lines = [l.rstrip("\n") for l in open(os.path.join(W8, "train.tsv")) if "\t" in l]
held_paths = {p for p, _ in val} | {p for p, _ in tsv}
for s in SEEDS:
    subs = {a: [l.rstrip("\n") for l in open(os.path.join(W8, "manifests", f"{a}_s{s}.tsv")) if "\t" in l] for a in ARMS}
    regen = {"R025": RS.within(train_lines, 0.25, s), "R050": RS.within(train_lines, 0.5, s),
             "R100": RS.within(train_lines, 1.0, s), "V050": RS.whole_video(train_lines, 6241, s)}
    for a in ARMS:
        assert subs[a] == regen[a], f"{a}_s{s}: local manifest != pinned builder output"
        assert not ({l.split('\t')[0] for l in subs[a]} & held_paths), f"{a}_s{s} leaks val/test"
    assert set(subs["R025"]) <= set(subs["R050"]) <= set(subs["R100"]) and subs["R100"] == train_lines
    assert [len(subs[a]) for a in ARMS] == [3141, 6241, 12439, 6241]
    assert len({vid(l.split('\t')[0]) for l in subs["R025"]}) == 90
print("SUBSETS OK: all 20 local manifests == pinned builder output; nested; R100 == train; 0 val/test leaks; sizes exact")

# ---------------------------------------------------------------- progress.json (40 evals, best = greedy-val max)
for a in ARMS:
    for s in SEEDS:
        p = json.load(open(os.path.join(W8, "runs", f"{a}_s{s}", "progress.json")))
        ev = [r for r in p["history"] if "bleu4" in r]
        assert len(ev) == 40 and p["status"] == "done", (a, s, len(ev))
        mx = max(r["bleu4"] for r in ev)
        assert abs(p["best"]["bleu4"] - mx) < 1e-9, (a, s)
print("PROGRESS OK: 40 greedy-validation evaluations per task (20/20); best.pt == greedy-val maximum")

# ---------------------------------------------------------------- dumps
norm_ws = lambda t: " ".join(t.lower().split())             # arm_readouts / paired_bootstrap definition
norm_dt = lambda t: " ".join(detokenize(t).lower().split())  # detokenised variant
full_ws = {norm_ws(t) for _, t in train}; full_dt = {norm_dt(t) for _, t in train}
unseen = [norm_ws(r) not in full_ws for r in ref_raw]
unseen2 = [norm_dt(r) not in full_dt for r in ref_raw]
assert unseen == unseen2 and sum(unseen) == 570, (sum(unseen), sum(unseen2))
print(f"UNSEEN: {sum(unseen)} clips, identical under raw-lower and detokenised-lower definitions")

D, maxdiff = {}, 0.0
for a in ARMS:
    for s in SEEDS:
        for sp, refs_m in (("val", [t for _, t in val]), ("test", ref_raw)):
            for ck in ("best", "soup_last5"):
                jj = json.load(open(os.path.join(W8, "runs", f"{a}_s{s}", f"beam5_{sp}_{ck}.json")))
                assert jj["beam"] == 5 and len(jj["beam_hyps"]) == 800
                assert [" ".join(x.split()) for x in jj["refs"]] == [" ".join(x.split()) for x in refs_m], (a, s, sp, ck)
                assert jj["checkpoint"].endswith(("best.pt" if ck == "best" else "soup_last5.pt")), jj["checkpoint"]
                b = B(jj["beam_hyps"], jj["refs"])
                maxdiff = max(maxdiff, abs(b - jj["beam_result"]["metrics"][KEY]))
                D[(a, s, sp, ck)] = (jj["beam_hyps"], refs_m)
print(f"DUMPS OK: 80 dumps, refs in manifest order, beam 5, checkpoint names match; max |rescore - recorded| = {maxdiff:.2e}")

S = {}
for k, (h, r) in D.items():
    S[k + ("all",)] = B(h, r)
    if k[2] == "test":
        S[k + ("uns",)] = B(h, r, unseen)

# ---------------------------------------------------------------- Q1
print("\n=== Q1 (R100 - R050, beam-5 best.pt; df 4, t 2.776) ===")
q = lambda sp, ck, sub: [S[("R100", s, sp, ck, sub)] - S[("R050", s, sp, ck, sub)] for s in SEEDS]
d_off, d_uns, d_val, d_soup = q("test", "best", "all"), q("test", "best", "uns"), q("val", "best", "all"), q("test", "soup_last5", "all")
d_vsoup = q("val", "soup_last5", "all")
for n_, d in (("Δ official-800 test", d_off), ("Δ unseen-570 test", d_uns), ("Δ validation", d_val),
              ("Δ official test soup (secondary)", d_soup), ("Δ validation soup (secondary)", d_vsoup)):
    print(line(n_, d))
try:
    from scipy.stats import t as tdist
    print(f"t check: scipy t.ppf(0.975, 4) = {tdist.ppf(0.975, 4):.4f}")
except Exception as e:
    print("scipy unavailable:", e)
x = [math.log2(N[a]) for a in ("R025", "R050", "R100")]
for sub in ("all", "uns"):
    sl = [float(np.polyfit(x, [S[(a, s, "test", "best", sub)] for a in ("R025", "R050", "R100")], 1)[0]) for s in SEEDS]
    print(line(f"OLS slope per doubling ({sub})", sl))
mo, lo_o, hi_o = ci(d_off); mu, lo_u, hi_u = ci(d_uns)
if lo_u > 0: rd = "(i)"
elif lo_o > 0 and lo_u <= 0 <= hi_u and mu < 0.3: rd = "(ii)"
elif lo_o > 0 and lo_u <= 0 <= hi_u: rd = "(ii')"
elif lo_o <= 0 <= hi_o and lo_u <= 0 <= hi_u: rd = "(iii)"
else: rd = "(iv)"
print(f"Q1 reading: {rd}; val sign {'agrees' if (ci(d_val)[0] > 0) == (mo > 0) else 'DISAGREES'}; "
      f"soup sign {'agrees' if (ci(d_soup)[0] > 0) == (mo > 0) else 'DISAGREES'}")

# pooled paired sentence bootstrap via sacreBLEU sufficient statistics (verified against corpus_bleu4 first)
bl = BLEU(tokenize="13a")
def stats(h, r):
    return np.array(bl._extract_corpus_statistics([detokenize(x) for x in h], [[detokenize(x) for x in r]]), dtype=float)
def score(st_):
    return bl._compute_score_from_stats(list(st_)).score
SR = {s: stats(D[("R100", s, "test", "best")][0], ref_raw) for s in SEEDS}
S5 = {s: stats(D[("R050", s, "test", "best")][0], ref_raw) for s in SEEDS}
assert abs(score(SR[42].sum(0)) - S[("R100", 42, "test", "best", "all")]) < 1e-9
assert abs(score(S5[42][unseen].sum(0)) - S[("R050", 42, "test", "best", "uns")]) < 1e-9
for lab, keep in (("official", [True] * 800), ("unseen", unseen)):
    idx = np.array([i for i, k in enumerate(keep) if k]); rng = np.random.default_rng(0); out = np.empty(2000)
    for t_ in range(2000):
        ii = rng.choice(idx, size=len(idx), replace=True)
        out[t_] = np.mean([score(SR[s][ii].sum(0)) - score(S5[s][ii].sum(0)) for s in SEEDS])
    lo, hi = np.percentile(out, [2.5, 97.5])
    print(f"pooled sentence bootstrap Δ {lab} (test sampling only): [{lo:+.3f}, {hi:+.3f}]")

# ---------------------------------------------------------------- Q2
print("\n=== Q2 (V050 - R050 on held-episode test clips minus kept-episode clips; cut video dropped) ===")
per_video = Counter(vid(p) for p, _ in train)
tv = [vid(p) for p, _ in tsv]
out = {k: [] for k in ("off", "uns", "held", "kept", "seen", "seen_dt", "lexA", "lexB")}
for s in SEEDS:
    vr = rows(os.path.join(W8, "manifests", f"V050_s{s}.tsv")); rr = rows(os.path.join(W8, "manifests", f"R050_s{s}.tsv"))
    kv = Counter(vid(p) for p, _ in vr)
    cut = [v for v in kv if kv[v] < per_video[v]]
    assert len(cut) == 1, (s, cut)
    held = [v not in kv for v in tv]; kept = [v in kv and v != cut[0] for v in tv]
    hv, hr = D[("V050", s, "test", "best")][0], D[("R050", s, "test", "best")][0]
    diff = lambda m: B(hv, ref_raw, m) - B(hr, ref_raw, m)
    U = lambda m: [a_ and u for a_, u in zip(m, unseen)]
    out["held"].append(diff(held)); out["kept"].append(diff(kept))
    out["off"].append(out["held"][-1] - out["kept"][-1]); out["uns"].append(diff(U(held)) - diff(U(kept)))
    for key, nf in (("seen", norm_ws), ("seen_dt", norm_dt)):
        vs = {nf(t) for _, t in vr}; rs = {nf(t) for _, t in rr}
        c = lambda m, st_: sum(1 for a_, r in zip(m, ref_raw) if a_ and nf(r) in st_)
        out[key].append((c(held, vs) - c(held, rs)) - (c(kept, vs) - c(kept, rs)))
    # secondary: both-arms lexicon, two content-word definitions (learning_curve's and arm_readouts')
    cwA = lambda t: content_words(detokenize(t).lower())
    cwB = lambda t: [w for w in t.lower().split() if w.isalpha() and w not in STOP]
    for key, cw in (("lexA", cwA), ("lexB", cwB)):
        Vw = {w for _, t in vr for w in cw(t)}; Rw = {w for _, t in rr for w in cw(t)}
        both = [all(w in Vw and w in Rw for w in cw(r)) for r in ref_raw]
        L = lambda m: [a_ and b_ for a_, b_ in zip(m, both)]
        out[key].append(diff(L(held)) - diff(L(kept)))
    print(f"seed {s}: cut video_{cut[0]} | V050 videos {len(kv)} | held {sum(held)} (unseen {sum(U(held))}) "
          f"kept {sum(kept)} (unseen {sum(U(kept))}) | dropped {800 - sum(held) - sum(kept)} | "
          f"held diff {out['held'][-1]:+.3f} kept diff {out['kept'][-1]:+.3f}")
print(line("DiD official (co-primary)", out["off"]))
print(line("DiD unseen (co-primary)", out["uns"]))
print(line("single diff held (V050-R050)", out["held"]))
print(line("single diff kept (V050-R050)", out["kept"]))
print(line("DiD both-lexicon secondary (detok def)", out["lexA"]))
print(line("DiD both-lexicon secondary (raw def)", out["lexB"]))
print(f"seen-count DiD (raw-lower): {out['seen']}   (detok-lower): {out['seen_dt']}")
res = lambda d: 1 if ci(d)[1] > 0 else (-1 if ci(d)[2] < 0 else 0)
ro, ru = res(out["off"]), res(out["uns"])
if -1 in (ro, ru): r2 = "R1"
elif ro == 0 and ru == 0 and abs(ci(out["off"])[0]) < 0.5 and abs(ci(out["uns"])[0]) < 0.5: r2 = "R2"
else: r2 = "R3"
print(f"Q2 reading (both co-primaries must meet R2): {r2}; per co-primary: official "
      f"{'R2-eligible' if ro == 0 and abs(ci(out['off'])[0]) < 0.5 else 'not R2'}, unseen "
      f"{'R2-eligible' if ru == 0 and abs(ci(out['uns'])[0]) < 0.5 else 'not R2'}")

# ---------------------------------------------------------------- arm means (descriptive)
print("\n=== per-arm means over 5 seeds (beam-5 best.pt unless noted) ===")
for a in ARMS:
    tb = [S[(a, s, "test", "best", "all")] for s in SEEDS]; vb = [S[(a, s, "val", "best", "all")] for s in SEEDS]
    ub = [S[(a, s, "test", "best", "uns")] for s in SEEDS]
    ts = [S[(a, s, "test", "soup_last5", "all")] for s in SEEDS]; vs_ = [S[(a, s, "val", "soup_last5", "all")] for s in SEEDS]
    w = [corpus_wer([detokenize(x) for x in D[(a, s, "test", "best")][0]], [detokenize(x) for x in ref_raw]) for s in SEEDS]
    own = []
    for s in SEEDS:
        tr = {norm_ws(t) for _, t in rows(os.path.join(W8, "manifests", f"{a}_s{s}.tsv"))}
        own.append(sum(norm_ws(r) in tr for r in ref_raw))
    print(f"{a}: val {st.mean(vb):.3f} ({st.stdev(vb):.3f}) | val soup {st.mean(vs_):.3f} | test {st.mean(tb):.3f} "
          f"({st.stdev(tb):.3f}) | test soup {st.mean(ts):.3f} | unseen {st.mean(ub):.3f} | WER {st.mean(w):.3f} | "
          f"seen(own) {st.mean(own):.1f}")

# ---------------------------------------------------------------- §2 soup sign check applied to Q2 (secondary only)
print("\n=== Q2 soup (secondary; §2: if best.pt and soup disagree in sign, report both, pick neither) ===")
so = {"off": [], "uns": []}
for s in SEEDS:
    vr = rows(os.path.join(W8, "manifests", f"V050_s{s}.tsv")); kv = Counter(vid(p) for p, _ in vr)
    cut = [v for v in kv if kv[v] < per_video[v]][0]
    held = [v not in kv for v in tv]; kept = [v in kv and v != cut for v in tv]
    hv, hr = D[("V050", s, "test", "soup_last5")][0], D[("R050", s, "test", "soup_last5")][0]
    diff = lambda m: B(hv, ref_raw, m) - B(hr, ref_raw, m)
    U = lambda m: [a_ and u for a_, u in zip(m, unseen)]
    so["off"].append(diff(held) - diff(kept)); so["uns"].append(diff(U(held)) - diff(U(kept)))
print(line("DiD official, soup", so["off"]))
print(line("DiD unseen, soup", so["uns"]))
