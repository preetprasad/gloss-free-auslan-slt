"""Wave 8 real-data learning curve (job 21734) and its count-only mechanism check.

Arms R025 / R050 / R100 (nested real-only subsets of 3,141 / 6,241 / 12,439 pairs) and V050 (whole episodes, 6,241),
seeds 42/101/202/303/404, beam 5, best checkpoint. Recomputes BLEU-4, unseen-target BLEU-4, content recall from the saved
decodes; the paired differences with seed-level t intervals; the held-minus-kept episode contrast; and the
count-only logistic prediction of R100 content recall.
"""
import json, statistics as st, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/tools"))
from slt.metrics import corpus_bleu4, detokenize
from stitch_lexicon import STOP
from arm_readouts import content, readouts

ARMS, SEEDS = ["R025", "R050", "R100", "V050"], [42, 101, 202, 303, 404]
T975_DF4 = 2.7764  # published two-sided 95% t quantile, 4 degrees of freedom
D = json.load(open(HERE / "derived.json"))
unseen = D["unseen"]
man = ROOT / "comm_run/manifests/train.tsv"
if man.exists():
    tr = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split()) for l in open(man, encoding="utf-8") if "\t" in l}
    tm = ROOT / "comm_run/manifests/test.tsv"
    refs0 = json.load(open(HERE / "data/R100_s42_test.json"))["refs"]
    assert [" ".join(r.lower().split()) not in tr for r in refs0] == unseen, "unseen flags differ from the training manifest"
    print("unseen flags match comm_run/manifests/train.tsv")
assert sum(unseen) == 570 and len(unseen) == 800

def bleu(h, r, keep=None):
    if keep is not None:
        h = [x for x, k in zip(h, keep) if k]; r = [x for x, k in zip(r, keep) if k]
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")

def interval(d):
    m = st.mean(d); hw = T975_DF4 * st.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw

R = {}
for a in ARMS:
    for s in SEEDS:
        for sp in ("test", "val"):
            j = json.load(open(HERE / "data" / f"{a}_s{s}_{sp}.json"))
            b = bleu(j["beam_hyps"], j["refs"])
            if abs(b - j["recorded_bleu4"]) > 1e-6:
                sys.exit(f"GATE FAILED {a}_s{s} {sp}: {b} vs recorded {j['recorded_bleu4']}")
            R[(a, s, sp)] = (j["beam_hyps"], j["refs"], b)
print("GATE: all 40 decodes re-score to the BLEU-4 recorded by the original job (<1e-6); unseen clips = 570 of 800")

M = {}
for a in ARMS:
    v = [R[(a, s, "val")][2] for s in SEEDS]; t = [R[(a, s, "test")][2] for s in SEEDS]
    u = [bleu(R[(a, s, "test")][0], R[(a, s, "test")][1], unseen) for s in SEEDS]
    c = [readouts(R[(a, s, "test")][0], R[(a, s, "test")][1], STOP)["content_recall"] for s in SEEDS]
    M[a] = dict(val=v, test=t, uns=u, crec=c)
tagr = {"R025": "c10-012", "R050": "c10-013", "R100": "c10-014", "V050": "c10-015"}
lab = {"R025": "25% of each episode", "R050": "50% of each episode", "R100": "All training pairs", "V050": "Whole episodes, 50%"}
print(f"Official test set: {len(unseen)} clips  [c10-017]; unseen-target clips: {sum(unseen)}  [c10-018]")
print("\nSetting: five seeds, beam 5, best checkpoint; unseen = %d of 800 test clips  [c10-010 c10-011 c10-021 c14-049]" % sum(unseen))
print("Subset sizes (rows of the saved training manifests, seed 42): " +
      ", ".join(f"{a} {D['sizes'][a + '_s42']:,}" for a in ARMS) + "  [c10-010 c10-016 c14-049]")
print("\nTable (five-seed means): pairs / epochs / val / test / unseen / content recall")
for a in ARMS:
    m = M[a]
    print(f"{lab[a]:22s} {D['sizes'][a+'_s42']:,}  {D['epochs'][a+'_s42']}  {st.mean(m['val']):.2f}  {st.mean(m['test']):.2f}  "
          f"{st.mean(m['uns']):.2f}  {st.mean(m['crec']):.1f}%  [{tagr[a]}]")
print("  (epochs read from the saved progress.json of each run)")
print(f"\nUnseen-target test BLEU-4, R050 {st.mean(M['R050']['uns']):.2f} -> R100 {st.mean(M['R100']['uns']):.2f} "
      f"(mean of five seeds)  [c01-019 c03-015 c10-018 c14-049]")

def diff(k, a, b): return [M[a][k][i] - M[b][k][i] for i in range(5)]
for k, name, ids in (("test", "official test (800 clips)", "c10-017 c10-020 c14-003 c14-027"),
                     ("uns", "unseen-target test (570 clips)", "c01-020 c10-018 c10-020 c14-049")):
    d = diff(k, "R100", "R050"); m, lo, hi = interval(d)
    print(f"R100 - R050, {name}: per seed {' '.join(f'{x:+.3f}' for x in d)}")
    print(f"  mean {m:+.2f} ({m:+.3f}), 95% t-interval {lo:+.2f} to {hi:+.2f}, positive in {sum(x>0 for x in d)} of 5 seeds  [{ids} c10-019]")
print(f"Test 6,241 -> 12,439 pairs: {st.mean(M['R050']['test']):.2f} -> {st.mean(M['R100']['test']):.2f}  [c10-017]")
print(f"Doubling from 6,241 to 12,439 pairs adds {interval(diff('test','R100','R050'))[0]:.2f} on test and "
      f"{interval(diff('uns','R100','R050'))[0]:.2f} on unseen targets  [c10-020 c14-003 c14-027 c14-049]")

# Q2: V050 - R050 on held-episode minus kept-episode test clips
did = {"off": [], "uns": []}
for s in SEEDS:
    q = D["q2"][str(s)]; hv, rf, _ = R[("V050", s, "test")]; hr = R[("R050", s, "test")][0]
    f = lambda keep: bleu(hv, rf, keep) - bleu(hr, rf, keep)
    cu = lambda c: [x and u for x, u in zip(c, unseen)]
    did["off"].append(f(q["held"]) - f(q["kept"])); did["uns"].append(f(cu(q["held"])) - f(cu(q["kept"])))
print("\nEpisode familiarity: (V050 - R050) on clips of held-out episodes minus the same difference on kept episodes, "
      "at equal size (five seeds; difference between two differences; the one cut episode dropped)  [c17-111]")
for k, name, i in (("off", "all affected test clips", "c17-112"), ("uns", "unseen-target clips", "c17-113")):
    m, lo, hi = interval(did[k])
    print(f"  {name}: {m:+.2f} ({lo:+.2f} to {hi:+.2f})  [{i}]")

# count-only mechanism: logistic hit ~ log(1+count), fitted on R025+R050
def wstats(h, r, cc):
    hits, occ, cnt = Counter(), Counter(), {}
    for x, y, cs in zip(h, r, cc):
        ws = content(y, STOP)
        for w, c in zip(ws, cs): cnt[w] = c
        a, b = Counter(content(x, STOP)), Counter(ws)
        hits.update(a & b); occ.update(b)
    return hits, occ, cnt
S = {(a, s): wstats(R[(a, s, "test")][0], R[(a, s, "test")][1], D["ref_counts"][f"{a}_s{s}"]) for a in ARMS for s in SEEDS}
xs, ys, ns = [], [], []
for (a, s), (hits, occ, cnt) in S.items():
    if a in ("R025", "R050"):
        for w, n in occ.items(): xs.append(np.log1p(cnt[w])); ys.append(hits[w]); ns.append(n)
x, y, n = (np.array(v, float) for v in (xs, ys, ns)); X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
for _ in range(100):
    p = 1 / (1 + np.exp(-(X @ b))); step = np.linalg.solve(X.T @ (X * (n * p * (1 - p))[:, None]), X.T @ (y - n * p)); b += step
    if np.abs(step).max() < 1e-12: break
obs, pred = [], []
for s in SEEDS:
    hits, occ, cnt = S[("R100", s)]
    ws = list(occ); xx = np.log1p([cnt[w] for w in ws]); nn = np.array([occ[w] for w in ws], float)
    pred.append(100 * (nn / (1 + np.exp(-(b[0] + b[1] * xx)))).sum() / nn.sum()); obs.append(100 * sum(hits.values()) / sum(occ.values()))
print(f"\nCount-only curve (logistic on log(1+training count), fitted on R025 and R050): predicts R100 content recall "
      f"{np.mean(pred):.2f}% against {np.mean(obs):.2f}% observed  [c18-091]")
