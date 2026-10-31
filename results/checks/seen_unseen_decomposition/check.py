"""Seen/unseen decomposition, verbatim overlap, split definition and per-signer breakdown (official Communication test split).

Recomputes from the saved test decodes of three single models (beam 5; semantic_adequacy_grading) and of the
reported ensemble (LP 0.3), the training-target digests, and the dataset spreadsheet (Split, Video_Name, Signer_ID).
"""
import hashlib, json, random, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4

SG = ROOT / "results/semantic_adequacy_grading"
MODELS = {"DWPose+CTC": SG / "hyps_dwpose_ctc.jsonl", "VLP-40": SG / "hyps_vlp.jsonl",
          "RGB-fusion": SG / "hyps_rgbfusion.jsonl"}
S6 = ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl"
digests = set(open(ROOT / "results/train_target_hashes.txt").read().split())
seen_fn = lambda r: hashlib.sha1(r.encode("utf-8")).hexdigest() in digests

df = pd.read_excel(ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
man = ROOT / "comm_run/manifests/test.tsv"
rows_m = [l.rstrip("\n").split("\t", 1) for l in open(man) if "\t" in l]
clips = [Path(p).name[:-4] for p, _ in rows_m]
sid = dict(zip(df.Video_Clip_Name.astype(str), df.Signer_ID)); vid = dict(zip(df.Video_Clip_Name.astype(str), df.Video_Name))
ids = [sid[c] for c in clips]; vids = [vid[c] for c in clips]

def load(p, key="hyp_beam"):
    rows = sorted((json.loads(l) for l in open(p)), key=lambda r: r["idx"])
    assert [r["ref"].strip() for r in rows] == [m[1].strip() for m in rows_m], "references differ from test manifest"
    return [detokenize(r[key]) for r in rows], [detokenize(r["ref"]) for r in rows]

# ---- split definition ----
sp = df.Split.value_counts()
ntr, nva, nte = int(sp["train"]), int(sp["dev"]), int(sp["test"])
print(f"Split sizes: train {ntr:,}, validation {nva}, test {nte}; total {len(df):,}  [c06-016 c21-016 c09-001 c09-007 c20-011]")
refs0 = [detokenize(m[1]) for m in rows_m]
seen = [seen_fn(r) for r in refs0]
ns = sum(seen); nu = len(seen) - ns
print(f"Test references occurring verbatim in training: {ns} of {len(seen)} ({100*ns/len(seen):.2f}%); unseen {nu}  [c14-010 c14-074 c21-016 c01-026]")
print(f"Share of test references seen in training {ns/len(seen):.2f}: over a quarter (> 0.25)  [c01-026]")
trvids = set(df[df.Split == "train"].Video_Name); tevids = set(df[df.Split == "test"].Video_Name)
print(f"Test videos {len(tevids)}; of these in training {len(tevids & trvids)} of {len(tevids)}  [c07-060]")
n1, n2 = ids.count(1), ids.count(2)
print(f"Test clips by signer: signer 1 {n1}, signer 2 {n2}; two most frequent signers {n1+n2} of {len(ids)} = {100*(n1+n2)/len(ids):.1f}%  [c05-053]")

# ---- gate: reported ensemble reproduces the recorded score ----
hs, rs = load(S6)
b = corpus_bleu4(hs, rs)
print(f"GATE S6 ensemble (beam 10, LP 0.3) official test BLEU-4 {b:.3f} (recorded 11.802)")
if round(b, 3) != 11.802: sys.exit("GATE FAILED")

# ---- seen/unseen for three single models, beam 5 ----
print(f"\nSetting: all {len(refs0)} official test clips, beam width 5, three single models")
su = {}; out_verb = {}
for name, p in MODELS.items():
    h, r = load(p)
    assert len(h) == 800
    allb = corpus_bleu4(h, r)
    s = [(a, c) for a, c, f in zip(h, r, seen) if f]; u = [(a, c) for a, c, f in zip(h, r, seen) if not f]
    bs = corpus_bleu4([a for a, _ in s], [c for _, c in s]); bu = corpus_bleu4([a for a, _ in u], [c for _, c in u])
    su[name] = (bs, bu)
    print(f"{name:11s} all {allb:6.2f} (n=800)  seen {bs:6.2f} (n={len(s)})  unseen {bu:6.2f} (n={len(u)})  ratio {bs/bu:.2f}  [c11-059 c11-066 c11-067 c14-075 c09-024]")
print(f"GATE recorded beam-5 test scores 9.52 / 10.27 / 10.33 for DWPose+CTC / VLP-40 / RGB-fusion")
for name, want in zip(MODELS, (9.52, 10.27, 10.33)):
    h, r = load(MODELS[name])
    if round(corpus_bleu4(h, r), 2) != want: sys.exit("GATE FAILED " + name)
lo = min(v[0] for v in su.values()); hi = max(v[0] for v in su.values())
ulo = min(v[1] for v in su.values()); uhi = max(v[1] for v in su.values())
rlo = min(v[0]/v[1] for v in su.values()); rhi = max(v[0]/v[1] for v in su.values())
print(f"Seen-target range {lo:.1f} to {hi:.1f}; unseen-target range {ulo:.2f} to {uhi:.2f}; ratio {rlo:.1f} to {rhi:.1f}  [c11-059 c14-075 c11-066]")
print(f"Seen about 30 (mean {sum(v[0] for v in su.values())/3:.1f}), unseen about 5 (mean {sum(v[1] for v in su.values())/3:.1f}); 230 seen / 570 unseen clips; 3 models  [c11-066 c14-075]")

# ---- verbatim-training share of outputs ----
sh = {}
for name, p in list(MODELS.items()) + [("S6", S6)]:
    h, _ = load(p)
    sh[name] = 100 * sum(seen_fn(x) for x in h) / len(h)
    print(f"Outputs that are verbatim training sentences, {name}: {sh[name]:.1f}%  [c10-038]")

# ---- per-signer gap between the two principal signers, single models ----
def gap(h, r, lab, mask=None):
    ix = [i for i in range(800) if lab[i] in (1, 2) and (mask is None or mask[i])]
    g = {}
    for k in (1, 2):
        j = [i for i in ix if lab[i] == k]
        g[k] = corpus_bleu4([h[i] for i in j], [r[i] for i in j])
    return g[1] - g[2]

def perm_p(h, r, nperm=2000, seed=0):
    rng = random.Random(seed)
    obs = gap(h, r, ids)
    byv = {}
    for i, v in enumerate(vids): byv.setdefault(v, []).append(i)
    cnt = 0
    for _ in range(nperm):
        lab = list(ids)
        for v, ix in byv.items():
            s = [ids[i] for i in ix]; rng.shuffle(s)
            for i, x in zip(ix, s): lab[i] = x
        if abs(gap(h, r, lab)) >= abs(obs): cnt += 1
    return obs, (cnt + 1) / (nperm + 1)

print("\nSigner 1 minus signer 2, official test, beam 5 (within-video permutation, 2000 draws, seed 0)")
gaps = []; ug = []; ps = []
for name, p in MODELS.items():
    h, r = load(p)
    obs, pv = perm_p(h, r)
    um = gap(h, r, ids, [not f for f in seen])
    gaps.append(obs); ug.append(um); ps.append(pv)
    print(f"{name:11s} signer1-signer2 {obs:+.2f}  perm p {pv:.3f}  unseen-target only {um:+.2f}")
hs6, rs6 = load(S6, "hyp_beam")
g6 = gap(hs6, rs6, ids); u6 = gap(hs6, rs6, ids, [not f for f in seen])
print(f"S6 ensemble (not a single model) signer1-signer2 {g6:+.2f}, unseen-target only {u6:+.2f}")
print(f"Single-model gap range {min(gaps):+.3f} to {max(gaps):+.3f} (to one decimal: {min(gaps):+.1f} to {max(gaps):+.1f}); permutation p between {min(ps):.2f} and {max(ps):.2f}  [c18-023]")
print("  (recorded p 0.02 to 0.29 came from an earlier 300-draw run (seed not kept); p agrees only within permutation noise)  [c18-023]")
print(f"Unseen-target gap, single models: {max(ug):+.2f} to {min(ug):+.2f}; with the S6 ensemble the upper end is {u6:+.2f}  [c18-024]")
