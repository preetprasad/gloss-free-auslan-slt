"""Content-word failure analysis of the saved test decodes (dissertation 'Error Analysis', Discussion, appendix retrieval row).

Single model: the 40-epoch pre-trained pose-only model, beam 5 (results/semantic_adequacy_grading/hyps_vlp.jsonl).
Ensemble: S6, beam 10 / LP 0.3 (results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl). Training targets: comm_run/manifests/train.tsv.
Definitions follow tools/content_failure_analysis.py and figures/make_failure_figure.py.
"""
import json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
from slt.metrics import detokenize, corpus_bleu4
from stitch_lexicon import STOP

rows = [json.loads(l) for l in open(ROOT / "results/semantic_adequacy_grading/hyps_vlp.jsonl")]
s6 = [json.loads(l) for l in open(ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl")]
tsv = lambda s: [l.rstrip("\n").split("\t")[1].strip() for l in open(ROOT / "comm_run/manifests" / f"{s}.tsv") if "\t" in l]
train = tsv("train"); test = tsv("test")
assert len(rows) == len(s6) == len(test) == 800
assert all(r["ref"].strip() == t for r, t in zip(rows, test)) and all(r["ref"].strip() == t for r, t in zip(s6, test)), "refs differ from test manifest"
print("references match comm_run/manifests/test.tsv: 800/800")

b = corpus_bleu4([detokenize(r["hyp_beam"]) for r in rows], [detokenize(r["ref"]) for r in rows])
print(f"GATE single model beam-5 test BLEU-4 {b:.4f} (recorded 10.27 for the 40-epoch pre-trained model)")
if round(b, 2) != 10.27:
    sys.exit("GATE FAILED")
b6 = corpus_bleu4([detokenize(r["hyp_beam"]) for r in s6], [detokenize(r["ref"]) for r in s6])
print(f"GATE S6 ensemble beam-10 LP 0.3 test BLEU-4 {b6:.3f} (recorded 11.802)")
if round(b6, 3) != 11.802:
    sys.exit("GATE FAILED")

C = lambda t: [w for w in t.lower().split() if w.isalpha() and w not in STOP]
freq = Counter(w for s in train for w in C(s)); trs = set(train)
print(f"\nContent words are the words outside a list of {len(STOP)} function words  [c06-009 c06-012 c10-028 c21-009]")
print("Content-word recall and precision are scored on a scale of 0 to 100%; higher is better  [c06-009]")
print(f"Setting: 40-epoch pre-trained pose-only model, beam width 5 (the 10.27 score above); ensemble panel at beam width 10  [c10-028 c10-030 c14-039 c21-009]")

# clips with a reference content word; content tokens
act = [r for r in rows if C(r["ref"])]
tot = sum(len(C(r["ref"])) for r in act)
print(f"\nClips with a reference content word: {len(act)} of {len(rows)}  [c10-023 c21-009]")
print(f"Reference content tokens: {tot:,}  [c21-008 c10-032]")
print(f"Mean reference content words per test clip: {tot/len(rows):.2f}  [c10-024]")

# recall by training frequency (multiset match per clip)
bk = lambda k: 0 if k <= 2 else 1 if k <= 50 else 2
nb, hb = Counter(), Counter()
for r in rows:
    h = Counter(C(r["hyp_beam"]))
    for w in C(r["ref"]):
        k = bk(freq[w]); nb[k] += 1
        if h[w] > 0:
            h[w] -= 1; hb[k] += 1
low = 100 * hb[0] / nb[0]
print(f"Recall of words seen twice or fewer: {low:.1f}%  [c10-031]")
print(f"Recall of words seen more than 50 times: {100*hb[2]/nb[2]:.1f}%  [c10-031 c10-033 c14-039]")
print(f"Tokens of words seen more than 50 times: {nb[2]:,} of {tot:,} ({100*nb[2]/tot:.0f}%)  [c10-032 c14-038]")
outw = {w for r in rows for w in C(r["hyp_beam"])}
fw = {w for r in rows for w in C(r["ref"]) if freq[w] > 50}
print(f"Of these frequent words, {100*len(fw & outw)/len(fw):.0f}% occur somewhere in the outputs ({len(fw & outw)} of {len(fw)} types)  [c10-033 c14-039]")

# recall vs input-blind null
hit = sum(sum((Counter(C(r["ref"])) & Counter(C(r["hyp_beam"]))).values()) for r in act)
top = [w for w, _ in freq.most_common(2)]
nul = sum(sum((Counter(C(r["ref"])) & Counter(top)).values()) for r in act)
print(f"\nContent-word recall of the model: {100*hit/tot:.1f}%  [c10-036]")
print(f"Null model (two most frequent training content words, every clip): {100*nul/tot:.1f}%  [c10-036]")
print(f"Ratio model / null: {hit/nul:.1f} times  [c10-036]")

# failure modes on all 800 clips
def cat(ref, hyp):
    rc, hc = C(ref), C(hyp)
    if not rc: return "no content word"
    if hyp.strip() == ref.strip(): return "exact"
    if all(w in hc for w in rc): return "reworded"
    if any(w in hc for w in rc): return "partial"
    return "memorised" if hyp.strip() in trs else "novel"
cc = Counter(cat(r["ref"], r["hyp_beam"]) for r in rows)
print(f"\nFailure modes over {len(rows)} clips: {dict(cc)}")
print(f"Clips classified: {len(rows)} (all test clips)  [c10-027]")
print(f"Verbatim training sentence sharing no content word with the reference: {100*cc['memorised']/800:.1f}%  [c10-027]")
print(f"Exact matches: {100*cc['exact']/800:.1f}%  [c10-027]")

# exact match on seen / unseen references, S6 ensemble (beam 10)
se = sn = ue = un = 0
for r in s6:
    ref, hyp = r["ref"].strip(), r["hyp_beam"].strip()
    if not C(ref): continue
    if ref in trs: sn += 1; se += hyp == ref
    else: un += 1; ue += hyp == ref
print(f"\nS6 ensemble, beam width 10: exact match on {ue} of the {un} unseen-target clips that contain content words  [c11-061 c21-009]")
print(f"S6 ensemble exact match on seen-target clips: {100*se/sn:.1f}% ({se} of {sn})  [c11-061]")
print(f"(single 40-epoch model, for comparison: unseen-target exact matches {sum(1 for r in rows if C(r['ref']) and r['ref'].strip() not in trs and r['hyp_beam'].strip()==r['ref'].strip())})")

