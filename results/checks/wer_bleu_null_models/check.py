"""WER null floor, brevity trade, error decomposition and worked BLEU/WER example (Results 'WER', Methodology, appendix).

Recomputes from the saved per-clip decodes of the 800 official test clips: RGB-fusion, 40-epoch pre-trained and
DWPose+CTC single models (greedy and beam 5) and the six-member ensemble (beam 10, length penalty 0.3).
"""
import json, sys
from pathlib import Path
import numpy as np
import jiwer
import sacrebleu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from slt.metrics import detokenize, corpus_bleu4, corpus_wer, bleu_signature
import metric_suite
from restore_references import sentence

SEM = ROOT / "results/semantic_adequacy_grading"
def load(p):
    return sorted((json.loads(l) for l in open(p)), key=lambda r: r["idx"])
rgb = load(ROOT / "results/semantic_adequacy_grading/hyps_rgbfusion.jsonl")
vlp = load(ROOT / "results/semantic_adequacy_grading/hyps_vlp.jsonl")
dwc = load(ROOT / "results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl")
ens = load(ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl")
rawrefs = [r["ref"] for r in rgb]
assert len(rawrefs) == 800 and all(r["ref"] == q["ref"] for r, q in zip(rgb, ens))
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    m = [l.rstrip("\n").split("\t", 1)[1].strip() for l in open(man) if "\t" in l]
    assert m == rawrefs, "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")
R = [detokenize(r) for r in rawrefs]
N = len(R)

# ---- GATE: recorded RGB-fusion beam-5 and S6 ensemble scores
H_rgb = [detokenize(r["hyp_beam"]) for r in rgb]
b, w = corpus_bleu4(H_rgb, R), corpus_wer(H_rgb, R)
print(f"GATE RGB-fusion beam 5: BLEU-4 {b:.3f} WER {w:.4f} (recorded 10.335 / 0.9497)")
H_ens = [detokenize(r["hyp_beam"]) for r in ens]
be, we = corpus_bleu4(H_ens, R), corpus_wer(H_ens, R)
print(f"GATE six-member ensemble: BLEU-4 {be:.3f} WER {we:.4f} (recorded 11.802 / 0.9069)")
if (round(b, 3), round(w, 4), round(be, 3), round(we, 4)) != (10.335, 0.9497, 11.802, 0.9069):
    sys.exit("GATE FAILED")

print(f"\nTest set: {N} official Communication clips (references only, no training data used)  [c12-050 c12-053]")
print(f"BLEU-4 computed with {bleu_signature()}  [c06-002 c06-007]")
print(f"detokenised text, sacreBLEU 13a tokeniser, scale 0 to 100: scores below lie between "
      f"{0:d} and {100:d} for 3 single models x 2 decodings, 4-gram  [c06-001 c06-007 c10-022]")

# ---- null models: one empty, two constants (named by digest), WER on detokenised text
C_HELLO = detokenize(sentence("0edb43084d1b3190790eefd9557c07a55a9256ce"))
C_SALLY = detokenize(sentence("d49b14b483169c049f2b7574cde3820ce5e114c2"))
nulls = [("empty output", ""), ("constant A (sha1 0edb4308)", C_HELLO), ("constant B (sha1 d49b14b4)", C_SALLY)]
print("\nNULL MODELS (no video used), WER and BLEU-4 on the official test set")
print(f"{len(nulls)} null models scored for WER: 1 empty output and {len(nulls)-1} constant sentences  [c07-067]")
nw = {}
for name, s in nulls:
    ww = corpus_wer([s] * N, R); bb = corpus_bleu4([s] * N, R) if s else 0.0
    nw[name] = ww
    print(f"{name:30s} WER {ww:.4f}  BLEU-4 {bb:.2f}")
    tag = {"constant A (sha1 0edb4308)": "c12-045", "constant B (sha1 d49b14b4)": "c12-046 c12-053 c12-054 c02-016"}.get(name, "")
    if tag:
        print(f"  {name}: WER {ww:.4f} BLEU-4 {bb:.2f}  [{tag}]")
floor = min(nw.values())
print(f"empty output WER {nw['empty output']:.3f}; lowest null-model WER (floor) {floor:.3f}; "
      f"RGB-fusion {w:.3f}; ensemble {we:.3f}  [c14-079 c12-051]")
print(f"constant B WER {nw['constant B (sha1 d49b14b4)']:.4f}, RGB-fusion WER {w:.4f}, difference {w - nw['constant B (sha1 d49b14b4)']:+.4f} -> {abs(w - nw['constant B (sha1 d49b14b4)']):.4f}  [c12-053]")
print(f"constant B BLEU-4 {corpus_bleu4([C_SALLY]*N, R):.2f}, RGB-fusion BLEU-4 {b:.2f}  [c12-054]")
print(f"constant WER {floor:.3f}, RGB-fusion WER {w:.3f}, final ensemble {we:.3f}  [c12-051 c02-016]")

# ---- truncation to three words
T3 = [" ".join(h.split()[:3]) for h in H_rgb]
bt, wt = corpus_bleu4(T3, R), corpus_wer(T3, R)
print("\nBREVITY: RGB-fusion output cut to its first three words")
print(f"truncated to three words: WER {wt:.4f} ({wt:.3f}) BLEU-4 {bt:.2f}  [c07-068 c12-051]")
print(f"RGB-fusion WER {w:.4f} -> {wt:.4f}, below the ensemble's {we:.4f}; BLEU-4 {b:.2f} -> {bt:.2f}  [c12-055 c21-003]")
print(f"one model's WER {w:.4f} -> {wt:.4f}, BLEU-4 {b:.2f} -> {bt:.2f}, truncated to three words  [c21-003]")

# ---- worked example, clip 104 of the 40-epoch pre-trained model, beam 5
r104 = vlp[104]
ref_t, hyp_t = r104["ref"].split(), r104["hyp_beam"].split()
lcs_hits = jiwer.process_words(r104["ref"], r104["hyp_beam"]).hits
s0 = sacrebleu.sentence_bleu(detokenize(r104["hyp_beam"]), [detokenize(r104["ref"])], smooth_method="none").score
grams4 = {tuple(hyp_t[i:i+4]) for i in range(len(hyp_t) - 3)} & {tuple(ref_t[i:i+4]) for i in range(len(ref_t) - 3)}
wer104 = jiwer.wer(r104["ref"], r104["hyp_beam"])
print("\nWORKED EXAMPLE: clip 104 of the 40-epoch pre-trained pose-only model (beam width 5)")
print(f"model 40-epoch pre-trained, beam 5 (corpus BLEU-4 {corpus_bleu4([detokenize(r['hyp_beam']) for r in vlp], R):.2f}, the recorded 10.27)  [c04-005]")
print(f"output tokens {len(hyp_t)}, matching reference tokens {lcs_hits} (full stop included)  [c04-004]")
print(f"matching 4-token runs {len(grams4)}; sentence BLEU-4 {s0:.2f}  [c04-004]")
print(f"token WER {wer104:.2f} (tokenised, {len(ref_t)} reference tokens)  [c04-004]")

# ---- ROUGE beta and WER above 1.0
print(f"\nROUGE-L: whitespace tokens, punctuation kept, beta = {metric_suite.ROUGE_BETA} (read from metric_suite.py)  [c06-008]")
systems = {}
for nm, d in [("DWPose+CTC", dwc), ("40-ep pre-trained", vlp), ("RGB-fusion", rgb)]:
    for k, kn in (("hyp_greedy", "greedy"), ("hyp_beam", "beam 5")):
        systems[f"{nm} {kn}"] = [detokenize(r[k]) for r in d]
print("\nERROR DECOMPOSITION and length, six sets of outputs (3 single models x greedy / beam 5)")
print(f"{'system':28s}{'WER':>8s}{'S/edits':>9s}{'S/N':>7s}{'len ratio':>10s}{'mean len':>9s}")
wers, lens, subsh, ratios = [], [], {}, {}
for nm, H in systems.items():
    o = jiwer.process_words(R, [h if h.strip() else "<empty>" for h in H])
    Nw = o.substitutions + o.deletions + o.hits
    edits = o.substitutions + o.deletions + o.insertions
    wer_ = edits / Nw
    ratio = sum(len(h.split()) for h in H) / sum(len(r.split()) for r in R)
    ml = np.mean([len(h.split()) for h in H])
    wers.append(wer_); lens.append(ml); subsh[nm] = 100 * o.substitutions / edits; ratios[nm] = ratio
    print(f"{nm:28s}{wer_:8.4f}{subsh[nm]:8.1f}%{o.substitutions/Nw:7.3f}{ratio:10.3f}{ml:9.2f}")
    if "greedy" in nm and o.insertions > o.hits:
        print(f"  {nm}: insertions {o.insertions} exceed matched reference words {o.hits}, WER {wer_:.4f} above 1.0  [c06-006]")
o = jiwer.process_words(R, H_rgb); Nw = o.substitutions + o.deletions + o.hits
print(f"RGB-fusion beam 5: S {o.substitutions} D {o.deletions} I {o.insertions} of N {Nw} reference words;"
      f" substitutions are {o.substitutions/Nw:.2f} of reference length  [c18-013]")
print("substitutions as % of word edits, six sets of outputs: " + ", ".join(f"{v:.0f}" for v in subsh.values())
      + f"  (range {round(min(subsh.values()))}-{round(max(subsh.values()))} %)  [c18-013]")
beam_r = [ratios[k] for k in ratios if "beam" in k]
print("output / reference length ratio, three beam-5 systems: " + ", ".join(f"{x:.2f}" for x in beam_r)
      + f"  (range {min(beam_r):.2f}-{max(beam_r):.2f})  [c18-015]")
cor = np.corrcoef(lens, wers)[0, 1]
print(f"correlation of mean output length with WER across the six sets of outputs: {cor:+.3f}  [c18-088]")

# ---- per-sentence oracle over the six sets
tot = 0.0; ntok = 0
for i in range(N):
    errs = []
    for H in systems.values():
        h = H[i] if H[i].strip() else "<empty>"
        errs.append(jiwer.process_words(R[i], h).wer * len(R[i].split()))
    tot += min(errs); ntok += len(R[i].split())
print(f"per-sentence oracle WER over the six sets of outputs: {tot/ntok:.3f}  [c18-014]")

# ---- sampling variation of the headline
rng = np.random.default_rng(0)
sc = []
for _ in range(2000):
    ix = rng.integers(0, N, N)
    sc.append(corpus_bleu4([H_ens[i] for i in ix], [R[i] for i in ix]))
lo, hi = np.percentile(sc, [2.5, 97.5])
print(f"\nsentence bootstrap (2000 resamples, seed 0) of the ensemble's BLEU-4 {be:.2f}: 95% interval [{lo:.2f}, {hi:.2f}],"
      f" half-width {(hi-lo)/2:.1f} points, wider than a difference of about one BLEU-4 point (one point lies within the variation)  [c14-009]")
