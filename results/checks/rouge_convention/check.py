"""ROUGE-L tokenisation convention and ROUGE-L / BLEU-1 ratio, recomputed from the saved
beam-5 test outputs of three single models (results/semantic_adequacy_grading/hyps_*.jsonl)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import metric_suite as M
from slt.metrics import detokenize

SG = ROOT / "results/semantic_adequacy_grading"
# (label, file key, recorded beam-5 test BLEU-4 of the original job = GATE, claim ids of its table row)
MODELS = [("Baseline model (DWPose + CTC)", "dwpose_ctc", 9.52),
          ("40-epoch pre-trained model", "vlp", 10.27),
          ("RGB-fusion model", "rgbfusion", 10.33)]
# published Auslan-Daily Communication pose results (benchmark paper Table 3): ROUGE-L, BLEU-1 (published values, cited)
PUBLISHED = [("SL-Luong", 37.27, 30.15), ("SL-Transformer", 35.65, 31.31), ("GASLT", 35.74, 28.19)]

def rouge(h, r, beta, detok):
    M.ROUGE_BETA = beta
    f = detokenize if detok else (lambda s: s)
    return 100.0 * sum(M.rouge_l(f(a), f(b)) for a, b in zip(h, r)) / len(h)

res = {}
for label, key, rec in MODELS:
    rows = [json.loads(l) for l in open(SG / f"hyps_{key}.jsonl")]
    refs = [r["ref"] for r in rows]
    hb = [r["hyp_beam"] for r in rows]; hg = [r["hyp_greedy"] for r in rows]
    dr = [detokenize(x) for x in refs]
    b4, b1, _, _ = M.corpus([detokenize(x) for x in hb], dr, hb, refs)
    g4 = M.corpus([detokenize(x) for x in hg], dr, hg, refs)[0]
    if round(b4, 2) != rec:   # GATE: recorded beam-5 BLEU-4 of the original job
        sys.exit(f"GATE FAILED: {key} BLEU-4 {b4:.3f} != recorded {rec}")
    res[key] = dict(label=label, n=len(rows), b4=b4, g4=g4, b1=b1,
                    d10=rouge(hb, refs, 1.0, True), t10=rouge(hb, refs, 1.0, False),
                    t12=rouge(hb, refs, 1.2, False))
    print(f"GATE {label}: beam-hypothesis BLEU-4 {b4:.2f} = recorded {rec} (greedy hypotheses give {g4:.2f}, so these are the beam outputs)")

print("\nOfficial Communication test set; saved outputs, one run per model; ROUGE-L x100")
R = res["dwpose_ctc"]
print(f"clips scored per model: {R['n']} of 800; one saved run per model; beam hypotheses (the beam-5 outputs, see GATE lines)  [c16-038]")
print(f"ROUGE-L per sentence = F_beta of the longest common subsequence, averaged over the {R['n']} sentences, multiplied by 100  [c06-003]")
for k, v in res.items():
    print(f"  {v['label']:30s} detokenised beta=1.0: {v['d10']:.2f}  whitespace-tokenised beta=1.0: {v['t10']:.2f}  whitespace-tokenised beta=1.2: {v['t12']:.2f}")
print("beta used per column: 1.0 / 1.0 / 1.2  [c16-043]")
print(f"Baseline model outputs: ROUGE-L {R['d10']:.2f} detokenised and {R['t10']:.2f} whitespace-tokenised, both at beta = 1.0, on {R['n']} clips  [c12-057]")
print(f"Baseline model (DWPose + CTC): {R['d10']:.2f} / {R['t10']:.2f} / {R['t12']:.2f}  [c16-040]")
print(f"ROUGE-L of one set of outputs is {R['d10']:.2f} or {R['t10']:.2f}  [c14-081 c21-004]")
V = res["vlp"]
print(f"40-epoch pre-trained model: {V['d10']:.2f} / {V['t10']:.2f} / {V['t12']:.2f}  [c16-041]")
print(f"RGB-fusion model: {res['rgbfusion']['d10']:.2f} / {res['rgbfusion']['t10']:.2f} / {res['rgbfusion']['t12']:.2f}")
sw = {k: v["t10"] - v["d10"] for k, v in res.items()}
for k, v in res.items():
    print(f"  swing {v['label']}: {sw[k]:+.2f} points")
ok = sw["dwpose_ctc"] > 19
print(f"tokenisation alone moves the baseline outputs by {sw['dwpose_ctc']:.2f} points, more than 19: {ok}  [c03-017 c12-058]")
if not ok: sys.exit("DISAGREES: swing not above 19")

print("\nROUGE-L (beta 1.2) divided by BLEU-1")
for n, rl, b1 in PUBLISHED:
    print(f"  published {n}: {rl:.2f} / {b1:.2f} = {rl/b1:.3f}")
qs = []
for k, v in res.items():
    q = v["t12"] / v["b1"]; qs.append(q)
    print(f"  {v['label']}: ROUGE-L {v['t12']:.2f} / BLEU-1 {v['b1']:.2f} = {q:.3f}")
print(f"ratio for the three single models: {min(qs):.3f} to {max(qs):.3f}  [c18-017]")
