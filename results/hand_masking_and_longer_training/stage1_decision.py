"""Wave 1 (job 1216919) Stage-1 futility decision, computed from the saved decode JSONs.

Pre-registered rule (results/PREREG_WAVE1_R2_2026-09-16.md, written before any number):
  Delta_s = beam-5 (LP 0.6) VALIDATION BLEU-4 of arm X's best.pt minus w1A's, at seed s,
            13a / detokenised, read from beam5_val_best.json.
  STOP X if mean(Delta_42, Delta_101, Delta_202) <= 0.  No test number enters this decision.
"""
import json, statistics as st
from pathlib import Path

SEEDS = [42, 101, 202]
ARMS = ["w1A", "w1H", "w1L60"]
KEY = "tokenize=13a;detok=True"

def bleu(arm, seed, split, ckpt):
    p = Path(f"runs/{arm}_s{seed}/beam5_{split}_{ckpt}.json")
    d = json.loads(p.read_text())
    assert d["beam"] == 5 and d["length_penalty"] == 0.6, p
    assert d["beam_result"]["n"] == 800, p
    if ckpt == "best":
        assert d["checkpoint"].endswith(f"{arm}_s{seed}/best.pt"), d["checkpoint"]
    if split == "val":
        assert d["manifest"].endswith("val.tsv"), d["manifest"]
    else:
        assert d["manifest"].endswith("test.tsv"), d["manifest"]
    return d["beam_result"]["metrics"][KEY]

def table(split, ckpt):
    rows = {a: [bleu(a, s, split, ckpt) for s in SEEDS] for a in ARMS}
    out = {}
    for a in ("w1H", "w1L60"):
        d = [rows[a][i] - rows["w1A"][i] for i in range(3)]
        out[a] = (d, st.mean(d), sum(x > 0 for x in d))
    return rows, out

print(__doc__)
for split, ckpt, label in [("val", "best", "*** PRE-REGISTERED CRITERION ***  beam-5 VALIDATION, best.pt"),
                           ("test", "best", "beam-5 test, best.pt (reported, NOT a criterion)"),
                           ("val", "soup_last5", "beam-5 validation, last-5 soup (secondary)"),
                           ("test", "soup_last5", "beam-5 test, last-5 soup (secondary)")]:
    rows, deltas = table(split, ckpt)
    print(f"\n{label}")
    for a in ARMS:
        print(f"  {a:6s} " + "  ".join(f"s{s}={v:6.3f}" for s, v in zip(SEEDS, rows[a])) + f"   mean={st.mean(rows[a]):6.3f}")
    for a, (d, m, npos) in deltas.items():
        ci = ""
        if len(d) > 1:
            sd = st.stdev(d); half = 4.303 * sd / len(d) ** 0.5   # t(0.975, df=2)
            ci = f"  95% t-interval [{m-half:+.3f}, {m+half:+.3f}]"
        print(f"  {a} - w1A: " + " ".join(f"{x:+6.3f}" for x in d) + f"  | mean {m:+.3f} | {npos}/3 positive{ci}")

print("\n" + "=" * 78)
print("STAGE-1 FUTILITY DECISION (validation only, best.pt):")
_, dec = table("val", "best")
for a, (d, m, _) in dec.items():
    print(f"  {a:6s} mean delta {m:+.4f}  ->  {'STOP (futile)' if m <= 0 else 'CONTINUE to Stage 2 (seeds 303/404)'}")
