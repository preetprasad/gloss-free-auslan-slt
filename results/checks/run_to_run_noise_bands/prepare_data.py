"""Run once (not by make): programmatic subset of saved outputs -> det_pairs.json.
Beam-5 test BLEU-4 of the baseline-with-augmentation control of wave 6 (summary_s*.json, key A) and of wave 16 (runs/A_s*/beam5_test_best.json)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
out = {}
for s in (42, 101, 202):
    w6 = json.load(open(R / f"wider_corpus_joint_training/summary_s{s}.json"))["A"]["beam5_test"]
    w16 = json.load(open(R / f"wave16_174916/runs/A_s{s}/beam5_test_best.json"))["beam_result"]["metrics"]["tokenize=13a;detok=True"]
    out[str(s)] = {"wave6_A": w6, "wave16_A": w16}
json.dump(out, open(Path(__file__).parent / "det_pairs.json", "w"), indent=1)
