"""Run once by hand: programmatic subset of the saved wave-23 job output (job set 179456).
Keeps per run and split the beam-5 hypotheses and references (key "ref") and copies progress.json and summary_w23.json byte for byte."""
import json, shutil
from pathlib import Path
SRC = Path("../../wave23_179456"); OUT = Path(".")
(OUT / "runs").mkdir(exist_ok=True)
for a in "CXY":
    for s in (42, 101, 202):
        d = SRC / "runs" / f"{a}_s{s}"
        shutil.copy(d / "progress.json", OUT / "runs" / f"{a}_s{s}_progress.json")
        for sp in ("val", "test"):
            j = json.load(open(d / f"beam5_{sp}_best.json"))
            assert len(j["beam_hyps"]) == len(j["refs"]) == 800
            with open(OUT / "runs" / f"{a}_s{s}_{sp}.jsonl", "w") as f:
                for h, r in zip(j["beam_hyps"], j["refs"]):
                    f.write(json.dumps({"hyp": h, "ref": r}) + "\n")
shutil.copy(SRC / "summary_w23.json", OUT / "summary_w23.json")

# --- earlier jobs with the same control recipe (local per-item outputs): wave 16 control A and wave 6 control A, official test, beam 5
# Keeps the beam-5 hypotheses and references (key "ref") and the score the job itself recorded for each run.
shutil.copy(SRC / "summary_179457.txt", OUT / "summary_179457.txt")   # byte copy of the read-out job's own output (gate line)
rec = {}
for tag, folder in (("wave16A", "wave16_174916"), ("wave6A", "wider_corpus_joint_training")):
    for s in (42, 101, 202):
        j = json.load(open(Path("../..") / folder / "runs" / f"A_s{s}" / "beam5_test_best.json"))
        assert len(j["beam_hyps"]) == len(j["refs"]) == 800
        rec[f"{tag}_s{s}"] = j["beam_result"]["metrics"]["tokenize=13a;detok=True"]
        with open(OUT / "runs" / f"earlier_{tag}_s{s}_test.jsonl", "w") as f:
            for h, r in zip(j["beam_hyps"], j["refs"]):
                f.write(json.dumps({"hyp": h, "ref": r}) + "\n")
json.dump(rec, open(OUT / "earlier_recorded.json", "w"), indent=1)
