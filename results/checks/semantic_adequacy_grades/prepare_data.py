"""One-off: reduce the saved per-clip grade CSVs to grades.jsonl (system, idx, adequacy); no text is kept."""
import csv, json
from pathlib import Path
S = Path(__file__).resolve().parents[2] / "semantic_regrade_10288"
with open(Path(__file__).parent / "grades.jsonl", "w") as f:
    for n in ["dwpose_ctc", "vlp", "rgbfusion", "s6"]:
        for r in csv.DictReader(open(S / f"{n}.graded_hf.csv")):
            f.write(json.dumps({"system": n, "idx": int(r["idx"]), "adequacy": int(r["adequacy"])}) + "\n")
