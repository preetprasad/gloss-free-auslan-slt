"""Run once by hand: subset results/semantic_regrade_10288/{s6,rgbfusion}.graded_hf.csv to idx + judge grade
(no sentences) -> adequacy_grades.json. Source: saved per-clip output of job 10288."""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
out = {}
for k in ("s6", "rgbfusion"):
    rows = sorted(csv.DictReader(open(ROOT / "results/semantic_regrade_10288" / f"{k}.graded_hf.csv", encoding="utf-8")), key=lambda r: int(r["idx"]))
    out[k] = [int(r["adequacy"]) for r in rows]
json.dump(out, open(Path(__file__).parent / "adequacy_grades.json", "w"))
