"""One-off subset builder (not run by make): joins each saved beam-5 decode with its saved self-CE / self-reliance
vectors (tools/self_reliance.py output), keeping only id, hypothesis, reference and the two scores."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for split, hyps, selfj in (("test", "results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl", "results/self_reliance_ctc.json"),
                           ("val", "results/hyps_val_dwpose_ctc.jsonl", "results/self_reliance_val.json")):
    recs = [json.loads(l) for l in open(ROOT / hyps)]
    d = json.load(open(ROOT / selfj))
    assert len(recs) == len(d["self_ce"]) == len(d["self_reliance"]) == 800
    with open(HERE / f"single_{split}.jsonl", "w") as f:
        for r, ce, rel in zip(recs, d["self_ce"], d["self_reliance"]):
            f.write(json.dumps({"id": r["idx"], "hyp": r["hyp_beam"], "ref": r["ref"], "self_ce": ce, "self_rel": rel}) + "\n")
