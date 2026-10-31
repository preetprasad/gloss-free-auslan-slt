"""Run once in the private tree (not by make): subsets of saved job outputs -> data/.
Per-run beam-5 hypotheses (+ references, stored once per split under "refs") from the run folders, byte copies of the small
summary / log files, and gloss counts of the adopted stitched corpus (audit_A.jsonl)."""
import json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
D = Path(__file__).resolve().parent / "data"
D.mkdir(exist_ok=True)

waves = {
    "wave10": (R / "wave10_49135/runs", [f"{a}_s{s}" for a in ("A", "S", "SC") for s in (42, 101, 202)]),
    "wave7": (R / "sign_density_stage1/runs", [f"{a}_s{s}" for a in ("A", "D") for s in (42, 101, 202)]),
    "wave7s2": (R / "sign_density_stage2/runs", [f"{a}_s{s}" for a in ("A", "D", "DL") for s in (303, 404)]),
    "wave6": (R / "wider_corpus_joint_training/runs", [f"{a}_s{s}" for a in ("A", "B") for s in (42, 101, 202, 303, 404)]
              + [f"C_s{s}" for s in (42, 101, 202)]),
}
refs = {}
for w, (base, runs) in waves.items():
    out = {}
    for r in runs:
        out[r] = {}
        for sp in ("val", "test"):
            j = json.load(open(base / r / f"beam5_{sp}_best.json"))
            assert refs.setdefault(sp, j["refs"]) == j["refs"], (w, r, sp)
            out[r][sp] = j["beam_hyps"]
    json.dump(out, open(D / f"hyps_{w}.json", "w"))
for sp, rf in refs.items():
    json.dump({"refs": rf}, open(D / f"refs_{sp}.json", "w"))

copies = {
    "wave10_49135/runs/summary_A_S_SC_R_s42.json": "wave10_summary_s42.json",
    "wave10_49135/runs/summary_A_S_SC_R_s101.json": "wave10_summary_s101.json",
    "wave10_49135/runs/summary_A_S_SC_R_s202.json": "wave10_summary_s202.json",
    "sign_density_stage1/summary_A_D_s42.json": "wave7_summary_s42.json",
    "sign_density_stage1/summary_A_D_s101.json": "wave7_summary_s101.json",
    "sign_density_stage1/summary_A_D_s202.json": "wave7_summary_s202.json",
    "sign_density_stage2/summary_s303.json": "wave7_summary_s303.json",
    "sign_density_stage2/summary_s404.json": "wave7_summary_s404.json",
    "sign_density_stage2/readout_output.txt": "wave7_readout_output.txt",
    "sign_density_stage2/logs/w7s2-prep-66643.out": "wave7_prep_66643.out",
    "wave12b_prereg/w12b_corpus_facts_output.txt": "wave12b_corpus_facts_output.txt",
    "wave12_build_2026-10-13/readout_logs/w12b-summary-138381.out": "wave12b_summary_138381.out",
}
for s in (42, 101, 202, 303, 404):
    copies[f"wider_corpus_joint_training/summary_s{s}.json"] = f"wave6_summary_s{s}.json"
for s in (42, 101, 202):
    copies[f"wave2_r2_5403/summary_s{s}.json"] = f"wave2_summary_s{s}.json"
for s in (303, 404):
    copies[f"letter_fix_and_longer_training_stage2/summary_s{s}.json"] = f"wave2_stage2_summary_s{s}.json"
for a, b in copies.items():
    shutil.copyfile(R / a, D / b)

g = {}
for l in open(R / "stitched_corpus_audit/audit_A.jsonl"):
    for s in json.loads(l)["segments"]:
        g[s["gloss"]] = g.get(s["gloss"], 0) + 1
json.dump(g, open(D / "adopted_corpus_gloss_counts.json", "w"), sort_keys=True)
