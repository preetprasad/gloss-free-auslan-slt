"""Run once by hand (not by make): copy the saved outputs this check reads into data/."""
import csv, json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = HERE.parents[1]  # results/
D = HERE / "data"
for s in (42, 101, 202, 303, 404):
    shutil.copy(R / f"video_disjoint_recipe_confirmation/summary_s{s}.json", D / f"summary_s{s}.json")
shutil.copy(R / "video_disjoint_recipe_confirmation/seed_decision_5seeds.txt", D / "seed_decision_5seeds.txt")
shutil.copy(R / "t2_vdens_10674/dumps_V6_test/hyps_LP=1.0.jsonl", D / "V6_test_LP1.0.jsonl")
shutil.copy(R / "video_disjoint_cross_modal_ensemble/prior/seedens_vdB/s101_test/hyps_LP=0.9.jsonl", D / "vdB_s101_test_LP0.9.jsonl")
# clip ids only (no text) of the three split manifests
with open(D / "vd_split_ids.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "clip"])
    for sp in ("train", "val", "test"):
        for line in open(R / f"video_disjoint_cross_modal_ensemble/manifests/vd_{sp}.tsv"):
            w.writerow([sp, Path(line.split("\t")[0]).stem])
# the one read-out line that names the six members of the ensemble V6
line = [l for l in open(R / "video_disjoint_cross_modal_ensemble/g1_readout.txt") if l.startswith("  V6 ")][0]
(D / "v6_members_line.txt").write_text(line)

# ---- added: single models chosen on validation
P = R / "video_disjoint_cross_modal_ensemble/prior/seedens_vdB"
for s, lp in (("42", "1.0"), ("101", "0.9"), ("202", "0.7")):
    shutil.copy(P / f"s{s}_val/hyps_LP={lp}.jsonl", D / f"vdB_s{s}_val_LP{lp}.jsonl")
for s, lp in (("42", "1.0"), ("202", "0.7")):
    shutil.copy(P / f"s{s}_test/hyps_LP={lp}.jsonl", D / f"vdB_s{s}_test_LP{lp}.jsonl")
rd = [l for l in open(R / "video_disjoint_cross_modal_ensemble/g1_readout.txt") if l.startswith("  vdB_s") and l.rstrip().endswith("single")]
(D / "g1_singles_lines.txt").write_text("".join(rd))
