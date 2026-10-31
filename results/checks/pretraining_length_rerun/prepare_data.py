"""Subset of the saved wave-11 job outputs (cluster job 116619 pre-training arms, 116620 News arms).
Keeps only scores and run settings (no hypotheses, no reference text). Run once by the author, not by make."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
W = ROOT / "results" / "validation_selected_pretraining_rerun"
KEY = "tokenize=13a;detok=True"
out = {}
for kind, arms in (("pre", ["P0", "P40", "P80", "P128", "P256"]), ("news", ["nF256", "nF512"])):
    for a in arms:
        for s in (42, 101, 202):
            d = W / f"runs_{kind}" / f"{a}_s{s}"
            p = json.load(open(d / "progress.json"))
            bv = json.load(open(d / "beam5_val_best.json")); bt = json.load(open(d / "beam5_test_best.json"))
            rec = dict(kind=kind, arm=a, seed=s, epochs=p["epochs"], best_epoch=p["best"]["epoch"],
                       greedy_val=p["best"]["bleu4"],
                       beam5_val=bv["beam_result"]["metrics"][KEY], beam5_test=bt["beam_result"]["metrics"][KEY],
                       n_val=bv["beam_result"]["n"], n_test=bt["beam_result"]["n"],
                       beam=bv["beam"], done=(W / f"runs_{kind}" / f"DONE_{a}_s{s}").exists())
            if kind == "pre" and a != "P0":
                rec["pretrain_epochs"] = json.load(open(W / "runs_pre" / f"vlp_e{a[1:]}_s{s}" / "progress.json"))["epochs"]
            out[f"{a}_s{s}"] = rec
json.dump(out, open(Path(__file__).parent / "runs.json", "w"), indent=1)
print(len(out), "runs")
