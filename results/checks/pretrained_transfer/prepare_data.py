"""Run once by hand (not by make): cut the hypotheses out of the saved job outputs. References are NOT copied (they come from make data)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
S1 = ROOT / "results/wave12nb_167035/runs"
S2 = ROOT / "results/wave12nb_175088/runs_s2"
for arm in ("ctrl_small", "u_scratch", "u_asl"):
    for s in (42, 101, 202, 303, 404):
        d = (S1 if s < 300 else S2) / f"{arm}_s{s}"
        for sp in ("val", "test"):
            j = json.load(open(d / f"beam5_{sp}_best.json"))
            out = dict(arm=arm, seed=s, split=sp, beam=j["beam"], length_penalty=j["length_penalty"],
                       recorded_bleu4=j["beam_result"]["metrics"]["tokenize=13a;detok=True"],
                       recorded_unique=j["beam_result"]["unique_hypotheses"], hyps=j["beam_hyps"])
            json.dump(out, open(HERE / "data" / f"{arm}_s{s}_{sp}.json", "w"))
e = json.load(open(ROOT / "results/wave26_195361/E_star.json"))
json.dump(dict(beam=e["beam"], length_penalty=e["length_penalty"], n=e["n"], members=len(e["ckpts"]),
               recorded_bleu4=e["bleu4"], hyps=e["hyps"]), open(HERE / "data" / "ensemble_Estar_test.json", "w"))
p = json.load(open(ROOT / "results/wave12nb_167035/gate/h2s_test_port.json"))
json.dump({k: p[k] for k in ("n", "beams", "bleu4_13a", "released_bleu4_13a_same_clips", "target", "tol")},
          open(HERE / "data" / "gate_how2sign_port.json", "w"), indent=1)
