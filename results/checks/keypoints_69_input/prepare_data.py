"""Builds data/ from the private result folders (run once in the private tree; the public repository ships data/ itself).

Keeps only model hypotheses (no reference sentences), the BLEU-4 values the original jobs recorded (for the gate), the job-set
marker of each run, and the few small summary numbers the check reads. Paths are relative to the repository root.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
KEY = "tokenize=13a;detok=True"


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj))
    assert p.stat().st_size < 300_000, (name, p.stat().st_size)


def runs(folder, arms, seeds, tag):
    for arm in arms:
        for s in seeds:
            d = R / folder / "runs" / f"{arm}_s{s}"
            done = (d / "DONE").read_text().split()
            assert done[1] == arm and int(done[2]) == s
            out = {"arm": arm, "seed": s, "job_set": done[0], "beam": 5, "lp": 0.6}
            for sp in ("val", "test"):
                j = json.load(open(d / f"beam5_{sp}_best.json"))
                assert j["beam"] == 5 and j["length_penalty"] == 0.6 and len(j["beam_hyps"]) == 800
                out[sp] = j["beam_hyps"]
                out[f"recorded_{sp}"] = j["beam_result"]["metrics"][KEY]
            dump(f"hyps_{arm}_s{s}_{done[0]}.json", out)


def sweeps(folder, arms, seeds):
    for arm in arms:
        for s in seeds:
            d = R / folder / "sweep" / f"{arm}_s{s}" / "val_lp.json"
            j = json.load(open(d))
            assert j["beam"] == 5 and j["gate"]["ok"] and j["n"] == 800
            dump(f"sweep_val_{arm}_s{s}_{folder.split('_')[1]}.json",
                 {"arm": arm, "seed": s, "lps": j["lps"], "hyps": j["hyps"], "recorded": j["bleu"]})


S3 = (42, 101, 202)
runs("wave13_168265", ("ctrl", "C133", "F69"), S3, "A")
runs("wave13_168266", ("ctrlB", "F69B", "P69TC", "SG"), S3, "B")
runs("wave13x_172447", ("ctrl", "F69"), (303, 404), "S2")
sweeps("wave13_168265", ("ctrl", "F69"), S3)
sweeps("wave13_168266", ("ctrlB", "F69B"), S3)

# wave 24: validation hypotheses only
for pose in ("c0", "v2"):
    for arm in ("ctrl", "F69"):
        for s in S3:
            j = json.load(open(R / "wave24_179416" / "dumps" / f"runs_{pose}" / f"{arm}_s{s}" / "beam5_val_best.json"))
            assert j["beam"] == 5 and len(j["beam_hyps"]) == 800
            dump(f"trackpose_{pose}_{arm}_s{s}.json", {"pose": pose, "arm": arm, "seed": s, "val": j["beam_hyps"],
                                                  "recorded_val": j["beam_result"]["metrics"][KEY]})
shutil_gate = json.load(open(R / "wave24_179416" / "pose_gate.json"))
dump("trackpose_pose_gate.json", {k: shutil_gate[k] for k in ("n", "clips_scale_differs_gt10pct", "body_scale_ratio_v2_over_c0_median")})

# parameter counts of the arms (smoke test of the training script) and input widths (first lines of each training log)
sm = json.load(open(R / "wave13_168266" / "smoke.json"))["arms"]
dump("params.json", {a: sm[a]["params_M"] for a in sm})
fd = {}
for folder, pat in (("wave13_168265", "w13-train-168265_*.out"), ("wave13_168266", "w13-train-168266_*.out")):
    for lf in sorted((R / folder / "logs").glob(pat)):
        txt = lf.read_text()
        m = re.search(r"group . task \d+ -> arm (\S+) seed (\d+)", txt)
        f = re.search(r"feat_dim=(\d+)", txt)
        fd[f"{m.group(1)}_s{m.group(2)}"] = int(f.group(1))
dump("feat_dim.json", fd)
print(sorted(fd.items()))
