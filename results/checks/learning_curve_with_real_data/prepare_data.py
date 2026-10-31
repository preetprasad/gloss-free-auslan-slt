"""One-off preparation (run once by hand, not by make): copies the compact saved outputs of job 21734 and derives
per-item numbers that need training text, so check.py never reads training text.
Run from src: python ../results/checks/learning_curve_with_real_data/prepare_data.py
"""
import json, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; W8 = ROOT / "results/real_data_learning_curve"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/tools"))
from slt.metrics import detokenize
from stitch_lexicon import STOP
from arm_readouts import content
ARMS, SEEDS = ["R025", "R050", "R100", "V050"], [42, 101, 202, 303, 404]
VID = re.compile(r"(video_\d+)_"); norm = lambda s: " ".join(detokenize(s).lower().split())
norm2 = lambda t: " ".join(t.lower().split())
rows = lambda p: [l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8") if "\t" in l]
full_train = {norm2(t) for _, t in rows(W8 / "train.tsv")}
tpaths = [p for p, _ in rows(W8 / "test.tsv")]
tvid = [VID.match(Path(p).name).group(1) for p in tpaths]
per_video = Counter(VID.match(Path(p).name).group(1) for p, _ in rows(W8 / "train.tsv"))
D = {"sizes": {}, "epochs": {}, "ref_counts": {}, "unseen": None, "q2": {}, "n_train": len(rows(W8 / "train.tsv"))}
for a in ARMS:
    for s in SEEDS:
        tag = f"{a}_s{s}"; d = W8 / "runs" / tag
        D["sizes"][tag] = len(rows(W8 / "manifests" / f"{tag}.tsv"))
        D["epochs"][tag] = json.load(open(d / "progress.json"))["epochs"]
        for sp in ("test", "val"):
            j = json.load(open(d / f"beam5_{sp}_best.json"))
            json.dump({"beam_hyps": j["beam_hyps"], "refs": j["refs"],
                       "recorded_bleu4": j["beam_result"]["metrics"]["tokenize=13a;detok=True"]}, open(HERE / "data" / f"{tag}_{sp}.json", "w"))
            if sp == "test":
                refs = j["refs"]
                cnt = Counter()
                for _, t in rows(W8 / "manifests" / f"{tag}.tsv"):
                    cnt.update(content(t, STOP))
                D["ref_counts"][tag] = [[cnt.get(w, 0) for w in content(r, STOP)] for r in refs]
                if D["unseen"] is None:
                    D["unseen"] = [norm2(r) not in full_train for r in refs]
for s in SEEDS:
    vp = [p for p, _ in rows(W8 / "manifests" / f"V050_s{s}.tsv")]
    kept_n = Counter(VID.match(Path(p).name).group(1) for p in vp)
    cut = [v for v, n in kept_n.items() if n < per_video[v]]
    assert len(cut) <= 1
    D["q2"][str(s)] = {"held": [v not in kept_n for v in tvid], "kept": [(v in kept_n and v not in cut) for v in tvid], "cut": cut}
json.dump(D, open(HERE / "derived.json", "w"))
