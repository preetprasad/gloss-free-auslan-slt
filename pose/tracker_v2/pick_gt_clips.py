"""Pick the ground-truth clips for tracker v2: 20 heavy (>0.3), 20 medium (0.1-0.3], 20 light (0-0.1] clips from the
official val+test splits (bins by frac_chosen_nonzero, results/ndet_per_clip.csv), random.Random(1) -- a different seed
from the v1 audit (seed 0) -- then split each bin 10/10 into DEV (to design on) and HELD-OUT (to score once)."""
import csv, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FRAC = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(open(ROOT / "results/ndet_per_clip.csv"))}
clips = [l.split("\t")[0].rsplit("/", 1)[1][:-4] for s in ("val", "test") for l in open(ROOT / f"results/real_data_learning_curve/{s}.tsv")]
audit = {v["clip"] for v in json.load(open(ROOT / "results/switch_video_89_193/audit/key.json")).values()}
rng = random.Random(1); out = {"dev": [], "heldout": []}
for name, f in (("heavy", lambda x: x > 0.3), ("medium", lambda x: 0.1 < x <= 0.3), ("light", lambda x: 0 < x <= 0.1)):
    pool = sorted(c for c in clips if f(FRAC[c]) and c not in audit and c != "video_89_193")
    pick = rng.sample(pool, 20)
    out["dev"] += [(c, name) for c in pick[:10]]; out["heldout"] += [(c, name) for c in pick[10:]]
json.dump(out, open(ROOT / "results/tracker_v2/gt_clips.json", "w"), indent=1)
print({k: len(v) for k, v in out.items()})
