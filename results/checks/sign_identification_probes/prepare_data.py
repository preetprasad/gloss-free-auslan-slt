"""Run once by hand (not by make): derives mmwl_gloss_counts.json from the MM-WLAuslan label files.
Only gloss-name counts are kept (no clips, no text)."""
import json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
d = ROOT / "data/raw/MM-WLAuslan/Annotation/Labels & Split"
c = collections.Counter()
for k in ("Train", "Valid"):
    for v in json.load(open(d / f"{k}.json")).values():
        c[v.strip().lower()] += 1
json.dump(dict(sorted(c.items())), open(Path(__file__).parent / "mmwl_gloss_counts.json", "w"), indent=0)
print(len(c), sum(c.values()))
