"""Score the blind tracker audit (render_audit.py panels, two raters each) against the hidden colour key.
Input: audit/labels_scored.json (key + both raters' raw labels). Consensus = both raters give the same label."""
import json
from collections import Counter
from pathlib import Path
L = json.load(open(Path(__file__).parent / "audit/labels_scored.json"))
assert len(L) == 160 and all(k in v for v in L.values() for k in ("r1", "r2", "m1", "m2", "stratum"))
print(f"raw inter-rater agreement {sum(v['r1'] == v['r2'] for v in L.values())}/160")
dec_t = dec_s = 0
for s in ("light", "medium", "heavy", "agree"):
    c = Counter(v["m1"] for v in L.values() if v["stratum"] == s and v["r1"] == v["r2"])
    t, i = c.get("tracked-only", 0), c.get("index0-only", 0); dec_t += t; dec_s += i
    print(f"{s:7s} consensus {sum(c.values())}/40: {dict(c)}" + (f" | tracker right in {t}/{t + i} decisive" if t + i else ""))
print(f"overall: tracker right in {dec_t}/{dec_t + dec_s} = {dec_t / (dec_t + dec_s):.1%} of decisive override frames")
