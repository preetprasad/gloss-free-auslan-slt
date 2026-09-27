"""Final ground-truth labels from the labelling workflow (PREREG_TRACKER_V2 §2): a panel's label is the two raters'
answer when they agree, else the answer the third rater shares with one of them (2 of 3), else "unclear".
Usage: build_labels.py WORKFLOW_RESULT.json OUT_LABELS.json"""
import json, sys
from collections import Counter
res = json.load(open(sys.argv[1])); r1, r2, adj = res["r1"], res["r2"], res["adj"]
ids = sorted(set(r1) | set(r2)); out = {}; how = Counter()
for p in ids:
    a, b, c = r1.get(p), r2.get(p), adj.get(p)
    if a is not None and a == b:
        out[p] = a; how["agree"] += 1
    elif c is not None and c in (a, b):
        out[p] = c; how["2-of-3"] += 1
    else:
        out[p] = "unclear"; how["unresolved"] += 1
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"{len(out)} panels | {dict(how)} | label mix {dict(Counter(out.values()))}")
