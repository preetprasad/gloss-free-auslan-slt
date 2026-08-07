"""Delegate semantic-adequacy grading to a local Ollama model (softer than BLEU):
for a sample of clips, ask llama3.2:3b how well each model's output conveys the
reference meaning (0 unrelated .. 3 fully correct). Confirms whether the low BLEU
hides partial semantic success or the outputs are genuinely content-empty.
"""
import csv, json, os, random, urllib.request, collections

OUT = os.path.expanduser("~/Honours/results/per_clip_quality_analysis")
MODEL = "llama3.2:3b"
URL = "http://localhost:11434/api/generate"
rows = list(csv.DictReader(open(f"{OUT}/per_clip.csv")))
random.Random(0).shuffle(rows)
sample = rows[:200]

def grade(ref, pred):
    prompt = (
        "You are grading a sign-language-to-English translation.\n"
        f"Correct meaning (reference): \"{ref}\"\n"
        f"Model output: \"{pred}\"\n"
        "How well does the model output convey the reference meaning?\n"
        "0 = unrelated, 1 = slight overlap, 2 = mostly correct, 3 = fully correct.\n"
        "Answer with ONLY the single digit 0, 1, 2, or 3."
    )
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 3}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())["response"]
    except Exception as e:
        return None
    for ch in r:
        if ch in "0123":
            return int(ch)
    return None

res = {"control": [], "transfer": []}
for i, row in enumerate(sample):
    for name in ("control", "transfer"):
        g = grade(row["ref"], row[name + "_pred"])
        if g is not None:
            res[name].append(g)
    if (i + 1) % 25 == 0:
        print(f"graded {i+1}/{len(sample)} clips", flush=True)

summary = {"model": MODEL, "n_clips": len(sample)}
for name in ("control", "transfer"):
    v = res[name]
    dist = collections.Counter(v)
    summary[name] = {
        "mean_adequacy": round(sum(v) / max(1, len(v)), 3),
        "pct_score0_unrelated": round(dist[0] / max(1, len(v)), 3),
        "pct_score>=2_mostly_correct": round((dist[2] + dist[3]) / max(1, len(v)), 3),
        "dist_0_1_2_3": [dist[0], dist[1], dist[2], dist[3]],
    }
json.dump(summary, open(f"{OUT}/ollama_adequacy.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
