"""Semantic-adequacy grading of SLT outputs with a local LLM (Ollama).

BLEU-4 requires n-gram overlap and punishes valid paraphrases (e.g. "the man is deaf" vs
"he is a deaf person" score poorly on BLEU but are semantically adequate). This grades each
(reference, hypothesis) pair on a 0-4 adequacy scale with a local model, so we can report how
much of the pipeline's real translation quality the BLEU number understates.

Runs against ANY Ollama endpoint (default the local Mac one). No Kaya / no GPU cluster needed:

  # on the Mac (Ollama running):
  python grade_semantic.py --in hyps_rgbfusion.jsonl --field hyp_beam --model qwen2.5:3b
  # -> writes hyps_rgbfusion.graded.csv + prints the summary

Env: OLLAMA_HOST (default http://localhost:11434).
"""
from __future__ import annotations
import argparse, csv, json, os, re
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

RUBRIC = (
    "You are grading a sign-language-to-English translation for MEANING only (ignore grammar, "
    "word order, and exact wording).\n"
    "Reference (correct meaning): \"{ref}\"\n"
    "Model output: \"{hyp}\"\n"
    "Score how much of the reference meaning the model output conveys, 0 to 4:\n"
    "0 = unrelated or empty; 1 = a shared topic word only; 2 = partial meaning; "
    "3 = most of the meaning, minor errors; 4 = fully adequate (a correct paraphrase counts as 4).\n"
    "Answer with ONLY the single integer 0, 1, 2, 3, or 4."
)


def ollama(model, prompt, timeout=120):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 4}}).encode()
    req = urllib.request.Request(HOST + "/api/generate", body,
                                {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def parse_score(txt):
    m = re.search(r"[0-4]", txt or "")
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="hyps jsonl from dump_hyps.py")
    ap.add_argument("--field", default="hyp_beam", help="hyp_beam | hyp_greedy")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.inp)[0] + ".graded.csv")

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    graded, scores, adequate = [], [], 0
    for i, r in enumerate(rows):
        ref, hyp = r["ref"], r.get(a.field, "") or ""
        if not hyp.strip():
            s = 0
        else:
            try:
                s = parse_score(ollama(a.model, RUBRIC.format(ref=ref, hyp=hyp)))
            except Exception as e:
                s = None
                print(f"[warn] idx {r.get('idx', i)} grading failed: {e}", flush=True)
        graded.append({"idx": r.get("idx", i), "ref": ref, "hyp": hyp, "adequacy": s})
        if s is not None:
            scores.append(s); adequate += int(s >= 3)
        if (i + 1) % 50 == 0:
            print(f"[grade] {i+1}/{len(rows)} mean={sum(scores)/max(1,len(scores)):.2f}", flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "ref", "hyp", "adequacy"])
        w.writeheader(); w.writerows(graded)

    n = len(scores)
    dist = {k: scores.count(k) for k in range(5)}
    print("\n===== SEMANTIC ADEQUACY (0-4) =====")
    print(f"model={a.model} field={a.field} graded={n}/{len(rows)}")
    if n:
        print(f"mean adequacy        : {sum(scores)/n:.3f}")
        print(f"% adequate (>=3)     : {100*adequate/n:.1f}%")
        print(f"% partial+ (>=2)     : {100*sum(s>=2 for s in scores)/n:.1f}%")
        print(f"distribution 0..4    : {dist}")
    print(f"per-clip -> {out}")


if __name__ == "__main__":
    main()
