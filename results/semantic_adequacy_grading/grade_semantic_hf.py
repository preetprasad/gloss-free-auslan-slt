"""Semantic-adequacy grading with a local HuggingFace instruct model (Kaya-runnable).

Same 0-4 adequacy rubric as grade_semantic.py, but uses a local transformers model as the
judge instead of an Ollama endpoint — so it runs on Kaya (CPU work node) with no external
service. Reads the dump_hyps jsonl, writes <name>.graded_hf.csv, prints the summary.

  python grade_semantic_hf.py --in hyps_rgbfusion.jsonl --field hyp_beam \
      --model Qwen/Qwen2.5-3B-Instruct --device cpu
"""
from __future__ import annotations
import argparse, csv, json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUBRIC = (
    "You are grading a sign-language-to-English translation for MEANING only (ignore grammar, "
    "word order, exact wording).\n"
    "Reference (correct meaning): \"{ref}\"\n"
    "Model output: \"{hyp}\"\n"
    "Score how much of the reference meaning the model output conveys, 0 to 4:\n"
    "0 unrelated/empty; 1 shared topic word only; 2 partial meaning; 3 most meaning, minor "
    "errors; 4 fully adequate (a correct paraphrase counts as 4).\n"
    "Answer with ONLY the single integer 0, 1, 2, 3, or 4."
)


def parse_score(txt):
    m = re.search(r"[0-4]", txt or "")
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--field", default="hyp_beam")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.inp)[0] + ".graded_hf.csv")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32 if a.device == "cpu" else torch.float16).to(a.device).eval()

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    graded, scores, adequate = [], [], 0
    for i, r in enumerate(rows):
        ref, hyp = r["ref"], (r.get(a.field) or "")
        if not hyp.strip():
            s = 0
        else:
            msgs = [{"role": "user", "content": RUBRIC.format(ref=ref, hyp=hyp)}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(a.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            s = parse_score(txt)
        graded.append({"idx": r.get("idx", i), "ref": ref, "hyp": hyp, "adequacy": s})
        if s is not None:
            scores.append(s); adequate += int(s >= 3)
        if (i + 1) % 50 == 0:
            print(f"[grade] {i+1}/{len(rows)} mean={sum(scores)/max(1,len(scores)):.2f}", flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "ref", "hyp", "adequacy"])
        w.writeheader(); w.writerows(graded)
    n = len(scores); dist = {k: scores.count(k) for k in range(5)}
    print("\n===== SEMANTIC ADEQUACY (0-4) =====", flush=True)
    print(f"file={a.inp} model={a.model} field={a.field} graded={n}/{len(rows)}", flush=True)
    if n:
        print(f"mean={sum(scores)/n:.3f} | %adequate(>=3)={100*adequate/n:.1f}% | "
              f"%partial+(>=2)={100*sum(s>=2 for s in scores)/n:.1f}% | dist={dist}", flush=True)
    print(f"per-clip -> {out}", flush=True)


if __name__ == "__main__":
    main()
