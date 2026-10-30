"""PGG-SLT Stage A: generate LLM pseudo-glosses from the English target sentences.

For each training subtitle, prompt an instruct LLM (Qwen2.5) with a few in-context examples to
produce an ordered sequence of sign-gloss tokens (uppercase content words, lemmatised, function
words dropped). Writes a TSV: <normalised-text>\t<GLOSS G2 G3 ...>, consumed by
train_ctc.py --gloss_map (which supervises a CTC head on these glosses).
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slt.data import read_manifest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from restore_references import sentence as S   # dataset sentences are named by digest, not stored here

ICL = (
    "Examples:\n"
    f"English: {S('0580869881cfa0555295bdd2be5956120250be5c')}  ->  GLOSS: ME FEEL HOT\n"
    "English: what is your name ?  ->  GLOSS: YOUR NAME WHAT\n"
    "English: i do not want to go .  ->  GLOSS: ME WANT GO NOT\n"
    f"English: {S('a790fa2af87942705aec0d6694bd9a83b909a8b8')}  ->  GLOSS: THANK-YOU\n"
)
INSTR = ("Convert the English sentence into sign-language gloss: an ordered, space-separated list "
         "of UPPERCASE content words (nouns, verbs, adjectives, adverbs, question words, negation), "
         "lemmatised (base form), dropping function words (a, the, is, to, of, ...). "
         "Output ONLY the glosses.\n")


def clean(txt):
    line = (txt or "").strip().splitlines()[0] if (txt or "").strip() else ""
    line = line.replace("GLOSS:", "").strip()
    toks = [t.strip(".,?!;:").upper() for t in line.split() if t.strip(".,?!;:")]
    toks = [t for t in toks if any(c.isalpha() for c in t)]
    return toks[:16]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    texts = []
    seen = set()
    for _, t in read_manifest(a.manifest):
        k = " ".join(t.lower().split())
        if k and k not in seen:
            seen.add(k); texts.append(t)
    if a.limit:
        texts = texts[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32).to(a.device).eval()

    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for txt in texts:
            msgs = [{"role": "user", "content": INSTR + ICL + f"English: {txt}  ->  GLOSS:"}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(a.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
            out = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            g = clean(out) or ["<g>"]
            fh.write(" ".join(txt.lower().split()) + "\t" + " ".join(g) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush(); print(f"[pgg-gen] {n}/{len(texts)}  e.g. {txt!r} -> {g}", flush=True)
    print(f"[pgg-gen] wrote {n} pseudo-glosses -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
