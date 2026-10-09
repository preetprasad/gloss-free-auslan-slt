"""Generate LLM paraphrases of the English training targets (multi-reference augmentation).

Auslan-Daily gives exactly one English reference per clip, so cross-entropy punishes every
valid alternative wording the decoder could produce. This prompts an instruct LLM for N
paraphrases of each unique training subtitle and writes a TSV consumed by
train_ctc.py --paraphrase_map (one `<original>\t<paraphrase>` line per paraphrase; the
trainer keeps only paraphrases whose tokens are already in the reference vocabulary).

Deliberately only ever reads a *training* manifest — paraphrasing val/test targets would
contaminate the evaluation.
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.data import read_manifest

INSTR = (
    "Rewrite the sentence in different words while keeping the exact same meaning. "
    "Use simple, everyday conversational English. Keep it about the same length. "
    "Do not add new information, do not answer the sentence, do not explain. "
    "Output ONLY the rewritten sentences, one per line.\n"
)
from restore_references import sentence as S   # dataset sentences are named by digest, not stored here

ICL = (
    "Examples:\n"
    f"Sentence: {S('0580869881cfa0555295bdd2be5956120250be5c')}\n"
    "Rewrites:\ni feel really hot .\ni am so hot .\nit feels very hot to me .\n\n"
    "Sentence: what is your name ?\n"
    "Rewrites:\nwhat do they call you ?\ncan you tell me your name ?\nwho are you ?\n\n"
)


def clean(line: str) -> str:
    line = line.strip()
    for prefix in ("Rewrites:", "Sentence:", "-", "*"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    if line[:2].rstrip(".").isdigit():          # strip "1." / "2)" list markers
        line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
    return " ".join(line.lower().split())


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="TRAIN manifest only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_paraphrases", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if os.path.basename(a.manifest).startswith(("val", "test")):
        raise SystemExit(f"refusing to paraphrase a non-train manifest: {a.manifest}")

    texts, seen = [], set()
    for _, t in read_manifest(a.manifest):
        k = " ".join(t.lower().split())
        if k and k not in seen:
            seen.add(k)
            texts.append(t)
    if a.limit:
        texts = texts[:a.limit]
    print(f"[para] {len(texts)} unique training sentences -> {a.num_paraphrases} paraphrases each",
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32
    ).to(a.device).eval()

    written = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for start in range(0, len(texts), a.batch_size):
            chunk = texts[start:start + a.batch_size]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": INSTR + ICL
                      + f"Sentence: {t}\nGive {a.num_paraphrases} rewrites.\nRewrites:"}],
                    tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(a.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=96, do_sample=True,
                                     temperature=0.9, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            for src, seq in zip(chunk, out):
                gen = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                kept, dedupe = [], {" ".join(src.lower().split())}
                for line in gen.splitlines():
                    c = clean(line)
                    if c and c not in dedupe and 2 <= len(c.split()) <= 40:
                        dedupe.add(c)
                        kept.append(c)
                    if len(kept) >= a.num_paraphrases:
                        break
                for c in kept:
                    fh.write(f"{src}\t{c}\n")
                    written += 1
            if (start // a.batch_size) % 20 == 0:
                print(f"[para] {start + len(chunk)}/{len(texts)} sentences, {written} paraphrases",
                      flush=True)
    print(f"[para] wrote {written} paraphrases to {a.out}", flush=True)


if __name__ == "__main__":
    main()
