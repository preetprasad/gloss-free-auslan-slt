"""PoseStitch v2 Stage A — LLM-generated NOVEL sentences for synthetic corpus construction.

The v1 stitcher re-realised Auslan-Daily's own sentences with different exemplar clips. The
2026 LLM-stitching ablation shows that exact configuration is the one that HURTS (clip/signer
reassignment without novel sentences: 20.04 vs 21.38 baseline on PHOENIX); the novel sentences
are the active ingredient (+2.92). So this generates new conversational sentences instead.

Sentences are anchored on real training sentences (register) and constrained to words that
(a) have a sign in the MM-WLAuslan dictionary, and (b) already exist in the Auslan-Daily
training vocabulary -- so the synthetic corpus never expands the output space and the
comparison against the control stays clean.

Reads the TRAIN manifest only.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stitch_lexicon import (build_lexicon, content_words, coverage,
                                  signable_phrases, STOP)

INSTR = (
    "You write short, natural, everyday conversational sentences, the kind spoken in a casual "
    "day-to-day chat (greetings, family, food, work, weather, feelings, plans).\n"
    "STRICT RULES:\n"
    "- Use ONLY words from the allowed word list, plus these function words: "
    "a an the is are am was were be to of in on at and or but i you he she it we they this that "
    "my your his her our their do does did not have has had will would can could my me.\n"
    "- One sentence per line, 4 to 12 words, lowercase, ending with a space then a full stop.\n"
    "- No names, no numbers, no explanations, no bullet points.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="MM-WLAuslan label JSONs")
    ap.add_argument("--ad_manifest", required=True, help="Auslan-Daily TRAIN manifest (anchors + vocab)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", type=int, default=12000, help="how many accepted sentences to keep")
    ap.add_argument("--min_cov", type=float, default=0.9)
    ap.add_argument("--min_signs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--per_prompt", type=int, default=10)
    ap.add_argument("--max_rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--constrained", action="store_true",
                    help="mask logits to a signable-phrase trie so out_of_vocab and "
                         "low_sign_coverage become STRUCTURALLY impossible (~13.8x yield)")
    ap.add_argument("--no_paren_alias", action="store_true",
                    help="do NOT index a gloss's parenthetical as a standalone alias -- it is "
                         "usually a sense disambiguator, so aliasing stitches wrong-sense signs")
    ap.add_argument("--full_dictionary", action="store_true",
                    help="let the generator use EVERY MM-WLAuslan surface form (plus inflected forms seen in "
                         "Auslan-Daily), not only words already in the Auslan-Daily training vocabulary")
    ap.add_argument("--morphology", action="store_true",
                    help="let inflected words match citation-form glosses (opt-in; see stitch_lexicon)")
    a = ap.parse_args()

    if os.path.basename(a.ad_manifest).startswith(("val", "test")):
        raise SystemExit(f"refusing to anchor on a non-train manifest: {a.ad_manifest}")
    random.seed(a.seed)

    lex = build_lexicon(a.labels, morphology=a.morphology, paren_alias=not a.no_paren_alias)
    real_sents, real_vocab = [], set()
    for ln in open(a.ad_manifest, encoding="utf-8"):
        if "\t" not in ln:
            continue
        t = ln.rstrip("\n").split("\t", 1)[1]
        real_sents.append(t)
        real_vocab.update(t.lower().split())
    real_keys = {" ".join(s.lower().split()) for s in real_sents}
    signable = signable_phrases(lex, real_vocab)
    allowed = real_vocab   # tokens a kept sentence may contain; widened below by --full_dictionary
    print(f"[gen] lexicon {len(lex.gloss2clips)} glosses / {len(lex.phrase2gloss)} surface forms; "
          f"{len(signable)} are signable AND in the Auslan-Daily vocab", flush=True)
    if a.full_dictionary:
        dict_words = {w for p in lex.phrase2gloss for w in p.split()}
        allowed = real_vocab | dict_words
        signable = signable_phrases(lex, allowed)
        print(f"[gen] FULL DICTIONARY: {len(signable)} signable phrases "
              f"({len(dict_words - real_vocab)} dictionary words are outside the Auslan-Daily vocab)", flush=True)

    import torch
    torch.manual_seed(a.seed)   # HF sampling draws from torch, so --seed must seed it too (also seeds CUDA)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32
    ).to(a.device).eval()

    trie = None
    if a.constrained:
        from tools.constrained_gen import PhraseTrie
        free = sorted(w for w in STOP if w in real_vocab)
        trie = PhraseTrie(tok, signable, free)
        print(f"[gen] CONSTRAINED decoding: trie over {trie.n_phrases} phrases "
              f"({len(signable)} signable + {len(free)} free function words); "
              f"out_of_vocab and low_sign_coverage are now structurally impossible", flush=True)

    kept, seen = [], set()
    rejected = collections.Counter()
    for rnd in range(a.max_rounds):
        if len(kept) >= a.target:
            break
        prompts = []
        for _ in range(a.batch_size):
            anchors = random.sample(real_sents, min(6, len(real_sents)))
            if trie is not None:
                # The mask already enforces the vocabulary, so the 90-word allowed-list is dead
                # weight in the prompt. Dropping it cuts the prefill (and the KV cache) sharply,
                # which is what lets many sequences share one prompt on a 16GB V100.
                msg = ("Write ONE short, natural, everyday conversational sentence, lowercase, "
                       "like these:\n" + "\n".join(anchors) + "\n\nSentence:\n")
            else:
                words = random.sample(signable, min(90, len(signable)))
                msg = (INSTR
                       + "\nAllowed word list: " + ", ".join(sorted(words))
                       + "\n\nExamples of the style to match:\n"
                       + "\n".join(anchors)
                       + f"\n\nWrite {a.per_prompt} new sentences:\n")
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(a.device)
        gkw = dict(do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
        if trie is not None:
            # Under constraint each sequence yields exactly ONE valid sentence, so ask for many
            # sequences off one shared prefill instead of many sentences inside one continuation.
            from transformers import LogitsProcessorList
            from tools.constrained_gen import build_logits_processor
            gkw.update(max_new_tokens=64, num_return_sequences=a.per_prompt,
                       logits_processor=LogitsProcessorList([build_logits_processor(
                           trie, enc["input_ids"].shape[1], eos_id=tok.eos_token_id,
                           min_words=4, max_words=14, min_signs=a.min_signs)]))
        else:
            gkw.update(max_new_tokens=220)
        with torch.no_grad():
            out = model.generate(**enc, **gkw)
        for seq in out:
            gen = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for line in gen.splitlines():
                s = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().lower()
                s = re.sub(r"\s+", " ", re.sub(r"[^a-z' .?]", " ", s)).strip()
                if not s:
                    continue
                if not s.endswith("."):
                    s = s.rstrip(" .") + " ."
                elif not s.endswith(" ."):
                    s = s[:-1].rstrip() + " ."
                key = " ".join(s.split())
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                if key in real_keys:
                    rejected["copies_real_sentence"] += 1
                    continue
                toks = key.split()
                if not 4 <= len(toks) <= 14:
                    rejected["length"] += 1
                    continue
                if any(t not in allowed for t in toks):
                    rejected["out_of_vocab"] += 1
                    continue
                cw = content_words(key)
                cov, hits = coverage(key, lex)
                if len(hits) < a.min_signs or not cw or cov < a.min_cov:
                    rejected["low_sign_coverage"] += 1
                    continue
                seen.add(key)
                kept.append(key)
        if rnd % 10 == 0:
            print(f"[gen] round {rnd}: kept {len(kept)}/{a.target} | rejects {dict(rejected)}",
                  flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept[:a.target]) + "\n")
    print(f"[gen] wrote {min(len(kept), a.target)} novel signable sentences to {a.out}", flush=True)
    print(f"[gen] rejection breakdown: {dict(rejected)}", flush=True)


if __name__ == "__main__":
    main()
