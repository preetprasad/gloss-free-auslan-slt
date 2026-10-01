"""Wave 10 arm SC: CTC targets that follow the SHUFFLED sign order (results/PREREG_WAVE10_SIGNORDER_2026-10-01.md).

train_ctc.py's CTC target for a sentence is to_gloss(text): its whitespace tokens that are alphabetic and not in
train_ctc.STOP, in English order, adjacent repeats collapsed. For each synthetic clip of corpus S this writes the SAME
content words re-ordered to follow the laid-out signs (read from S's audit sidecar: english_position per segment), as a
--gloss_map TSV (text<TAB>tokens). Each content word goes with the sign whose span covers it; a word no sign covers
stays attached to the nearest preceding covered word's sign (or the next one if none precedes).

Self-checks (all must hold or it exits non-zero):
  * the stitcher's spans recomputed from the text match the audit (count and gloss at every position);
  * in English order the construction reproduces train_ctc.to_gloss(text) exactly, for every sentence;
  * the shuffled target holds the same words as the English one (multiset, before collapsing repeats);
  * no synthetic text collides with a real train/validation text (the map is keyed by normalised text; CTC targets are
    only computed for training rows, so a test match cannot matter).
An audit without english_position (an unshuffled build) is treated as the identity order; the map then equals to_gloss.

python tools/w10_ctc_map.py --audit S/audit.jsonl --labels ... --npz_dirs ... --trainer_dir DIR \
    --real train.tsv val.tsv test.tsv --out ctc_map_S.tsv
"""
import argparse, importlib.util, json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("stitch_lexicon", os.path.join(HERE, "stitch_lexicon.py"))
SL = sys.modules["stitch_lexicon"] = importlib.util.module_from_spec(_spec)   # this snapshot's lexicon, by path
_spec.loader.exec_module(SL)


def collapse(toks):
    out = []
    for w in toks:
        if not out or out[-1] != w:
            out.append(w)
    return out


def targets(text, spans, laid, tc):
    """(English-order target, shuffled-order target, raw English list, raw shuffled list)."""
    rw = SL.words(text)
    owner = {i: e for e, (s, t, _) in enumerate(spans) for i in range(s, t)}
    covered = sorted(owner)
    content, ptr = [], 0                        # (token, regex word index) for to_gloss's content tokens
    for tok in tc.tokenize(text, "whitespace"):
        n = len(re.findall(r"[a-z']+", tok))
        if tok.isalpha() and tok not in tc.STOP:
            assert n == 1 and rw[ptr] == tok, (text, tok)
            content.append((tok, ptr))
        ptr += n
    assert ptr == len(rw), text

    def sign_of(i):
        if i in owner:
            return owner[i]
        prev = [j for j in covered if j < i]
        return owner[prev[-1]] if prev else owner[covered[0]]
    eng = [w for w, _ in content]
    shuf = [w for w, i in sorted(content, key=lambda c: (laid[sign_of(c[1])], c[1]))]
    return collapse(eng), collapse(shuf), eng, shuf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True); ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--npz_dirs", nargs="+", required=True); ap.add_argument("--trainer_dir", required=True)
    ap.add_argument("--real", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.append(os.path.abspath(a.trainer_dir))
    import train_ctc as tc
    assert not tc.GLOSS_MAP
    lex = SL.build_lexicon(a.labels, a.npz_dirs, morphology=True, paren_alias=False, letter_fix=False)
    real_keys = {tc._normkey(l.rstrip("\n").split("\t", 1)[1]) for m in a.real for l in open(m) if "\t" in l}
    rows, changed, keys = [], 0, set()
    for line in open(a.audit):
        r = json.loads(line)
        text, segs = r["text"], r["segments"]
        _, spans = SL.coverage(text, lex)
        assert len(spans) == len(segs), f"span count {len(spans)} != {len(segs)}: {text}"
        pos_of = [s.get("english_position", p) for p, s in enumerate(segs)]
        assert sorted(pos_of) == list(range(len(segs))), text
        laid = [None] * len(segs)
        for p, e in enumerate(pos_of):
            assert spans[e][2] == segs[p]["gloss"], (text, e, spans[e][2], segs[p]["gloss"])
            laid[e] = p
        eng, shuf, raw_e, raw_s = targets(text, spans, laid, tc)
        assert eng == tc.to_gloss(text) or (not eng and tc.to_gloss(text) == ["<g>"]), (text, eng, tc.to_gloss(text))
        assert Counter(raw_e) == Counter(raw_s), text
        k = tc._normkey(text)
        assert k not in real_keys, f"synthetic text also in a real manifest: {text}"
        assert k not in keys, f"duplicate synthetic text: {text}"
        keys.add(k)
        changed += shuf != eng
        if shuf:
            rows.append(f"{text}\t{' '.join(shuf)}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"CTC MAP OK: {len(keys)} sentences, {len(rows)} map rows, target order changed in {changed} "
          f"({changed / max(len(keys), 1):.1%}); English-order reconstruction == to_gloss on all; no real-text collisions")


if __name__ == "__main__":
    main()
