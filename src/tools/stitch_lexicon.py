"""Shared MM-WLAuslan sign lexicon + phrase-aware sentence matching (PoseStitch v2).

v1 looked signs up with a single-word test (`word in gloss_set`), which structurally could not
match the 612 of 3,215 glosses that are multi-word or parenthesised ("TURN ON (START)",
"BROTHER (OLDER)") -- 19% of the dictionary was unreachable. This module indexes every gloss by
its full phrase AND its parenthetical alias, then matches greedily longest-first over n-grams,
so those signs are usable.

Used by both gen_stitch_sentences.py (Stage A) and build_posestitch_corpus_v2.py (Stage B) so
the two stages agree exactly on what counts as "signable".
"""
from __future__ import annotations
import collections, json, os, re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

STOP = set("a an the is are am was were be been being to of in on at and or but i you he she it we "
           "they this that these those my your his her its our their do does did not have has had will "
           "would can could should may might must shall there here then than so as if for with from up "
           "out about into over after by me".split())
MAX_NGRAM = 3


def words(t: str) -> List[str]:
    return re.findall(r"[a-z']+", t.lower())


def content_words(t: str) -> List[str]:
    return [w for w in words(t) if w not in STOP]


# ── morphology (OPT-IN via build_lexicon(morphology=True)) ──────────────────
# MM-WLAuslan glosses are citation forms (APPLE, ASK, ARRANGE), so exact matching calls
# "apples"/"asked"/"arranging" unsignable. That is a false negative, not a real constraint:
# Auslan does not inflect for English plural or tense -- one sign realises the paradigm, and
# the grammatical information is carried by non-manuals/context. The English TARGET text is
# left untouched ("wants" stays "wants"), so the model still has to learn the inflection from
# context -- which is exactly the real Auslan->English task, not false supervision.
#
# DEFAULT OFF on purpose: the queued/completed stitch jobs import this module at runtime, and
# flipping the default would silently change a corpus mid-experiment. Enable per-run.
IRREGULAR = {
    "went": "go", "gone": "go", "goes": "go", "going": "go",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "said": "say", "made": "make", "took": "take", "taken": "take",
    "gave": "give", "given": "give", "came": "come", "got": "get", "gotten": "get",
    "saw": "see", "seen": "see", "knew": "know", "known": "know",
    "thought": "think", "told": "tell", "found": "find", "felt": "feel",
    "bought": "buy", "brought": "bring", "ate": "eat", "eaten": "eat",
    "drank": "drink", "drunk": "drink", "ran": "run", "wrote": "write", "written": "write",
    "paid": "pay", "met": "meet", "lost": "lose", "sold": "sell", "sent": "send",
    "slept": "sleep", "spoke": "speak", "spoken": "speak", "stood": "stand",
    "understood": "understand", "wore": "wear", "worn": "wear", "won": "win",
}
# -est/-ly are safe-ish (smallest->small, sadly->sad). -er is NOT: it conflates comparatives
# ("faster"->fast, right) with AGENTIVE nouns ("farmer"->farm, "painter"->paint, "researcher"->
# research), which in Auslan are VERB+PERSON -- two signs, not one. Without POS tagging the two
# cannot be told apart, and false supervision costs more than missed coverage, so -er is OFF.
MORPH_COMPARATIVE_ER = False
MORPH_AGGRESSIVE = True


def _suffix_lemmas(w: str) -> List[str]:
    """Candidate citation forms for an inflected word. Over-generates deliberately -- a
    candidate is only ever used if it is actually present in the gloss lexicon."""
    out = []
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")                       # babies -> baby
    if w.endswith(("ses", "xes", "zes", "ches", "shes")) and len(w) > 4:
        out.append(w[:-2])                             # boxes -> box, dishes -> dish
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        out.append(w[:-1])                             # apples -> apple ("bees" -> "bee", NOT "be")
    if w.endswith("ied") and len(w) > 4:
        out.append(w[:-3] + "y")                       # tried -> try
    if w.endswith("ed") and len(w) > 3:
        out.append(w[:-2])                             # asked -> ask
        out.append(w[:-1])                             # liked -> like
        if len(w) > 4 and w[-3] == w[-4]:
            out.append(w[:-3])                         # stopped -> stop
    if w.endswith("ing") and len(w) > 4:
        out.append(w[:-3])                             # asking -> ask
        out.append(w[:-3] + "e")                       # making -> make
        if len(w) > 5 and w[-4] == w[-5]:
            out.append(w[:-4])                         # running -> run
    if MORPH_AGGRESSIVE:
        if w.endswith("est") and len(w) > 4:
            out += [w[:-3], w[:-2]]
        if MORPH_COMPARATIVE_ER and w.endswith("er") and len(w) > 4:
            out += [w[:-2], w[:-1]]
        if w.endswith("ly") and len(w) > 4:
            out.append(w[:-2])
    return out


def lemma_candidates(w: str) -> List[str]:
    if w in IRREGULAR:
        return [IRREGULAR[w]]
    return _suffix_lemmas(w)


@dataclass
class Lexicon:
    gloss2clips: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    phrase2gloss: Dict[str, str] = field(default_factory=dict)
    words: set = field(default_factory=set)
    morphology: bool = False


def _surface_forms(gloss: str, paren_alias: bool = True) -> List[str]:
    """Matchable phrases for a gloss label, e.g. 'TURN ON (START)' -> ['turn on', 'start'].

    AUDIT 2026-09-05: indexing the parenthetical as a standalone alias is often WRONG. In
    MM-WLAuslan the bracket is usually a SENSE DISAMBIGUATOR, not a synonym, so aliasing it
    stitches a semantically wrong sign under a correct English word -- measured over the real
    train corpus, alias matches are 4.8% of all matched sign tokens (2,423), and the frequent
    ones are bad: "at" -> PERSON (AT) 326x, "watch" -> WIND UP (WATCH) 233x, "water" -> WAVE
    (WATER) 141x, "door" -> OPEN (DOOR), "drink" -> FINISH (DRINK). A minority are fine or even
    better than the base ("do not" -> DO (DO NOT), "ok" -> FINE (OK)), so this is a flag, not a
    deletion -- but base-only is the safer default for new corpora.
    """
    g = gloss.strip().lower()
    forms = []
    base = re.sub(r"\([^)]*\)", " ", g)
    cands = [base] + (re.findall(r"\(([^)]*)\)", g) if paren_alias else [])
    for cand in cands:
        cand = " ".join(re.findall(r"[a-z']+", cand))
        if cand and len(cand.split()) <= MAX_NGRAM:
            forms.append(cand)
    return forms


# ── fingerspelling-letter fix (OPT-IN via build_lexicon(letter_fix=True)) ──────────────
# MM-WLAuslan ships the FINGERSPELLING ALPHABET as 24 single-character glosses (A-Z minus O
# and Q, 12 clips each). Lowercased, those become the phrase keys 'a', 'i', 's', ... so the
# English pronoun "I" is rendered as the handshape for the LETTER I and the article "a" as the
# LETTER A. Neither is the Auslan sign: first person is an index point at the chest (the
# dictionary carries it as ME), and English articles have no Auslan equivalent at all.
#
# MEASURED on the corpus that was actually trained (stitch audit, job 1216905): single-letter
# glosses are 8,187 of 94,428 segments in the 1.1:1 build = 8.67%, of which 'i' is 5,323
# (5.6%) and 'a' is 2,172 (2.3%). stitch-scram (2026-09-10e) showed the stitch gain comes from
# showing the CORRECT sign for each word, so those segments are active mis-supervision.
#
# letter_fix=True drops every single-character surface form from the phrase index and aliases
# "i" to the ME gloss when the dictionary has one. DEFAULT OFF, like morphology above: the
# corpora already trained on must keep rebuilding byte-identically.
LETTER_PRONOUN_ALIAS = {"i": "me"}

# ── wrong-sense glosses (OPT-IN via build_posestitch_corpus_v2.py --drop_wrong_sense; wave 12b) ──────────
# With paren_alias=False a gloss is indexed by its label MINUS the bracket, and in these 14 labels the bracket negates
# the base or names another sense. So the English word "good" is rendered as GOOD (NO GOOD) (Dictionary.json keywords:
# "unsatisfactory"), "be" as BE (NOT BE) ("have none"), and the modal "can" as CAN (TIN). MM-WLAuslan has no
# positive-sense gloss for nine of the 14 base forms (be, believe, can, enough, good, know, responsible, want, yet).
# MEASURED on the adopted 13,760-clip corpus (audit sidecar of its array-identical rebuild, wave-4 corpus A): 3,257 of
# 94,428 rendered signs = 3.45%, in 2,890 clips; 11 of the 14 glosses occur. The other four negated glosses of the
# dictionary (clever, like, my, your) never win their base form, because a plain gloss comes first, so they are not here.
# This constant changes nothing by itself: build_lexicon and match_signs do not read it.
WRONG_SENSE_GLOSSES = frozenset({
    "be (not be)", "before that (not before that)", "believe (not believe)", "bother (no bother)", "can (tin)",
    "enough (not enough)", "good (no good)", "know (not know)", "matter (not matter)", "mind (not mind)",
    "prior to (not prior to)", "responsible (not responsible)", "want (not want)", "yet (not yet)",
})


def build_lexicon(label_paths, npz_dirs=None, morphology: bool = False,
                  paren_alias: bool = True, letter_fix: bool = False) -> Lexicon:
    """label_paths: MM-WLAuslan {clip_id: GLOSS} JSONs. npz_dirs (optional, parallel list):
    only keep clips whose .npz actually exists in the matching directory.
    morphology: allow inflected sentence words to match citation-form glosses (see above).
    letter_fix: drop fingerspelled single letters from the phrase index and alias "i" -> ME."""
    gloss2clips = collections.defaultdict(list)
    for i, p in enumerate(label_paths):
        d = npz_dirs[i] if npz_dirs else None
        have = None
        if d:
            have = {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}
        for cid, g in json.load(open(p)).items():
            cid = str(cid)
            if have is not None and cid not in have:
                continue
            gloss2clips[str(g).strip().lower()].append((d or "", cid))
    phrase2gloss, vocab = {}, set()
    for g in gloss2clips:
        for form in _surface_forms(g, paren_alias):
            if letter_fix and len(form) == 1:
                continue                               # fingerspelled letter, not an English word
            # first gloss wins; keeps the mapping deterministic
            phrase2gloss.setdefault(form, g)
            vocab.update(form.split())
    if letter_fix:
        for word, gloss in LETTER_PRONOUN_ALIAS.items():
            if gloss in gloss2clips:
                phrase2gloss[word] = gloss
                vocab.add(word)
    return Lexicon(dict(gloss2clips), phrase2gloss, vocab, morphology)


def _lookup(lex: Lexicon, toks: List[str]) -> str | None:
    """Exact phrase lookup, with an optional citation-form fallback.

    Exact always wins, so enabling morphology can only ADD matches, never change existing ones.
    """
    g = lex.phrase2gloss.get(" ".join(toks))
    if g or not lex.morphology:
        return g
    # try replacing each token in turn with a citation-form candidate (cheap: n <= 3)
    for i, w in enumerate(toks):
        for cand in lemma_candidates(w):
            g = lex.phrase2gloss.get(" ".join(toks[:i] + [cand] + toks[i + 1:]))
            if g:
                return g
    return None


def match_signs(text: str, lex: Lexicon) -> List[Tuple[int, int, str]]:
    """Greedy longest-first n-gram match over the sentence's words.

    Returns (start, end, gloss) spans in word order -- the sign sequence to stitch.
    """
    toks = words(text)
    spans, i = [], 0
    while i < len(toks):
        hit = None
        for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
            g = _lookup(lex, toks[i:i + n])
            if g:
                hit = (i, i + n, g)
                break
        if hit:
            spans.append(hit)
            i = hit[1]
        else:
            i += 1
    return spans


def signable_phrases(lex: Lexicon, vocab: set) -> List[str]:
    """Phrases the generator may use: matchable by `match_signs` AND already in `vocab`.

    Uses phrase2gloss keys (not lex.words) so a word that only ever appears inside a multi-word
    gloss ("you" from "thank you") is never offered as if it were signable alone. When morphology
    is on, inflected forms present in the vocabulary are offered too -- otherwise the generator
    would only ever be shown citation forms and could not write natural English.
    """
    out = {p for p in lex.phrase2gloss if all(w in vocab for w in p.split())}
    if lex.morphology:
        out |= {w for w in vocab
                if w not in out and any(c in lex.phrase2gloss for c in lemma_candidates(w))}
    return sorted(out)


def coverage(text: str, lex: Lexicon) -> Tuple[float, List[Tuple[int, int, str]]]:
    """Fraction of CONTENT words covered by matched signs, and the matched spans.

    Function words inside a matched phrase still count as covered (they are part of the sign).
    """
    toks = words(text)
    cw_idx = {i for i, w in enumerate(toks) if w not in STOP}
    if not cw_idx:
        return 0.0, []
    spans = match_signs(text, lex)
    covered = {i for s, e, _ in spans for i in range(s, e)}
    return len(cw_idx & covered) / len(cw_idx), spans
