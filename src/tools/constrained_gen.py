"""Constrained decoding over a signable-phrase trie (PoseStitch v2 Stage A accelerator).

WHY: sentence generation is FILTER-bound, not sampler-bound. Measured on the real data, even
genuine human Auslan-Daily sentences pass our own filters only 9.34% of the time, and our LLM
already runs at 6.75% = 72% of that natural-text ceiling -- so prompt engineering is spent.
Conditionally, `out_of_vocab` removes 50.4% of survivors and `low_sign_coverage` then removes
85.5% of what is left.

Both filters can be made STRUCTURALLY IMPOSSIBLE instead of being applied after the fact:

  * mask the logits so a word boundary may only begin an ALLOWED phrase, where
        allowed = (signable phrases INTERSECT Auslan-Daily vocab)  UNION  (STOP words)
  * `out_of_vocab` then cannot occur, because every emitted word is in the vocabulary;
  * `low_sign_coverage` cannot occur either, and this is the load-bearing point: coverage is
    computed over CONTENT words only, and every non-STOP phrase we allow is signable by
    construction, so content-word coverage is identically 1.0.

That leaves only the length/duplicate/copies-a-real-sentence checks, which are cheap. The
existing post-hoc filters are deliberately kept in `gen_stitch_sentences.py` as ASSERTIONS --
under constraint they should reject ~0 on vocab/coverage, and any rejection means a bug here.

Subword pitfalls this handles explicitly (each one silently corrupts output if ignored):
  * PREFIX CLOSURE -- after emitting " croc" the model must be forced to finish "crocodile";
    otherwise it appends "odiles" and produces an out-of-vocabulary word.
  * TERMINAL-AND-PREFIX AMBIGUITY -- after " be", both "end the word" and "continue to beak"
    are legal. A single-pointer DFA silently drops one branch, so we track a SET of live nodes.
  * SENTENCE-INITIAL vs MID-SENTENCE -- "cat" and " cat" are different tokens, so the first
    word gets its own trie.
  * MULTI-WORD GLOSSES -- "thank you" is ONE sign, so phrases are trie paths spanning a space
    rather than independent words; this also keeps the sign count honest.
  * CANONICAL TOKENISATION ONLY -- we admit just the tokenizer's own encoding of each phrase.
    Permitting every token path that spells the word would push the model onto segmentations it
    never saw in pretraining, which invites repetition loops.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set


class PhraseTrie:
    """Trie over the CANONICAL token encodings of allowed phrases.

    Two roots: `root_mid` for mid-sentence phrases (encoded with a leading space) and
    `root_first` for the sentence-initial phrase (no leading space).
    Node = {"kids": {token_id: node}, "end": Optional[bool is_sign]}.
    """

    def __init__(self, tokenizer, sign_phrases: Sequence[str], stop_phrases: Sequence[str]):
        self.tok = tokenizer
        self.root_mid: Dict = {"kids": {}, "end": None}
        self.root_first: Dict = {"kids": {}, "end": None}
        self.n_phrases = 0
        seen = set()
        for phrase, is_sign in [(p, True) for p in sign_phrases] + [(p, False) for p in stop_phrases]:
            p = " ".join(str(phrase).split())
            if not p or p in seen:
                continue
            seen.add(p)
            self.n_phrases += 1
            self._add(self.root_mid, tokenizer.encode(" " + p, add_special_tokens=False), is_sign)
            self._add(self.root_first, tokenizer.encode(p, add_special_tokens=False), is_sign)
        # the terminator is its own path so it can be gated by the counters
        self.end_ids = tokenizer.encode(" .", add_special_tokens=False)

    @staticmethod
    def _add(root: Dict, ids: Sequence[int], is_sign: bool) -> None:
        if not ids:
            return
        node = root
        for t in ids:
            node = node["kids"].setdefault(t, {"kids": {}, "end": None})
        # a phrase already marked as a sign stays a sign (signable beats stop-word)
        node["end"] = bool(is_sign) or bool(node["end"])

    def starts(self, first: bool) -> Set[int]:
        return set((self.root_first if first else self.root_mid)["kids"].keys())

    def root(self, first: bool) -> Dict:
        return self.root_first if first else self.root_mid


class SignableConstraint:
    """Per-sequence state machine over a PhraseTrie.

    Exposes `allowed()` -> set of legal next token ids, and `advance(tok)`.
    Deliberately model-agnostic and pure-python so it can be unit-tested without a GPU or an LLM.
    """

    def __init__(self, trie: PhraseTrie, min_words=4, max_words=14, min_signs=3):
        self.t = trie
        self.min_words, self.max_words, self.min_signs = min_words, max_words, min_signs
        self.live: List[Dict] = []      # live trie nodes (empty => at a word boundary)
        self.first = True
        self.n_words = 0
        self.n_signs = 0
        self.end_pos: Optional[int] = None   # index into trie.end_ids while emitting " ."
        self.done = False

    def _pending(self):
        """The phrase under the cursor is complete but not yet committed (commit happens when the
        NEXT token arrives). Counting only committed phrases lets one extra phrase start past the
        limit -- an off-by-one that a randomised self-test caught as 15-word output at max 14."""
        if self.live and any(n["end"] is not None for n in self.live):
            return 1, int(any(n["end"] is True for n in self.live))
        return 0, 0

    def _can_finish(self) -> bool:
        pw, ps = self._pending()
        return (self.n_words + pw) >= self.min_words and (self.n_signs + ps) >= self.min_signs

    def _must_finish(self) -> bool:
        pw, _ = self._pending()
        return (self.n_words + pw) >= self.max_words

    def allowed(self) -> Set[int]:
        if self.done:
            return set()
        if self.end_pos is not None:                      # mid-terminator
            return {self.t.end_ids[self.end_pos]}
        if not self.live:                                 # at a word boundary
            out: Set[int] = set()
            if not self._must_finish():
                out |= self.t.starts(self.first)
            if self._can_finish():
                out.add(self.t.end_ids[0])
            # never return empty: if the counters box us in, allow the terminator anyway
            return out or {self.t.end_ids[0]}
        # inside a phrase: continuations from every live node, plus (if any live node is a
        # terminal) the option to start a NEW phrase or terminate
        out = set()
        terminal = any(n["end"] is not None for n in self.live)
        for n in self.live:
            out |= set(n["kids"].keys())
        if terminal:
            if not self._must_finish():
                out |= self.t.starts(False)
            if self._can_finish():
                out.add(self.t.end_ids[0])
        return out or {self.t.end_ids[0]}

    def advance(self, tok: int) -> None:
        if self.done:
            return
        if self.end_pos is not None:
            self.end_pos += 1
            if self.end_pos >= len(self.t.end_ids):
                self.done = True
            return
        if tok == self.t.end_ids[0] and (not self.live or any(n["end"] is not None for n in self.live)):
            # commit any completed phrase before terminating
            self._commit()
            self.end_pos = 1
            if self.end_pos >= len(self.t.end_ids):
                self.done = True
            return
        nxt = [n["kids"][tok] for n in self.live if tok in n["kids"]]
        if nxt:
            self.live = nxt
            return
        # token did not continue the current phrase -> it must START a new one
        self._commit()
        root = self.t.root(self.first)
        if tok in root["kids"]:
            self.live = [root["kids"][tok]]
            self.first = False
        else:
            self.live = []   # unreachable under masking; stay at a boundary rather than crash

    def _commit(self) -> None:
        """Bank the phrase currently under the cursor (longest live terminal wins)."""
        if not self.live:
            return
        sign = any(n["end"] is True for n in self.live)
        if any(n["end"] is not None for n in self.live):
            self.n_words += 1
            self.n_signs += int(sign)
        self.live = []
        self.first = False


def build_logits_processor(trie: PhraseTrie, prompt_len: int, eos_id: int, **kw):
    """HF LogitsProcessor enforcing the constraint. Imported lazily so this module stays
    importable (and unit-testable) without torch/transformers installed.

    `eos_id` is REQUIRED and is the fix for a real failure: once a sequence has finished its
    sentence, `SignableConstraint.allowed()` is empty by design, which masked every logit to
    -inf -> softmax NaN -> `CUDA error: device-side assert triggered` inside multinomial
    sampling. HF keeps sampling finished rows until the whole batch stops, so a completed row
    must always retain a legal token; we give it EOS.
    """
    import torch
    from transformers import LogitsProcessor

    class SignableMask(LogitsProcessor):
        def __init__(self):
            self.states: Dict[int, SignableConstraint] = {}
            self.seen: Dict[int, int] = {}

        def __call__(self, input_ids, scores):
            V = scores.shape[-1]
            for b in range(input_ids.shape[0]):
                st = self.states.get(b)
                if st is None:
                    st = self.states[b] = SignableConstraint(trie, **kw)
                    self.seen[b] = prompt_len
                row = input_ids[b].tolist()
                for t in row[self.seen[b]:]:
                    st.advance(int(t))
                self.seen[b] = len(row)
                ok = {t for t in st.allowed() if 0 <= t < V}
                if st.done or not ok:
                    ok = {eos_id}                      # finished row: only EOS stays legal
                mask = torch.ones(V, device=scores.device, dtype=torch.bool)
                mask[torch.tensor(sorted(ok), device=scores.device, dtype=torch.long)] = False
                scores[b] = scores[b].masked_fill(mask, float("-inf"))
            return scores

    return SignableMask()
