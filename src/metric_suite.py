"""Beyond-BLEU metric suite + sentence-disjoint generalization breakdown.

Reads a dump_hyps jsonl ({ref, hyp_beam, hyp_greedy}) and reports, on the beam field:
BLEU-4, BLEU-1, chrF, ROUGE-L. Also splits the test set by whether the reference appears
verbatim in the TRAIN targets (memorisation vs generalisation) and reports BLEU-4 on each —
the 22.7% train-test target overlap means the 'unseen' number is the honest generalisation score.
"""
from __future__ import annotations
import hashlib, argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slt.data import read_manifest
from slt.metrics import detokenize
import sacrebleu


ROUGE_BETA = 1.2   # SignJoey / GASLT convention (recall-weighted F-measure)


def rouge_l(h, r):
    hs, rs = h.split(), r.split()
    if not hs or not rs: return 0.0
    dp = [[0]*(len(rs)+1) for _ in range(len(hs)+1)]
    for i in range(1, len(hs)+1):
        for j in range(1, len(rs)+1):
            dp[i][j] = dp[i-1][j-1]+1 if hs[i-1]==rs[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[len(hs)][len(rs)]; p, rc = lcs/len(hs), lcs/len(rs)
    if p + rc == 0:
        return 0.0
    # SignJoey (signjoey/external_metrics/mscoco_rouge.py) and GASLT both use a RECALL-WEIGHTED
    # F-measure with beta = 1.2, not F1. beta = 1.0 over-scores a model that under-generates
    # (this one: length ratio 0.79-0.86) because it up-weights the stronger precision term.
    # Corrected 2026-09-24; the beta=1.0 values previously produced here run ~0.4-0.6 high.
    b2 = ROUGE_BETA ** 2
    return (1 + b2) * p * rc / (rc + b2 * p)


def corpus(hyps, refs, hyps_tok=None, refs_tok=None):
    """BLEU/chrF on DETOKENISED text (sacreBLEU 13a re-splits punctuation itself), but
    ROUGE-L on the WHITESPACE-TOKENISED text.

    BUG FIXED 2026-09-05: ROUGE-L was being computed on the detokenised strings too. Detokenising
    glues the sentence-final period onto the last word ("great ." -> "great."), so the final
    CONTENT word could never match its reference -- and Auslan-Daily references are short
    (mean 6.14 whitespace tokens) and 100% end in a separated " .", so that one corrupted token
    is a large share of the LCS. Measured on the saved beam hypotheses:
        dwpose_ctc  ROUGE-L 16.56 (detok, as previously reported) vs 36.22 (whitespace-tokenised)
        vlp         17.74 vs 36.68
        rgbfusion   17.84 vs 37.67
    The published Auslan-Daily comparators (Table 3, Communication/Pose) are SL-Luong 37.27,
    SL-Transformer 35.65, GASLT 35.74 on pre-tokenised references, i.e. the tokenised convention
    -- so the corrected numbers sit AT/ABOVE the published baselines where the old ones read as
    less than half of them. CAVEAT: their exact tokenisation is inferred from the pre-tokenised
    reference format and the matching value range at matched BLEU-4, not read off their code.
    Note punctuation-STRIPPED ROUGE-L is only ~19.7-21.1: most of the gap is the free period
    match, which is exactly why the convention must be stated explicitly when reporting.
    """
    from sacrebleu.metrics import BLEU, CHRF
    b4 = BLEU(tokenize="13a").corpus_score(hyps, [refs]).score
    b1 = BLEU(tokenize="13a", max_ngram_order=1).corpus_score(hyps, [refs]).score
    chrf = CHRF().corpus_score(hyps, [refs]).score
    ht = hyps_tok if hyps_tok is not None else hyps
    rt = refs_tok if refs_tok is not None else refs
    rl = 100.0 * sum(rouge_l(h, r) for h, r in zip(ht, rt)) / max(1, len(ht))
    return b4, b1, chrf, rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--field", default="hyp_beam")
    ap.add_argument("--train_manifest", default="")
    ap.add_argument("--train_hashes", default="",
                    help="SHA-1 digests of the training targets (tools/train_target_hashes.py); used for the "
                         "seen/unseen split when the training manifest is not on disk")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    raw_h = [(r.get(a.field) or "") for r in rows]
    raw_r = [r["ref"] for r in rows]
    hyps = [detokenize(h) for h in raw_h]        # BLEU/chrF: sacreBLEU re-splits punctuation
    refs = [detokenize(r) for r in raw_r]
    # ROUGE-L on the WHITESPACE-TOKENISED text -- see corpus() for why detokenising broke it
    b4, b1, chrf, rl = corpus(hyps, refs, raw_h, raw_r)
    print(f"\n===== {os.path.basename(a.inp)} ({a.field}, n={len(rows)}) =====")
    print(f"BLEU-4={b4:.2f}  BLEU-1={b1:.2f}  chrF={chrf:.2f}  ROUGE-L={rl:.2f}"
          "   [BLEU/chrF detokenised; ROUGE-L whitespace-tokenised]")

    in_train = None
    if a.train_manifest and os.path.exists(a.train_manifest):
        train_t = set(" ".join(detokenize(t).lower().split()) for _, t in read_manifest(a.train_manifest))
        in_train = train_t.__contains__
    elif a.train_hashes and os.path.exists(a.train_hashes):
        digests = set(open(a.train_hashes).read().split())
        in_train = lambda key: hashlib.sha1(key.encode("utf-8")).hexdigest() in digests
    if in_train is not None:
        seen_h, seen_r, uns_h, uns_r = [], [], [], []
        for h, r in zip(hyps, refs):
            key = " ".join(r.lower().split())
            (seen_h if in_train(key) else uns_h).append(h)
            (seen_r if in_train(key) else uns_r).append(r)
        if seen_h:
            sb = sacrebleu.corpus_bleu(seen_h, [seen_r], tokenize="13a").score
            print(f"  seen-in-train targets:   n={len(seen_h):3d}  BLEU-4={sb:.2f}")
        if uns_h:
            ub = sacrebleu.corpus_bleu(uns_h, [uns_r], tokenize="13a").score
            print(f"  UNSEEN targets (generalisation): n={len(uns_h):3d}  BLEU-4={ub:.2f}")


if __name__ == "__main__":
    main()
