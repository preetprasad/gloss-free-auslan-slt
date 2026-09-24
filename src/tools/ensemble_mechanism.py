"""t3-ensnovel: is the ensembling gain token-level partial correctness (novel sentences) or member selection?

Inputs: the ensemble's hyps_LP=<lp>.jsonl and one hyps file per member, all from tools/ensemble_mbr_wer.py on the
same manifest. Pre-registration: results/ENSEMBLE_MECHANISM_2026-09-22.md. Decisive statistic, paired within clip:
on clips whose ensemble output matches no member, the share where the ensemble's sentence chrF beats EVERY member
versus the share where it is below EVERY member.
"""
import argparse, itertools, json, statistics, sys
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402


def load(p):
    rows = [json.loads(l) for l in open(p)]
    return [r["ref"] for r in rows], [r["hyp_beam"] for r in rows]


def chrf(h, r):
    return sacrebleu.sentence_chrf(detokenize(h), [detokenize(r)]).score


def bleu(hyps, refs):
    return corpus_bleu4([detokenize(x) for x in hyps], [detokenize(x) for x in refs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--expect", type=float, required=True)
    a = ap.parse_args()
    refs, E = load(a.ensemble)
    M = []
    for p in a.members:
        r, h = load(p)
        assert r == refs, f"{p}: references differ from the ensemble dump"
        M.append(h)
    n, k = len(refs), len(M)
    eb = bleu(E, refs)
    print(f"gate: ensemble BLEU-4 {eb:.3f} (recorded {a.expect}) | members " + " ".join(f"{bleu(h, refs):.2f}" for h in M))
    if abs(eb - a.expect) > 0.005:
        sys.exit("HARNESS GATE FAILED")

    cE = [chrf(E[i], refs[i]) for i in range(n)]
    cM = [[chrf(M[j][i], refs[i]) for i in range(n)] for j in range(k)]
    novel = [i for i in range(n) if E[i] not in {M[j][i] for j in range(k)}]
    matched = [i for i in range(n) if i not in set(novel)]

    def split(idx, tag):
        above = sum(cE[i] > max(cM[j][i] for j in range(k)) for i in idx)
        below = sum(cE[i] < min(cM[j][i] for j in range(k)) for i in idx)
        print(f"{tag:8s} clips {len(idx):3d} | ensemble beats EVERY member {above} ({100*above/max(1,len(idx)):.1f}%) | "
              f"below EVERY member {below} ({100*below/max(1,len(idx)):.1f}%) | other {len(idx)-above-below}")
        return above, below
    print(f"novel ensemble outputs: {len(novel)}/{n} = {100*len(novel)/n:.1f}%")
    a_nov, b_nov = split(novel, "novel")
    split(matched, "matched")
    split(list(range(n)), "all")
    print(f"DECISIVE (novel clips): beats-every {a_nov} vs below-every {b_nov} -> "
          f"{'SUPPORTS token-level partial correctness' if a_nov > b_nov else 'does NOT support it'}")
    sub = lambda idx, hs: bleu([hs[i] for i in idx], [refs[i] for i in idx])
    print(f"(not decisive) corpus BLEU on novel clips: ensemble {sub(novel, E):.2f} vs best member there "
          f"{max(sub(novel, h) for h in M):.2f} | matched clips: ensemble {sub(matched, E):.2f}")

    best_single = max(bleu(h, refs) for h in M)
    print("member oracle by pool size (per-sentence best chrF; mean corpus BLEU over all subsets):")
    oracle6 = None
    for size in range(1, k + 1):
        vals = []
        for S in itertools.combinations(range(k), size):
            pick = [max(S, key=lambda j: cM[j][i]) for i in range(n)]
            vals.append(bleu([M[pick[i]][i] for i in range(n)], refs))
        print(f"  pool {size}: {statistics.mean(vals):6.2f} ({len(vals)} subsets)")
        oracle6 = statistics.mean(vals)
    print(f"capture of the oracle gap: (ensemble {eb:.2f} - best single {best_single:.2f}) / (oracle {oracle6:.2f} - "
          f"best single) = {100*(eb-best_single)/(oracle6-best_single):.1f}%")
    # EXPLORATORY (not pre-registered): where does the gain over the best single member come from?
    b = max(range(k), key=lambda j: bleu(M[j], refs)); nov = set(novel)
    only_novel = [E[i] if i in nov else M[b][i] for i in range(n)]
    only_matched = [M[b][i] if i in nov else E[i] for i in range(n)]
    print(f"EXPLORATORY decomposition from the best member ({bleu(M[b], refs):.2f}): ensemble output on NOVEL clips only "
          f"{bleu(only_novel, refs):.2f} | on MATCHED clips only {bleu(only_matched, refs):.2f} | both {eb:.2f}")
    agree = [sum(E[i] == M[j][i] for j in range(k)) for i in matched]
    print(f"EXPLORATORY matched clips: ensemble output shared by a median of {statistics.median(agree)} of {k} members; "
          f"best member's output differs from the ensemble on {sum(E[i] != M[b][i] for i in matched)} of them")
    same = [100 * sum(M[x][i] == M[y][i] for i in range(n)) / n for x, y in itertools.combinations(range(k), 2)]
    print(f"member pairs with identical output: mean {statistics.mean(same):.1f}% (range {min(same):.1f}-{max(same):.1f})")


if __name__ == "__main__":
    main()
