"""How novel and how varied is the stitched synthetic corpus, and how much VISUAL variety stands behind it?

Answers two questions (2026-09-22): (1) are the synthetic sentences new relative to Auslan-Daily, and how wide is
their variety; (2) when more sentences are generated, does the visual evidence per word grow, or do the same
MM-WLAuslan recordings get re-used? Inputs are copied into results/stitched_corpus_audit/ (syn_13760.tsv = the adopted
1.1:1 corpus text, syn_20608.tsv = the full pool, audit_A.jsonl = per-segment provenance of the adopted corpus,
which reproduces it array-for-array). Usage: python tools/stitch_corpus_audit.py [selftest]
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES, A = ROOT / "results", ROOT / "results" / "stitched_corpus_audit"


def sents(path):
    return [l.rstrip("\n").split("\t")[1] for l in open(path) if "\t" in l]


def words(s):
    return [w for w in s.split() if w not in {".", ","}]


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def profile(S, real_train=None):
    """Size, uniqueness, vocabulary, length, distinct-n, and n-gram novelty against real train."""
    W = [words(s) for s in S]
    out = {"sentences": len(S), "unique": len(set(S)), "vocab": len({w for ws in W for w in ws}),
           "mean_words": round(statistics.mean(len(ws) for ws in W), 2)}
    for n in (1, 2, 3):
        g = [x for ws in W for x in ngrams(ws, n)]
        out[f"distinct_{n}"] = round(len(set(g)) / len(g), 4)
    start = Counter(tuple(ws[:2]) for ws in W if len(ws) >= 2)
    out["top10_opening_bigrams_share"] = round(sum(c for _, c in start.most_common(10)) / len(W), 4)
    if real_train is not None:
        RW = [words(s) for s in real_train]
        for n in (2, 3):
            real = {x for ws in RW for x in ngrams(ws, n)}
            g = [x for ws in W for x in ngrams(ws, n)]
            out[f"{n}grams_unseen_in_real_train"] = round(sum(x not in real for x in g) / len(g), 4)
        rv = {w for ws in RW for w in ws}
        sv = {w for ws in W for w in ws}
        out["words_not_in_real_train_vocab"] = len(sv - rv)
        out["real_train_vocab_covered"] = f"{len(sv & rv)} of {len(rv)}"
    return out


def main():
    tr = sents(RES / "_kaya_train_manifest.tsv")
    va, te = sents(RES / "_local_val_manifest.tsv"), sents(RES / "TEST_manifest_official.tsv")
    s13, s20 = sents(A / "syn_13760.tsv"), sents(A / "syn_20608.tsv")
    rep = {"real_train": profile(tr), "syn_13760_adopted": profile(s13, tr), "syn_20608_pool": profile(s20, tr)}
    for name, S in (("syn_13760_adopted", s13), ("syn_20608_pool", s20)):
        u = set(S)
        rep[name]["exact_in_real_train"] = len(u & set(tr))
        rep[name]["exact_in_val"] = len(u & set(va))
        rep[name]["exact_in_test"] = len(u & set(te))
        rep[name]["exact_in_test_not_in_train"] = len(u & (set(te) - set(tr)))
    # visual supply behind the adopted corpus
    lab = {**json.load(open(ROOT / "mmwl_pose" / "Train.json")), **json.load(open(ROOT / "mmwl_pose" / "Valid.json"))}
    per_gloss = Counter(str(g).lower() for g in lab.values())
    segs = [s for l in open(A / "audit_A.jsonl") for s in json.loads(l)["segments"]]
    used_ex = Counter(s["source_id"] for s in segs)
    used_gl = Counter(s["gloss"] for s in segs)
    ex_per_used_gloss = Counter()
    for s in segs:
        ex_per_used_gloss[s["gloss"]] = ex_per_used_gloss[s["gloss"]]            # keep order
    ex_sets = {}
    for s in segs:
        ex_sets.setdefault(s["gloss"], set()).add(s["source_id"])
    uses = sorted(used_ex.values())
    rep["visual_supply_adopted"] = {
        "mmwl_clips": len(lab), "mmwl_glosses": len(per_gloss),
        "mmwl_mean_exemplars_per_gloss": round(len(lab) / len(per_gloss), 2),
        "stitched_sign_segments": len(segs), "distinct_glosses_used": len(used_gl),
        "distinct_recordings_used": len(used_ex),
        "segments_per_recording": {"mean": round(statistics.mean(uses), 2), "median": statistics.median(uses),
                                   "p90": uses[int(0.9 * len(uses))], "max": uses[-1]},
        "distinct_recordings_per_used_gloss_median": statistics.median(len(v) for v in ex_sets.values()),
        "top10_glosses_share_of_segments": round(sum(c for _, c in used_gl.most_common(10)) / len(segs), 4),
        "top10_glosses": used_gl.most_common(10),
    }
    json.dump(rep, open(A / "stitch_corpus_audit.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


def selftest():
    p = profile(["a b c .", "a b c .", "a d ."], real_train=["a b ."])
    assert p["sentences"] == 3 and p["unique"] == 2 and p["vocab"] == 4 and p["mean_words"] == round(8 / 3, 2)
    assert p["distinct_1"] == round(4 / 8, 4) and p["distinct_2"] == round(3 / 5, 4)
    assert p["2grams_unseen_in_real_train"] == round(3 / 5, 4) and p["words_not_in_real_train_vocab"] == 2
    assert words("hi , there .") == ["hi", "there"] and ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    print("selftest passed")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
