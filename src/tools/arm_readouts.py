"""Protocol readouts for one decoded arm (R2 rule 7), from a saved eval_beam --output_json dump.

BLEU-4, WER, hypothesis/reference length ratio, distinct outputs, pooled content-word precision and recall
(stop list = stitch_lexicon.STOP), and BLEU on the unseen-target subset when a train manifest is given
("seen" = the reference occurs verbatim, lowercased and whitespace-collapsed, among training targets --
the same definition as tools/paired_bootstrap.py).

  python tools/arm_readouts.py RUN_DIR [RUN_DIR ...] [--split test] [--ckpt best] [--train_manifest F]
  python tools/arm_readouts.py selftest
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slt.metrics import corpus_bleu4, corpus_wer, detokenize  # noqa: E402

KEY = "tokenize=13a;detok=True"


def content(text, stop):
    return [w for w in text.lower().split() if w.isalpha() and w not in stop]


def readouts(hyps, refs, stop, train=None):
    H, R = [detokenize(h) for h in hyps], [detokenize(r) for r in refs]
    hit = nh = nr = 0
    for h, r in zip(hyps, refs):
        ch, cr = content(h, stop), content(r, stop)
        hit += sum((Counter(ch) & Counter(cr)).values()); nh += len(ch); nr += len(cr)
    out = {"n": len(R), "bleu4": corpus_bleu4(H, R, "13a"), "wer": corpus_wer(H, R),
           "len_ratio": sum(len(h.split()) for h in H) / max(sum(len(r.split()) for r in R), 1),
           "distinct": len(set(H)),
           "content_precision": 100 * hit / max(nh, 1), "content_recall": 100 * hit / max(nr, 1)}
    if train is not None:
        norm = lambda t: " ".join(t.lower().split())
        un = [i for i, r in enumerate(refs) if norm(r) not in train]
        out["unseen_n"] = len(un)
        out["bleu4_unseen"] = corpus_bleu4([H[i] for i in un], [R[i] for i in un], "13a")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--train_manifest", default="")
    a = ap.parse_args()
    train = None
    if a.train_manifest:
        train = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split())
                 for l in open(a.train_manifest, encoding="utf-8") if "\t" in l}
    from stitch_lexicon import STOP
    for d in a.run_dirs:
        p = os.path.join(d, f"beam5_{a.split}_{a.ckpt}.json")
        if not os.path.exists(p):
            print(f"{os.path.basename(d)}: NO DUMP ({p})"); continue
        j = json.load(open(p))
        r = readouts(j["beam_hyps"], j["refs"], STOP, train)
        assert abs(r["bleu4"] - j["beam_result"]["metrics"][KEY]) < 1e-6, \
            f"{d}: recomputed BLEU {r['bleu4']} != recorded {j['beam_result']['metrics'][KEY]}"
        print(os.path.basename(d), json.dumps(r))


def selftest():
    stop = {"the", "is"}
    hyps = ["the cat sat .", "dog runs ."]
    refs = ["the cat sat .", "a bird flies ."]
    r = readouts(hyps, refs, stop, train={"the cat sat ."})
    assert r["n"] == 2 and r["distinct"] == 2 and r["unseen_n"] == 1, r
    assert abs(r["content_precision"] - 100 * 2 / 4) < 1e-9, r          # cat,sat hit of cat,sat,dog,runs
    assert abs(r["content_recall"] - 100 * 2 / 5) < 1e-9, r             # of cat,sat,a?,bird,flies -> a is alpha, not stop
    assert r["wer"] > 0 and 0 < r["len_ratio"] < 2, r
    print("selftest passed", json.dumps(r))


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
