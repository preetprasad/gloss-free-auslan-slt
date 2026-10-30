"""t1-enssel: selective prediction on an ENSEMBLE, with thresholds fixed on VALIDATION and carried to test.

Every selective-prediction number so far came from one pose-only single model, and its thresholds were
quantiles of the TEST distribution (descriptive only). This reads the ensemble n-best dumps written by
tools/ensemble_mbr_wer.py --dump_dir, where each candidate carries raw = the summed weighted-mean log-prob over
every generated token including EOS. Confidence of the selected hypothesis = raw / (words + 1) = -self-CE.

Harness gate (exits non-zero): re-selecting with raw / words**LP must reproduce the dumped hyps_LP=<lp>.jsonl
row-for-row and the recorded test BLEU-4; seen-in-train counts must match the recorded 230 (test) / 251 (val).
"""
from __future__ import annotations

import hashlib, argparse, json, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.metrics import corpus_bleu4, corpus_wer, detokenize          # noqa: E402
from tools.selective_prediction import curve                           # noqa: E402

COVS = (1.0, 0.9, 0.75, 0.5, 0.25)          # 25% is never quoted on unseen (val/test did not replicate there, 09-09y)


def load(dump, lp):
    rows = [json.loads(l) for l in open(Path(dump) / "nbest.jsonl")]
    refs, hyps, conf = [], [], []
    for r in rows:
        raw, h = max(r["nbest"], key=lambda c: c[0] / (max(1, len(c[1].split())) ** lp))
        refs.append(r["ref"]); hyps.append(h); conf.append(raw / (len(h.split()) + 1))
    return refs, hyps, np.array(conf)


def bleu(h, r):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], tokenize="13a")


def retained(mask, h, r):
    idx = np.flatnonzero(mask)
    return bleu([h[i] for i in idx], [r[i] for i in idx]), len(idx) / len(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dump", required=True)
    ap.add_argument("--test_dump", required=True)
    ap.add_argument("--lp", type=float, required=True)
    ap.add_argument("--train_manifest", default="")
    ap.add_argument("--train_hashes", default="",
                    help="SHA-1 digests of the training targets (tools/train_target_hashes.py), used when the "
                         "training manifest is not on disk")
    ap.add_argument("--expect_test_bleu", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.train_manifest:
        train = {l.rstrip("\n").split("\t", 1)[1].strip() for l in open(a.train_manifest) if "\t" in l}
        in_train = train.__contains__
    elif a.train_hashes:
        digests = set(open(a.train_hashes).read().split())
        in_train = lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest() in digests
    else:
        raise SystemExit("FATAL: give --train_manifest or --train_hashes")
    S = {}
    for name, dump in (("val", a.val_dump), ("test", a.test_dump)):
        r, h, c = load(dump, a.lp)
        S[name] = dict(refs=r, hyps=h, conf=c, seen=np.array([in_train(x.strip()) for x in r]))

    # ---- harness gate ----
    lpfile = Path(a.test_dump) / f"hyps_LP={a.lp}.jsonl"
    dumped = [json.loads(l)["hyp_beam"] for l in open(lpfile)]
    t = S["test"]
    same = sum(x == y for x, y in zip(dumped, t["hyps"]))
    got = bleu(t["hyps"], t["refs"])
    print(f"gate: {same}/{len(dumped)} hyps match {lpfile.name}; test BLEU {got:.3f} (recorded "
          f"{a.expect_test_bleu}); WER {corpus_wer([detokenize(x) for x in t['hyps']], [detokenize(x) for x in t['refs']]):.4f}; "
          f"seen val {S['val']['seen'].sum()} test {t['seen'].sum()}")
    if same != len(dumped) or abs(got - a.expect_test_bleu) > 0.005 or t["seen"].sum() != 230 or S["val"]["seen"].sum() != 251:
        sys.exit("HARNESS GATE FAILED")

    out = dict(lp=a.lp, rows=[])
    rng = random.Random(0)
    for subset in ("all", "unseen"):
        for split in ("val", "test"):
            s = S[split]
            keep = np.ones(len(s["refs"]), bool) if subset == "all" else ~s["seen"]
            h = [x for x, k in zip(s["hyps"], keep) if k]; r = [x for x, k in zip(s["refs"], keep) if k]
            c = s["conf"][keep]
            hd, rd = [detokenize(x) for x in h], [detokenize(x) for x in r]
            own = curve(c, hd, rd, COVS)                                  # descriptive: own-split quantiles
            longest = curve(np.array([len(x.split()) for x in h], float), hd, rd, COVS)
            rand = [np.mean([bleu(*zip(*rng.sample(list(zip(h, r)), int(round(cv * len(h)))))) for _ in range(50)])
                    for cv in COVS]
            row = dict(subset=subset, split=split, n=len(h), own_quantile=own, keep_longest=longest, random=rand)
            if split == "test":                                           # deployable: thresholds from VAL
                vc = S["val"]["conf"] if subset == "all" else S["val"]["conf"][~S["val"]["seen"]]
                row["val_threshold"] = [retained(c >= np.quantile(vc, 1 - cv), h, r) if cv < 1 else (own[0], 1.0)
                                        for cv in COVS]
            out["rows"].append(row)
            print(f"{subset:6s} {split:4s} n={len(h):3d} | own-quantile " + " ".join(f"{x:6.2f}" for x in own)
                  + " | longest " + " ".join(f"{x:6.2f}" for x in longest)
                  + " | random " + " ".join(f"{x:6.2f}" for x in rand))
            if split == "test":
                print("                     val-fixed thresholds -> " +
                      "  ".join(f"{b:6.2f} @ {cov:5.1%}" for b, cov in row["val_threshold"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
