"""Wave-5 corpus gate: concatenate the build shards, check them, and write the nested prefix manifests.

Pre-registration: results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md sections 3, 5 and 7 (G-build). Refuses to write
anything unless: every clip path exists; each shard's audit sidecar matches its manifest row for row; no synthetic
sentence equals a Communication train / validation / test reference; frames per sign sit in the adopted range (mean
21-25.5, median 19-21); at least max(--sizes) clips were built (a shortfall STOPS the pipeline, prereg section 3).
Then writes syn_<N>.tsv for each size as the FIRST N rows (strict prefixes) and a JSON report.
Usage:
  python tools/w5_prep_check.py --shard_dirs d0 d1 ... --refs train.tsv val.tsv test.tsv --out_dir OUT \
      --sizes 25000 100000 200000 300000
"""
import argparse, json, os, re, statistics, sys

MANIFEST, AUDIT = "posestitch_v2_train.tsv", "audit.jsonl"


def key(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def load_shards(dirs):
    rows, segs_per_row = [], []
    for d in dirs:
        man = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, MANIFEST)) if "\t" in l]
        aud = [json.loads(l) for l in open(os.path.join(d, AUDIT)) if l.strip()]
        assert len(man) == len(aud), f"{d}: manifest {len(man)} rows vs audit {len(aud)}"
        for (p, t), r in zip(man, aud):
            assert r["clip_path"] == p and r["text"] == t, f"{d}: audit and manifest disagree at {p}"
            rows.append((p, t))
            segs_per_row.append(r["segments"])
    return rows, segs_per_row


def describe(rows, segs, ad_vocab):
    words = [w for _, t in rows for w in t.split() if w not in {".", ","}]
    return {"clips": len(rows), "distinct_sentences": len({key(t) for _, t in rows}),
            "distinct_words": len(set(words)),
            "tokens_outside_ad_vocab_share": round(sum(w not in ad_vocab for w in words) / len(words), 4),
            "distinct_glosses": len({s["gloss"] for ss in segs for s in ss}),
            "distinct_recordings": len({s["source_id"] for ss in segs for s in ss}),
            "signs": sum(len(ss) for ss in segs)}


def check(rows, segs, ref_texts, ad_vocab, sizes, exists=os.path.exists):
    missing = [p for p, _ in rows if not exists(p)]
    assert not missing, f"{len(missing)} clip paths missing, e.g. {missing[:3]}"
    leaks = {key(t) for _, t in rows} & {key(t) for t in ref_texts}
    assert not leaks, f"{len(leaks)} synthetic sentences equal a real reference, e.g. {sorted(leaks)[:3]}"
    fr = [s["output_frames"] for ss in segs for s in ss]
    mean, med = statistics.mean(fr), statistics.median(fr)
    assert 21 <= mean <= 25.5 and 19 <= med <= 21, f"frames/sign mean {mean:.2f} median {med} outside the adopted range"
    assert len(rows) >= max(sizes), (f"only {len(rows)} clips built, the largest arm needs {max(sizes)}: "
                                     "stop, regenerate the pool, rebuild (prereg section 3)")
    report = {"total_clips": len(rows), "frames_per_sign": {"mean": round(mean, 2), "median": med},
              "leaks_vs_train_val_test": 0, "sizes": {}}
    for n in sizes:
        report["sizes"][n] = describe(rows[:n], segs[:n], ad_vocab)
    return report, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", required=True, help="Communication train / val / test manifests")
    ap.add_argument("--ad_train", required=True, help="Auslan-Daily train manifest (for the vocabulary share)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[25000, 100000, 200000, 300000])
    ap.add_argument("--sentences_in", type=int, required=True, help="sentences fed to the builder (for the skip count)")
    a = ap.parse_args()
    rows, segs = load_shards(a.shard_dirs)
    refs = [l.rstrip("\n").split("\t")[1] for m in a.refs for l in open(m) if "\t" in l]
    ad_vocab = {w for l in open(a.ad_train) if "\t" in l for w in l.rstrip("\n").split("\t")[1].split()}
    report, sizes = check(rows, segs, refs, ad_vocab, a.sizes)
    report["sentences_in"], report["skipped_by_builder"] = a.sentences_in, a.sentences_in - len(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    for n in sizes:
        with open(os.path.join(a.out_dir, f"syn_{n}.tsv"), "w") as f:
            f.writelines(f"{p}\t{t}\n" for p, t in rows[:n])
    json.dump(report, open(os.path.join(a.out_dir, "w5_prep_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("W5_PREP_OK")


def selftest():
    seg = lambda g, sid, o=20: {"gloss": g, "source_id": sid, "output_frames": o}
    rows = [("a.npz", "hi there ."), ("b.npz", "we eat pizza ."), ("c.npz", "you go now .")]
    segs = [[seg("hi", "1", 26), seg("there", "2")], [seg("we", "3", 22), seg("eat", "4"), seg("pizza", "5", 24)],
            [seg("you", "6"), seg("go", "7"), seg("now", "8")]]
    rep, sizes = check(rows, segs, ["i am possum ."], {"hi", "there", "we", "eat", "you", "go", "now"}, [2, 3],
                       exists=lambda p: True)
    assert sizes == [2, 3] and rep["sizes"][2]["clips"] == 2 and rep["sizes"][3]["distinct_glosses"] == 8
    assert rep["sizes"][3]["tokens_outside_ad_vocab_share"] == round(1 / 8, 4)
    for bad, msg in ((["We eat pizza."], "leak"), (None, "missing")):
        try:
            check(rows, segs, bad or [], set(), [3], exists=(lambda p: p != "b.npz") if bad is None else (lambda p: True))
        except AssertionError:
            continue
        raise SystemExit(f"selftest: {msg} not caught")
    for bad_segs, bad_sizes, msg in (([[seg("x", "1", 60)]] * 3, [3], "frame range"), (segs, [2, 4], "shortfall")):
        try:
            check(rows, bad_segs, [], set(), bad_sizes, exists=lambda p: True)
        except AssertionError:
            continue
        raise SystemExit(f"selftest: {msg} not enforced")
    print("selftest passed")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
