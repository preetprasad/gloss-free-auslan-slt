#!/usr/bin/env python3
"""Write SHA-1 digests of the training targets.

The seen-target / unseen-target scores need to know whether a test reference occurs verbatim among the
training targets. The training subtitles are dataset content and are not redistributed, so the repository
ships their digests instead: membership can be tested, the sentences cannot be read back.

    python tools/train_target_hashes.py <train.tsv> ../results/train_target_hashes.txt

One hex digest per line, sorted. Each tool compares references in its own form, so four forms of every
target are hashed: the detokenised sentence (tools/reproduce_headline.py), the lower-cased and
whitespace-collapsed sentence (tools/paired_bootstrap.py), the detokenised, lower-cased and
whitespace-collapsed sentence (metric_suite.py) and the stripped raw sentence (tools/ensemble_selective.py).
Anyone holding the dataset can rebuild the file and compare.
"""
import hashlib, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.metrics import detokenize  # noqa: E402


def digest(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():
    src, out = sys.argv[1], sys.argv[2]
    targets = [line.rstrip("\n").split("\t", 1)[1] for line in open(src, encoding="utf-8") if "\t" in line]
    forms = set()
    for t in targets:
        forms |= {digest(detokenize(t)), digest(" ".join(t.lower().split())),
                  digest(" ".join(detokenize(t).lower().split())), digest(t.strip())}
    with open(out, "w") as fh:
        fh.write("\n".join(sorted(forms)) + "\n")
    print(f"{len(targets)} training targets, {len(set(targets))} distinct, {len(forms)} digests -> {out}")


if __name__ == "__main__":
    main()
