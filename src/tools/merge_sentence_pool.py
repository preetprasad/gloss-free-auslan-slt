"""Merge generator shards into one pool: de-duplicate, drop anything matching a real reference, shuffle.

Pre-registration: results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md section 3. Matching uses a normalised key (lowercase,
punctuation removed, whitespace collapsed), so "i am possum ." and "I am possum." are the same sentence. The output keeps
each sentence's first-seen surface form. Usage:
  python tools/merge_sentence_pool.py --shards s0.txt s1.txt ... --exclude train.tsv val.tsv test.tsv \
      --out pool.txt --min_size 300000 --seed 42
"""
import argparse, random, re, sys


def key(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def merge(shard_lines, exclude_texts, seed):
    banned = {key(t) for t in exclude_texts}
    seen, pool, dup, leak = set(), [], 0, 0
    for s in shard_lines:
        k = key(s)
        if not k:
            continue
        if k in banned:
            leak += 1
            continue
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        pool.append(s.strip())
    random.Random(seed).shuffle(pool)
    return pool, dup, leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="+", required=True, help="manifests whose texts must never appear in the pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_size", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    lines = [l for f in a.shards for l in open(f) if l.strip()]
    excl = [l.rstrip("\n").split("\t")[1] for m in a.exclude for l in open(m) if "\t" in l]
    pool, dup, leak = merge(lines, excl, a.seed)
    assert not {key(s) for s in pool} & {key(t) for t in excl}, "leak survived the merge"
    open(a.out, "w").write("\n".join(pool) + "\n")
    print(f"shards {len(a.shards)} | lines {len(lines)} | cross-shard duplicates {dup} | dropped as matching a real "
          f"reference {leak} | pool {len(pool)} -> {a.out}")
    if len(pool) < a.min_size:
        sys.exit(f"POOL TOO SMALL: {len(pool)} < {a.min_size}")


def selftest():
    pool, dup, leak = merge(["Hello there .", "hello  there.", "i am possum .", "a new one ."], ["i am possum ."], 0)
    assert dup == 1 and leak == 1 and sorted(pool) == ["Hello there .", "a new one ."]
    assert key("It's OK, Sally!") == "it's ok sally"
    print("selftest passed")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
