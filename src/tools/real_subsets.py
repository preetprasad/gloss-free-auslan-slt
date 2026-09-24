#!/usr/bin/env python3
"""Real-data subsets for the wave-8 learning curve (results/PREREG_WAVE8_REALDATA_2026-09-24.md).

Two samplers over the 12,439-clip Communication train manifest, both deterministic in the seed:

  within  -- for each of the 90 videos (sorted), shuffle its clips with one random.Random(seed) stream and keep the
             first ceil(f * n_video). Prefixes of one shuffle, so f=0.25 is a subset of f=0.5 is a subset of f=1,
             and every video is present at every size.
  video   -- whole videos in random.Random(f"video:{seed}") order until `size` clips are reached; the last video
             is cut to hit `size` exactly, using the same stream. Holds out whole recording sessions.

Clips are grouped by re.match(r"(video_\\d+)_", basename). Splitting on the last underscore is WRONG: sub-clips such
as video_13_5_1 exist and would form fake videos.

  python tools/real_subsets.py --train M/train.tsv --val M/val.tsv --test M/test.tsv \
      --sampler within --frac 0.5 --seed 42 --out subset.tsv [--expect 6241]
  python tools/real_subsets.py --selftest --train results/_kaya_train_manifest.tsv
"""
import argparse, math, os, random, re, sys
from collections import OrderedDict

VID = re.compile(r"(video_\d+)_")


def read(path):
    return [l.rstrip("\n") for l in open(path) if "\t" in l]


def video_of(line):
    m = VID.match(os.path.basename(line.split("\t", 1)[0]))
    if not m:
        raise SystemExit(f"cannot parse a video id from {line.split(chr(9))[0]!r}")
    return m.group(1)


def by_video(lines):
    g = OrderedDict()
    for l in lines:
        g.setdefault(video_of(l), []).append(l)
    return OrderedDict(sorted(g.items(), key=lambda kv: int(kv[0].split("_")[1])))


def within(lines, frac, seed):
    rng, keep = random.Random(seed), set()
    for _, clips in by_video(lines).items():
        order = clips[:]
        rng.shuffle(order)
        keep.update(order[:math.ceil(frac * len(order))])
    return [l for l in lines if l in keep]          # manifest order preserved


def whole_video(lines, size, seed):
    rng, keep, vids = random.Random(f"video:{seed}"), [], list(by_video(lines).items())
    rng.shuffle(vids)
    for _, clips in vids:
        if len(keep) >= size:
            break
        need = size - len(keep)
        if len(clips) <= need:
            keep += clips
        else:
            order = clips[:]
            rng.shuffle(order)
            keep += order[:need]
    keep = set(keep)
    return [l for l in lines if l in keep]


def check(sub, full, val, test, expect=None):
    paths = {l.split("\t", 1)[0] for l in sub}
    assert len(paths) == len(sub), "duplicate clip in subset"
    assert set(sub) <= set(full), "subset row not in the train manifest"
    held = {l.split("\t", 1)[0] for l in val + test}
    assert not paths & held, f"{len(paths & held)} subset clips are in val/test"
    if expect is not None:
        assert len(sub) == expect, f"subset has {len(sub)} rows, expected {expect}"


def selftest(train):
    full = read(train)
    n_vid = len(by_video(full))
    assert len(full) == 12439 and n_vid == 90, (len(full), n_vid)
    assert len(by_video(full)) != len({l.split("\t")[0].rsplit("_", 1)[0] for l in full}), \
        "the last-underscore split should disagree with the regex (sub-clips exist)"
    for s in (42, 101, 202, 303, 404):
        a, b, c = (within(full, f, s) for f in (0.25, 0.5, 1.0))
        assert (len(a), len(b), len(c)) == (3141, 6241, 12439), (len(a), len(b), len(c))
        assert set(a) <= set(b) <= set(c) and c == full, "nesting broken"
        assert len(by_video(a)) == len(by_video(b)) == 90, "a video is missing"
        assert within(full, 0.5, s) == b, "not deterministic"
        v = whole_video(full, 6241, s)
        assert len(v) == 6241 and whole_video(full, 6241, s) == v
        assert len(by_video(v)) < 90, "video sampler kept every video"
    assert within(full, 0.5, 42) != within(full, 0.5, 101), "seed has no effect"
    print(f"selftest OK: 12439 clips / 90 videos; within 3141/6241/12439 nested for 5 seeds; "
          f"video sampler seed 42 keeps {len(by_video(whole_video(full, 6241, 42)))} videos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val"); ap.add_argument("--test")
    ap.add_argument("--sampler", choices=["within", "video"])
    ap.add_argument("--frac", type=float); ap.add_argument("--size", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--out"); ap.add_argument("--expect", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.train)
    full, val, test = read(a.train), read(a.val), read(a.test)
    if a.sampler == "within":
        sub = within(full, a.frac, a.seed)
        smaller = [within(full, f, a.seed) for f in (0.25, 0.5, 1.0) if f <= a.frac]
        for x, y in zip(smaller, smaller[1:]):
            assert set(x) <= set(y), "nesting broken"
        assert len(by_video(sub)) == len(by_video(full)), "a video is missing"
    else:
        sub = whole_video(full, a.size, a.seed)
    check(sub, full, val, test, a.expect)
    with open(a.out, "w") as f:
        f.write("\n".join(sub) + "\n")
    kept = sorted(by_video(sub), key=lambda v: int(v.split("_")[1]))
    print(f"[subset] {a.sampler} seed {a.seed}: {len(sub)} rows, {len(kept)} videos -> {a.out}")
    if a.sampler == "video":
        print("[subset] videos kept:", " ".join(kept))


if __name__ == "__main__":
    sys.exit(main())
