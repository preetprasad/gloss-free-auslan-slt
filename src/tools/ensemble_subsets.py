"""Full-coverage ablation of an ensemble: EVERY non-empty subset of its members, decoded as the headline was.

Decoding is the headline tool's own (ensemble_mbr_wer: n-best at --beam, top-1 by raw / words**LP), not
ensemble_decode.ensemble_beam, which divides by the token count INCLUDING BOS and EOS and therefore picks a
different candidate (six members, validation: 12.383 instead of the recorded 12.470).
Each member is encoded once per batch; every subset re-uses those encodings.

  decode one shard:  python tools/ensemble_subsets.py decode --ckpts a.pt,b.pt,.. --manifest val.tsv \
                         --rgb_feat_dir DIR --out OUT --shard 0 --nshards 4
  report:            python tools/ensemble_subsets.py report --out OUT --names A,B,.. --types p,f,.. \
                         --expect 12.470 --expect_wer 0.9164 [--boot 500]
  self-test:         python tools/ensemble_subsets.py self_test

A subset is a bit mask over the member list (bit i = member i). The report refuses to print anything but the
gate line unless the full set reproduces --expect.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from itertools import combinations
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def shapley(v, n):
    """Shapley value of each of n members; v maps a bit mask to a score, v[0] is the empty set."""
    phi = [0.0] * n
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                m = sum(1 << j for j in S)
                phi[i] += w * (v[m | 1 << i] - v[m])
    return phi


def members_of(mask, n):
    return [i for i in range(n) if mask >> i & 1]


def decode(a):
    import torch
    from torch.utils.data import DataLoader
    from config import Config
    from slt.data import Vocab
    from slt.preproc_guard import assert_preproc_matches_checkpoints
    from slt.rgb_fusion import RGBDataset, collate_rgb
    from ensemble_decode import Member
    from ensemble_mbr_wer import _align_vocabs, ensemble_nbest

    cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame", rgb_feat_dir=a.rgb_feat_dir,
                 rgb_feature_dim=1152, test_manifest=a.manifest, device="cpu")
    dev = torch.device("cpu")
    paths = a.ckpts.split(",")
    n = len(paths)
    assert os.path.isdir(a.rgb_feat_dir), f"rgb_feat_dir {a.rgb_feat_dir!r} is not a directory"
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    assert_preproc_matches_checkpoints(cfg, cks, False)
    vocab = Vocab(cks[0]["vocab"])
    members = _align_vocabs(cks, [Member(c, cfg, Vocab(c["vocab"]), dev) for c in cks],
                            list(cks[0]["vocab"]), dev, False)
    masks = ([int(m) for m in a.masks.split(",")] if a.masks else
             [m for k, m in enumerate(sorted(range(1, 1 << n), key=lambda m: (bin(m).count("1"), m)))
              if k % a.nshards == a.shard])
    ds = RGBDataset(a.manifest, vocab, cfg, "")
    if a.limit: ds.samples = ds.samples[:a.limit]
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_rgb)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    print(f"[subsets] members {n} fusion={[m.fusion for m in members]} | clips {len(ds)} | masks {len(masks)} "
          f"| beam {a.beam} LP {a.lp} | threads {torch.get_num_threads()}", flush=True)

    hyps = {m: [] for m in masks}; refs = []
    for bi, (src, src_pad, rgb, rgb_pad, _t, texts) in enumerate(dl):
        with torch.no_grad():
            mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in members]      # encode once
            for mask in masks:
                idx = members_of(mask, n)
                nb = ensemble_nbest([members[i] for i in idx], [mems[i] for i in idx], mx, a.beam, dev)
                for cands in nb:
                    txt = [(raw, vocab.decode(seq)) for raw, seq in cands]
                    hyps[mask].append(max(txt, key=lambda c: c[0] / (max(1, len(c[1].split())) ** a.lp))[1])
        refs += list(texts)
        if bi % 10 == 0: print(f"[subsets] batch {bi + 1}/{len(dl)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    tag = "masks_" + "_".join(map(str, masks)) if a.masks else f"shard{a.shard}of{a.nshards}"
    f = os.path.join(a.out, f"{tag}.json")
    json.dump({"n": n, "ckpts": paths, "ckpt_md5": [hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths],
               "manifest": a.manifest, "beam": a.beam, "lp": a.lp, "refs": refs,
               "hyps": {str(m): h for m, h in hyps.items()}}, open(f, "w"))
    print(f"[subsets] wrote {f}", flush=True)


def report(a):
    import numpy as np
    from ensemble_mbr_wer import corpus_scores

    files = sorted(glob.glob(os.path.join(a.out, "shard*.json")) + glob.glob(os.path.join(a.out, "masks_*.json")))
    assert files, f"no decode files in {a.out}"
    refs, hyps, meta = None, {}, None
    for f in files:
        d = json.load(open(f))
        key = (d["n"], d["ckpt_md5"], d["beam"], d["lp"])
        assert meta in (None, key), f"{f}: members or decoding differ from the other files"
        assert refs in (None, d["refs"]), f"{f}: references differ from the other files"
        meta, refs = key, d["refs"]
        hyps.update({int(m): h for m, h in d["hyps"].items()})
    n = meta[0]
    full = (1 << n) - 1
    names = a.names.split(",") if a.names else [f"m{i}" for i in range(n)]
    assert len(names) == n
    assert all(len(h) == len(refs) for h in hyps.values())

    bleu, wer, *_ = corpus_scores(hyps[full], refs)
    ok = abs(bleu - a.expect) < 0.005 and (a.expect_wer is None or abs(wer - a.expect_wer) < 0.0005)
    print(f"GATE {'PASSED' if ok else 'FAILED'}: full set BLEU-4 {bleu:.3f} WER {wer:.4f} | recorded {a.expect} "
          f"{a.expect_wer} | clips {len(refs)} | beam {meta[2]} LP {meta[3]}")
    if not ok: sys.exit(1)
    missing = [m for m in range(1, full + 1) if m not in hyps]
    assert not missing, f"{len(missing)} subsets not decoded: {missing[:8]}"

    sc = {m: corpus_scores(hyps[m], refs) for m in range(1, full + 1)}
    v = {0: 0.0, **{m: s[0] for m, s in sc.items()}}
    lab = lambda m: "+".join(names[i] for i in members_of(m, n))
    print(f"\n{'subset':60s} {'size':>4s} {'BLEU-4':>7s} {'WER':>7s} {'uniq':>5s}")
    for m in sorted(sc, key=lambda m: (bin(m).count("1"), -sc[m][0])):
        print(f"{lab(m):60s} {bin(m).count('1'):4d} {sc[m][0]:7.3f} {sc[m][1]:7.4f} {sc[m][5]:5d}")

    print("\nmean BLEU-4 by number of members (min, max):")
    for k in range(1, n + 1):
        x = [sc[m][0] for m in sc if bin(m).count("1") == k]
        print(f"  {k}: {np.mean(x):6.3f}  ({min(x):.3f}, {max(x):.3f})  over {len(x)} subsets")

    phi = shapley(v, n)
    assert abs(sum(phi) - v[full]) < 1e-9
    gain = lambda i, vv: np.mean([vv[m | 1 << i] - vv[m] for m in range(1, full + 1) if not m >> i & 1])
    print("\nper member: alone | leave-one-out loss | mean gain when added to a non-empty subset | Shapley (empty set = 0)")
    for i in range(n):
        print(f"  {names[i]:12s} {v[1 << i]:7.3f} | {v[full] - v[full ^ 1 << i]:+6.3f} | {gain(i, v):+6.3f} | {phi[i]:6.3f}")

    if a.types:
        ty = a.types.split(","); assert len(ty) == n
        print("\nby type:")
        for t in sorted(set(ty)):
            m = sum(1 << i for i in range(n) if ty[i] == t)
            print(f"  all {t} members ({lab(m)}): BLEU-4 {sc[m][0]:.3f} WER {sc[m][1]:.4f}")
            print(f"  Shapley sum of {t} members: {sum(phi[i] for i in range(n) if ty[i] == t):.3f}")

    out = {"names": names, "clips": len(refs), "beam": meta[2], "lp": meta[3],
           "scores": {lab(m): {"mask": m, "bleu4": sc[m][0], "wer": sc[m][1], "uniq": sc[m][5]} for m in sc},
           "shapley": dict(zip(names, phi))}

    if a.boot:
        # sentence bootstrap of the member-level readouts ("test sampling only": the checkpoints are fixed)
        from slt.metrics import corpus_bleu4, detokenize
        R = [detokenize(x) for x in refs]; H = {m: [detokenize(x) for x in hyps[m]] for m in sc}
        rng = np.random.default_rng(0); rows = []; tdiff = []; tmask = None
        if a.types and len(set(a.types.split(","))) == 2:
            tnames = sorted(set(a.types.split(",")))
            tmask = [sum(1 << i for i in range(n) if a.types.split(",")[i] == t) for t in tnames]
        for b in range(a.boot):
            ix = rng.integers(0, len(R), len(R)); r = [R[j] for j in ix]
            vb = {0: 0.0, **{m: corpus_bleu4([H[m][j] for j in ix], r, tokenize="13a") for m in sc}}
            pb = shapley(vb, n)
            rows.append([vb[full] - vb[full ^ 1 << i] for i in range(n)] + [gain(i, vb) for i in range(n)] + pb)
            if tmask: tdiff.append((sum(pb[i] for i in members_of(tmask[0], n)) - sum(pb[i] for i in members_of(tmask[1], n)),
                                    vb[tmask[0]] - vb[tmask[1]]))
            if b % 50 == 0: print(f"[boot] {b}/{a.boot}", flush=True)
        lo, hi = np.percentile(np.array(rows), [2.5, 97.5], axis=0)
        if tmask:
            (l1, l2), (h1, h2) = np.percentile(np.array(tdiff), [2.5, 97.5], axis=0)
            print(f"\n{tnames[0]} minus {tnames[1]}, 95% interval: Shapley sums [{l1:+.3f}, {h1:+.3f}] | "
                  f"type-only ensembles [{l2:+.3f}, {h2:+.3f}]")
            out["type_diff"] = {"types": tnames, "shapley_sum": [l1, h1], "type_only_bleu4": [l2, h2]}
        print(f"\n95% intervals, {a.boot} sentence resamples: leave-one-out loss | mean gain | Shapley")
        for i in range(n):
            j = [i, n + i, 2 * n + i]
            print(f"  {names[i]:12s} " + " | ".join(f"[{lo[k]:+.3f}, {hi[k]:+.3f}]" for k in j))
        out["boot"] = {"resamples": a.boot, "lo": lo.tolist(), "hi": hi.tolist(),
                       "columns": ["loo"] * n + ["mean_gain"] * n + ["shapley"] * n}
    json.dump(out, open(os.path.join(a.out, "subsets_scores.json"), "w"), indent=1)


def self_test(_):
    w = [3.0, 1.0, 0.5, 2.0]                                   # additive game: Shapley value = own weight
    v = {m: sum(w[i] for i in members_of(m, 4)) for m in range(16)}
    assert all(abs(p - x) < 1e-12 for p, x in zip(shapley(v, 4), w))
    v = {m: float(bin(m).count("1") >= 2) for m in range(8)}   # symmetric game: equal shares, summing to v(full)
    assert all(abs(p - 1 / 3) < 1e-12 for p in shapley(v, 3))
    assert members_of(0b101001, 6) == [0, 3, 5]
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--ckpts", required=True); d.add_argument("--manifest", required=True)
    d.add_argument("--rgb_feat_dir", required=True); d.add_argument("--out", required=True)
    d.add_argument("--beam", type=int, default=10); d.add_argument("--lp", type=float, default=0.3)
    d.add_argument("--shard", type=int, default=0); d.add_argument("--nshards", type=int, default=1)
    d.add_argument("--masks", default="", help="explicit comma-separated masks instead of a shard")
    d.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True); r.add_argument("--names", default=""); r.add_argument("--types", default="")
    r.add_argument("--expect", type=float, required=True); r.add_argument("--expect_wer", type=float, default=None)
    r.add_argument("--boot", type=int, default=0)
    sub.add_parser("self_test")
    a = ap.parse_args()
    {"decode": decode, "report": report, "self_test": self_test}[a.cmd](a)
