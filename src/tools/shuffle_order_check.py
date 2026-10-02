"""Gates for build_posestitch_corpus_v2.py --shuffle_order (wave 10, results/PREREG_WAVE10_SIGNORDER_2026-10-01.md).

  shuffle_order_check.py inert OLD_DIR NEW_DIR   NEW (flag off) reproduces OLD byte-for-byte: every clip's KPS/BBOX
                                                 arrays, the manifest text, and the audit sidecar if both have one.
  shuffle_order_check.py order OFF_DIR ON_DIR    ON (--shuffle_order) differs from OFF only in sign order: same English
                                                 row-for-row, same clip length, each sign's frames identical, the same
                                                 transition budget at each position, a non-identity permutation in
                                                 every clip. Needs both audit sidecars (audit.jsonl in each dir).
  shuffle_order_check.py match A_DIR D_DIR DL_DIR   DL (--match_frames A) has D's signs, exemplars and transition
                                                 budgets, A's clip length, and sign lengths that are the exact
                                                 largest-remainder rescale of D's sampled lengths (wave 7 Stage 2,
                                                 results/PREREG_WAVE7_DENSITY_2026-09-23.md section 10).
Each DIR holds posestitch_v2_train.tsv (and audit.jsonl). Exits non-zero on the first failure.
Verified locally on 300 production-flag sentences over 6,430 MM-WLAuslan valid clips: inert 300/300; order 300/300,
48.3% of sign pairs still in English relative order; match 300/300; every mode fails when handed the wrong corpus."""
import itertools, json, os, sys
import numpy as np


def rows(d):
    return [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(d, "posestitch_v2_train.tsv")) if "\t" in l]


def audit(d):
    p = os.path.join(d, "audit.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def arrays(p):
    z = np.load(p)
    return z["KPS"], z["BBOX"]


def inert(old_d, new_d):
    old, new = rows(old_d), rows(new_d)
    assert len(old) == len(new) > 0, (len(old), len(new))
    for (po, to), (pn, tn) in zip(old, new):
        assert to == tn and os.path.basename(po) == os.path.basename(pn), (po, pn)
        (ko, bo), (kn, bn) = arrays(po), arrays(pn)
        assert np.array_equal(ko, kn) and np.array_equal(bo, bn), pn
    ao, an = audit(old_d), audit(new_d)
    if ao is not None and an is not None:
        strip = lambda a: json.dumps({**a, "clip_path": os.path.basename(a["clip_path"])}, sort_keys=True)
        assert [strip(a) for a in ao] == [strip(a) for a in an], "audit sidecars differ"
    print(f"INERT OK: {len(new)}/{len(old)} clips byte-identical (arrays + manifest text"
          f"{' + audit' if ao is not None and an is not None else ''})")


def order(off_d, on_d):
    off, on, off_a, on_a = rows(off_d), rows(on_d), audit(off_d), audit(on_d)
    assert off_a is not None and on_a is not None, "both dirs need audit.jsonl"
    assert len(off) == len(on) == len(off_a) == len(on_a) > 0
    kept = total = same_glosses = 0
    for (pf, tf), (pn, tn), af, an in zip(off, on, off_a, on_a):
        assert tf == tn and af["text"] == tf and an["text"] == tn, "English text differs"
        (kf, bf), (kn, bn) = arrays(pf), arrays(pn)
        assert kf.shape == kn.shape and bf.shape == bn.shape, f"clip length differs: {pn}"
        sf, sn = af["segments"], an["segments"]
        perm = [s["english_position"] for s in sn]
        assert sorted(perm) == list(range(len(sf))) and perm != sorted(perm), f"not a non-identity permutation: {perm}"
        prev_end = -1
        for pos, s in enumerate(sn):
            ref = sf[s["english_position"]]
            for key in ("gloss", "source_id", "output_frames"):
                assert s[key] == ref[key], (pn, key)
            n_t = s["transition_frames_before"]
            assert n_t == sf[pos]["transition_frames_before"], f"budget moved: {pn}"
            assert s["start_frame"] == prev_end + 1 + n_t, f"segments not contiguous: {pn}"
            if n_t:                                     # blend of the LAID-OUT neighbours, box held from the previous sign
                w = np.linspace(0, 1, n_t + 2)[1:-1][:, None, None]
                blend = (1 - w) * kn[prev_end][None] + w * kn[s["start_frame"]][None]
                assert np.allclose(kn[prev_end + 1:s["start_frame"]], blend, atol=1e-5), f"transition frames: {pn}"
                assert np.array_equal(bn[prev_end + 1:s["start_frame"]], np.repeat(bn[prev_end][None], n_t, 0)), pn
            a0, b0, a1, b1 = ref["start_frame"], ref["end_frame"] + 1, s["start_frame"], s["end_frame"] + 1
            assert np.array_equal(kf[a0:b0], kn[a1:b1]) and np.array_equal(bf[a0:b0], bn[a1:b1]), f"sign frames: {pn}"
            prev_end = s["end_frame"]
        assert prev_end + 1 == kn.shape[0], f"frames after the last sign: {pn}"
        same_glosses += [s["gloss"] for s in sn] == [s["gloss"] for s in sf]   # a repeated sign can hide the shuffle
        for i, j in itertools.combinations(range(len(perm)), 2):
            if sn[i]["gloss"] != sn[j]["gloss"]:
                total += 1
                kept += perm[i] < perm[j]
    print(f"ORDER OK: {len(on)}/{len(off)} clips permuted (non-identity); same English, length, sign frames, per-position "
          f"transition budgets, contiguous segments, transitions blend the laid-out neighbours; {same_glosses} clips show "
          f"the same gloss sequence as English order (repeated signs); pairs of different signs still in English relative "
          f"order {kept}/{total} = {kept / total:.1%}")


def _alloc(lengths, total):
    """Independent re-implementation of the builder's largest-remainder allocation (exact sum, ties by position)."""
    from fractions import Fraction
    s = sum(lengths)
    raw = [Fraction(n * total, s) for n in lengths]
    out = [int(r) for r in raw]                      # floor (all positive)
    rest = sorted(range(len(raw)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in rest[:total - sum(out)]:
        out[i] += 1
    return out


def match(a_d, d_d, dl_d):
    """DL (--match_frames A) against D (same flags, no --match_frames) and A (the reference): same English row for row,
    every DL clip exactly as long as its A clip, the same signs/exemplars/transition budgets as D, sign lengths rescaled
    from D's sampled lengths by the exact largest-remainder rule, contiguous segments, transitions that blend the
    laid-out neighbours. Needs audit.jsonl in D_DIR and DL_DIR."""
    A, D, L, da, la = rows(a_d), rows(d_d), rows(dl_d), audit(d_d), audit(dl_d)
    assert da is not None and la is not None, "D and DL dirs need audit.jsonl"
    assert len(A) == len(D) == len(L) == len(da) == len(la) > 0, (len(A), len(D), len(L))
    ratios, longer = [], 0
    for (pa, ta), (pd, td), (pl, tl), ad, al in zip(A, D, L, da, la):
        assert ta == td == tl == ad["text"] == al["text"], "English text differs"
        ka, _ = arrays(pa)
        kd, _ = arrays(pd)
        kl, bl = arrays(pl)
        assert kl.shape[0] == ka.shape[0] == al["match_frames_total"], f"DL length != A length: {pl}"
        sd, sl = ad["segments"], al["segments"]
        assert len(sd) == len(sl) > 0, pl
        trans = 0
        for s, r in zip(sl, sd):
            for key in ("gloss", "source_id", "transition_frames_before"):
                assert s[key] == r[key], (pl, key)
            trans += s["transition_frames_before"]
        assert al["sampled_frames"] == [r["output_frames"] for r in sd], f"DL's sampled lengths are not D's: {pl}"
        want = _alloc(al["sampled_frames"], kl.shape[0] - trans)
        assert [s["output_frames"] for s in sl] == want, f"sign lengths are not the exact proportional allocation: {pl}"
        prev_end = -1
        for s in sl:
            n_t = s["transition_frames_before"]
            assert s["start_frame"] == prev_end + 1 + n_t, f"segments not contiguous: {pl}"
            if n_t:
                w = np.linspace(0, 1, n_t + 2)[1:-1][:, None, None]
                blend = (1 - w) * kl[prev_end][None] + w * kl[s["start_frame"]][None]
                assert np.allclose(kl[prev_end + 1:s["start_frame"]], blend, atol=1e-5), f"transition frames: {pl}"
                assert np.array_equal(bl[prev_end + 1:s["start_frame"]], np.repeat(bl[prev_end][None], n_t, 0)), pl
            prev_end = s["end_frame"]
        assert prev_end + 1 == kl.shape[0], f"frames after the last sign: {pl}"
        ratios.append(sum(want) / sum(al["sampled_frames"]))
        longer += kl.shape[0] > kd.shape[0]
    r = np.array(ratios)
    print(f"MATCH OK: {len(L)}/{len(A)} DL clips as long as their A clip; same signs, exemplars and transition budgets as D; "
          f"sign lengths = exact proportional rescale of D's sampled lengths; contiguous; transitions blend neighbours. "
          f"Sign-length scale factor median {np.median(r):.3f} (p10 {np.percentile(r, 10):.3f}, p90 {np.percentile(r, 90):.3f}); "
          f"DL longer than D in {longer}/{len(L)} clips")


if __name__ == "__main__":
    mode = sys.argv[1]
    {"inert": inert, "order": order, "match": match}[mode](*sys.argv[2:5 if mode == "match" else 4])
