"""Gates for build_posestitch_corpus_v2.py --shuffle_order (wave 10, results/PREREG_WAVE10_SIGNORDER_2026-10-01.md).

  shuffle_order_check.py inert OLD_DIR NEW_DIR   NEW (flag off) reproduces OLD byte-for-byte: every clip's KPS/BBOX
                                                 arrays, the manifest text, and the audit sidecar if both have one.
  shuffle_order_check.py order OFF_DIR ON_DIR    ON (--shuffle_order) differs from OFF only in sign order: same English
                                                 row-for-row, same clip length, each sign's frames identical, the same
                                                 transition budget at each position, a non-identity permutation in
                                                 every clip. Needs both audit sidecars (audit.jsonl in each dir).
Each DIR holds posestitch_v2_train.tsv (and audit.jsonl). Exits non-zero on the first failure.
Verified locally on 300 production-flag sentences over 6,430 MM-WLAuslan valid clips: inert 300/300; order 300/300,
48.3% of sign pairs still in English relative order; both modes fail when handed the wrong corpus."""
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


if __name__ == "__main__":
    {"inert": inert, "order": order}[sys.argv[1]](*sys.argv[2:4])
