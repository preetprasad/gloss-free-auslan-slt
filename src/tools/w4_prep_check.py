"""Wave-4 corpus gate: prove the three stitch corpora are what the pre-registration says, then write the
joint manifests. Exits non-zero on ANY violation, so the dependent training array never starts.

  A  rebuild of the adopted corpus (trim 0.15, 1 realisation) -> must reproduce stitch_scale2/syn_13760
     array-for-array (not by md5: np.savez_compressed stamps zip mtimes, so identical data hashes differently).
  T  identical build with --trim_frac 0.0 -> same texts, and per segment the same gloss, exemplar,
     output length, transition and frame bounds as A. Only the trimmed SOURCE content may differ.
  Q  --realizations 2 -> the first 6,880 of A's texts, each exactly twice, adjacent.
  all: sign duration lands on the Sign_Spotting anchor (mean ~23.1 / median 20 frames per sign).

python tools/w4_prep_check.py --selftest   runs the check against deliberately broken inputs.
"""
import argparse, hashlib, json, os, statistics, sys
from multiprocessing import Pool

import numpy as np

SEG_KEYS = ("gloss", "source_dir", "source_id", "output_frames", "transition_frames_before",
            "start_frame", "end_frame")


def rows(path):
    return [l.rstrip("\n").split("\t", 1) for l in open(path, encoding="utf-8") if "\t" in l]


def audit(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def arr_hash(p):
    z = np.load(p)
    return hashlib.md5(z["KPS"].tobytes() + z["BBOX"].tobytes()).hexdigest()


def check(real, orig, A, T, Q, audA, audT, audQ, n=13760, pool=None):
    """Pure check over already-loaded rows/audits; returns a list of violation strings."""
    bad = []
    txt = lambda r: [t for _, t in r]
    if len(A) != n or len(T) != n or len(Q) != n:
        bad.append(f"sizes A {len(A)} T {len(T)} Q {len(Q)} (want {n} each)")
    if txt(A) != txt(orig[:n]):
        bad.append("A texts do not reproduce syn_13760")
    if pool is not None and not bad:
        ha = pool.map(arr_hash, [p for p, _ in A], chunksize=64)
        ho = pool.map(arr_hash, [p for p, _ in orig[:n]], chunksize=64)
        diff = sum(a != b for a, b in zip(ha, ho))
        print(f"A vs adopted corpus: {n - diff}/{n} clips array-identical")
        if diff:
            bad.append(f"A differs from the adopted corpus in {diff} clips")
    if txt(T) != txt(A):
        bad.append("T texts differ from A row-for-row")
    seg_bad = sum(len(a["segments"]) != len(t["segments"]) or
                  any(sa[k] != st[k] for sa, st in zip(a["segments"], t["segments"]) for k in SEG_KEYS)
                  for a, t in zip(audA, audT))
    if seg_bad or len(audA) != len(audT):
        bad.append(f"T not paired with A: {seg_bad} clips with segment mismatches")
    untrimmed = [s["source_frames_after_trim"] == s["source_frames_original"] for a in audT for s in a["segments"]]
    if not all(untrimmed):
        bad.append("T contains trimmed segments")
    if all(s["source_frames_after_trim"] == s["source_frames_original"] for a in audA for s in a["segments"]):
        bad.append("A is untrimmed (trim did not apply)")
    half = txt(A)[: n // 2]
    if txt(Q)[0::2] != half or txt(Q)[1::2] != half:
        bad.append("Q is not the first n/2 A texts, each twice")
    same = sum([s["source_id"] for s in x["segments"]] == [s["source_id"] for s in y["segments"]]
               for x, y in zip(audQ[0::2], audQ[1::2]))
    print(f"Q: {n // 2} sentences x 2; realisation pairs with identical exemplar sequence: {same}")
    for name, aud in (("A", audA), ("T", audT), ("Q", audQ)):
        fr = [s["output_frames"] for a in aud for s in a["segments"]]
        mean, med = statistics.mean(fr), statistics.median(fr)
        print(f"{name}: {len(fr)} signs | frames/sign mean {mean:.2f} median {med}")
        if not (21.0 <= mean <= 25.5 and 19 <= med <= 21):
            bad.append(f"{name} sign duration off the 23.1/20 anchor ({mean:.2f}/{med})")
    return bad


def selftest():
    seg = lambda sid, o=20: dict(gloss="G", source_dir="d", source_id=sid, output_frames=o,
                                 transition_frames_before=0, start_frame=0, end_frame=o - 1,
                                 source_frames_original=80, source_frames_after_trim=56)
    n = 4
    A = [(f"a{i}", f"t{i}") for i in range(n)]
    audA = [dict(segments=[seg(f"s{i}", (20, 20, 20, 32)[i])]) for i in range(n)]
    audT = [dict(segments=[dict(s, source_frames_after_trim=80) for s in a["segments"]]) for a in audA]
    Q = [(f"q{i}", A[i // 2][1]) for i in range(n)]
    audQ = [dict(segments=[seg(f"s{i}", (20, 20, 20, 32)[i])]) for i in range(n)]
    ok = check(A, A, A, A, Q, audA, audT, audQ, n=n)
    assert ok == [], ok
    broken = [dict(segments=[dict(audT[0]["segments"][0], source_id="other")])] + audT[1:]
    assert any("not paired" in b for b in check(A, A, A, A, Q, audA, broken, audQ, n=n))
    assert any("trimmed" in b for b in check(A, A, A, A, Q, audA, audA, audQ, n=n))
    assert any("Q is not" in b for b in check(A, A, A, A, A, audA, audT, audQ, n=n))
    longer = [dict(segments=[seg("s", 40)]) for _ in range(n)]
    assert any("duration" in b for b in check(A, A, A, A, Q, longer, longer, audQ, n=n))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real")          # real Communication train manifest (on /group)
    ap.add_argument("--control")       # stitch_scale2/joint_13760.tsv = the adopted training manifest
    ap.add_argument("--orig")          # stitch_scale2/syn_13760.tsv
    ap.add_argument("--corpora")       # dir holding A/ T/ Q/ builder outputs + audit_{A,T,Q}.jsonl
    ap.add_argument("--out")           # dir for joint manifests
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    real, control, orig = rows(a.real), rows(a.control), rows(a.orig)
    c = {k: rows(os.path.join(a.corpora, k, "posestitch_v2_train.tsv")) for k in "ATQ"}
    aud = {k: audit(os.path.join(a.corpora, f"audit_{k}.jsonl")) for k in "ATQ"}
    bad = []
    if control != real + orig:
        bad.append("adopted joint_13760 is not real train + syn_13760")
    with Pool(16) as pool:
        bad += check(real, orig, c["A"], c["T"], c["Q"], aud["A"], aud["T"], aud["Q"], pool=pool)
    if bad:
        print("GATE FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    for k, syn in (("A", orig), ("T", c["T"]), ("Q", c["Q"])):   # control uses the literal adopted rows
        with open(os.path.join(a.out, f"joint_{k}.tsv"), "w", encoding="utf-8") as f:
            f.writelines(f"{p}\t{t}\n" for p, t in real + syn[:13760])
        print(f"joint_{k}.tsv: {len(real) + 13760} rows")
    print("W4_PREP_GATE_PASSED")


if __name__ == "__main__":
    main()
