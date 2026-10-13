"""Wave-12b corpus gate (results/PREREG_WAVE12B_WRONGSENSE_2026-10-13.md). Exits non-zero on ANY violation, so the
dependent training array never starts.

  A  rebuild of the adopted corpus with the edited builder, flag off -> must reproduce stitch_scale2/syn_13760
     array-for-array (not by file md5: np.savez_compressed stamps zip mtimes). The control arm then trains on the
     literal adopted rows (stitch_scale2/joint_13760.tsv), as in wave 4.
  W  the same build with --drop_wrong_sense -> same English row for row; every clip is A's clip minus the signs whose
     gloss is in stitch_lexicon.WRONG_SENSE_GLOSSES: the signs that stay are the same exemplar, the same length and the
     same frames; a transition keeps the budget drawn for the sign it precedes and blends the new neighbours; a clip
     with no wrong-sense sign (or with nothing else) is A's clip unchanged.
  counts  the numbers measured on the adopted corpus before any run (PIN below) must come out again.
  leaks   no synthetic sentence equals a real-train or validation text; the only test match is the one known since
          16 Sep (shared by both arms, pinned by digest).

  python tools/w12b_prep_check.py selftest                      the checks against deliberately broken inputs
  python tools/w12b_prep_check.py lexicon Train.json Valid.json the 14 surface forms, under the adopted lexicon flags
  python tools/w12b_prep_check.py plan AUDIT_A.jsonl            counts from A's audit sidecar alone (no pose needed)
  python tools/w12b_prep_check.py gate --real .. --val .. --test .. --control .. --orig .. --A DIR --W DIR --out DIR
Each DIR holds posestitch_v2_train.tsv and audit.jsonl (written by build_posestitch_corpus_v2.py --audit_jsonl).
"""
import argparse, collections, hashlib, json, os, sys, tempfile
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_lexicon import WRONG_SENSE_GLOSSES as WS, build_lexicon  # noqa: E402

SAME = ("gloss", "source_dir", "source_id", "output_frames", "source_frames_original", "source_frames_after_trim")
LAYOUT = ("transition_frames_before", "start_frame", "end_frame")
FORMS = {"be", "before that", "believe", "bother", "can", "enough", "good", "know", "matter", "mind", "prior to",
         "responsible", "want", "yet"}
# Measured on 13 Oct 2026 from results/stitched_corpus_audit/audit_A.jsonl (the audit sidecar of wave-4 corpus A, which
# job 7948 showed array-identical to the adopted corpus). "removed" excludes the one clip whose three signs are all
# wrong-sense; "short" = clips left with fewer than 3 signs.
PIN = dict(clips=13760, signs=94428, wrong=3257, removed=3254, changed=2889, whole=1, short=70,
           per_gloss={"be (not be)": 537, "believe (not believe)": 20, "bother (no bother)": 3, "can (tin)": 991,
                      "enough (not enough)": 42, "good (no good)": 770, "know (not know)": 412,
                      "matter (not matter)": 13, "mind (not mind)": 30, "want (not want)": 321, "yet (not yet)": 118})
TEST_MATCH = {"b0577ddd157ebeaebdb5d872d22e8c3c8d97c3e0"}   # sha1 of the one synthetic sentence equal to a test reference


def rows(path):
    return [l.rstrip("\n").split("\t", 1) for l in open(path, encoding="utf-8") if "\t" in l]


def audit(d):
    return [json.loads(l) for l in open(os.path.join(d, "audit.jsonl"), encoding="utf-8")]


def norm(t):
    return " ".join(t.lower().split())


def kept(segs):
    """The signs W renders for a clip whose A segments are `segs` (the builder's rule)."""
    return [s for s in segs if s["gloss"] not in WS] or segs


def plan(aud):
    c = collections.Counter(s["gloss"] for a in aud for s in a["segments"] if s["gloss"] in WS)
    k = [(len(a["segments"]), len(kept(a["segments"]))) for a in aud]
    return dict(clips=len(aud), signs=sum(n for n, _ in k), wrong=sum(c.values()), removed=sum(n - m for n, m in k),
                changed=sum(n != m for n, m in k), short=sum(m < 3 for n, m in k),
                whole=sum(all(s["gloss"] in WS for s in a["segments"]) for a in aud), per_gloss=dict(sorted(c.items())))


def same_arrays(job):
    a, b = np.load(job[0]), np.load(job[1])
    return np.array_equal(a["KPS"], b["KPS"]) and np.array_equal(a["BBOX"], b["BBOX"])


def pair_clip(job):
    """'' if W's clip is A's clip minus its wrong-sense signs, else what is wrong."""
    pa, pw, sa, sw = job
    want = kept(sa)
    if len(sw) != len(want) or any(s[k] != r[k] for s, r in zip(sw, want) for k in SAME):
        return f"{pw}: signs are not A's signs minus the wrong-sense ones"
    za, zw = np.load(pa), np.load(pw)
    ka, ba, kw, bw = za["KPS"], za["BBOX"], zw["KPS"], zw["BBOX"]
    if len(want) == len(sa):                              # nothing removed: the clip must be A's clip
        ok = np.array_equal(ka, kw) and np.array_equal(ba, bw) and all(s[k] == r[k] for s, r in zip(sw, sa) for k in LAYOUT)
        return "" if ok else f"{pw}: differs from A although no sign was removed"
    prev = -1
    for j, (s, r) in enumerate(zip(sw, want)):
        n_t = s["transition_frames_before"]
        if n_t != (r["transition_frames_before"] if j else 0):
            return f"{pw}: sign {j} does not keep its own transition budget"
        w0, w1, a0, a1 = s["start_frame"], s["end_frame"] + 1, r["start_frame"], r["end_frame"] + 1
        if w0 != prev + 1 + n_t:
            return f"{pw}: segments not contiguous"
        if not (np.array_equal(ka[a0:a1], kw[w0:w1]) and np.array_equal(ba[a0:a1], bw[w0:w1])):
            return f"{pw}: frames of sign {j} differ from A"
        if n_t:                                           # blend of the laid-out neighbours, box held from the previous sign
            w = np.linspace(0, 1, n_t + 2)[1:-1][:, None, None]
            blend = (1 - w) * kw[prev][None] + w * kw[w0][None]
            if not (np.allclose(kw[prev + 1:w0], blend, atol=1e-5)
                    and np.array_equal(bw[prev + 1:w0], np.repeat(bw[prev][None], n_t, 0))):
                return f"{pw}: transition before sign {j} is not the blend of its neighbours"
        prev = s["end_frame"]
    return "" if prev + 1 == kw.shape[0] else f"{pw}: frames after the last sign"


def check(A, W, audA, audW, orig, pool, pin=PIN):
    """A, W, orig: manifest rows; audA, audW: audit records. Returns the list of violations."""
    bad = []
    txt = lambda r: [t for _, t in r]
    n = pin["clips"]
    if not (len(A) == len(W) == len(audA) == len(audW) == n and len(orig) >= n):
        return [f"sizes A {len(A)} W {len(W)} audits {len(audA)}/{len(audW)} adopted {len(orig)} (want {n})"]
    orig = orig[:n]
    if txt(A) != txt(orig):
        bad.append("A's English rows do not reproduce the adopted corpus")
    if txt(W) != txt(A) or txt(A) != [a["text"] for a in audA] or txt(W) != [a["text"] for a in audW]:
        bad.append("W's English rows differ from A's (or an audit sidecar is not row-aligned)")
    if {p for p, _ in A} & {p for p, _ in W} or {p for p, _ in W} & {p for p, _ in orig}:
        bad.append("W shares clip files with A or with the adopted corpus")
    if bad:
        return bad
    diff = n - sum(pool.map(same_arrays, [(p, q) for (p, _), (q, _) in zip(A, orig)], chunksize=64))
    print(f"A vs adopted corpus: {n - diff}/{n} clips array-identical")
    if diff:
        bad.append(f"A differs from the adopted corpus in {diff} clips (the edited builder is not inert with the flag off)")
    errs = [e for e in pool.map(pair_clip, [(pa, pw, a["segments"], w["segments"])
                                            for (pa, _), (pw, _), a, w in zip(A, W, audA, audW)], chunksize=64) if e]
    print(f"W vs A: {n - len(errs)}/{n} clips are A's clip minus its wrong-sense signs")
    bad += errs[:5] + ([f"... and {len(errs) - 5} more clips"] if len(errs) > 5 else [])
    got = plan(audA)
    left = sum(s["gloss"] in WS for w in audW for s in w["segments"])
    n_w = sum(len(w["segments"]) for w in audW)
    print(f"A: {got}\nW: {n_w} signs, {left} wrong-sense signs left (in the {got['whole']} clip(s) that have no other sign)")
    if got != pin:
        bad.append(f"counts differ from the pre-registered ones: got {got}, pinned {pin}")
    if n_w != got["signs"] - got["removed"] or left != got["wrong"] - got["removed"]:
        bad.append(f"W has {n_w} signs / {left} wrong-sense left; expected {got['signs'] - got['removed']} / "
                   f"{got['wrong'] - got['removed']}")
    return bad


def leaks(syn, real, val, test):
    s = {norm(t) for _, t in syn}
    sha = lambda r: {hashlib.sha1(x.encode()).hexdigest() for x in s & {norm(t) for _, t in r}}
    bad = []
    if sha(real) or sha(val):
        bad.append(f"synthetic sentences equal to real-train ({len(sha(real))}) or validation ({len(sha(val))}) texts")
    if sha(test) != TEST_MATCH:
        bad.append(f"test matches {sorted(sha(test))} are not the one disclosed match")
    return bad


def lexicon(labels):
    lex = build_lexicon(labels, morphology=True, paren_alias=False)
    forms = {f for f, g in lex.phrase2gloss.items() if g in WS}
    assert len(lex.phrase2gloss) == 3090, f"{len(lex.phrase2gloss)} surface forms, the adopted lexicon has 3090"
    assert len(WS) == 14 and WS <= set(lex.gloss2clips), sorted(WS - set(lex.gloss2clips))
    assert forms == FORMS, f"surface forms that reach a wrong-sense gloss: {sorted(forms ^ FORMS)} differ from the 14"
    print(f"LEXICON OK: 3090 surface forms; exactly 14 reach a wrong-sense gloss: {', '.join(sorted(forms))}")


# ---------------------------------------------------------------- self-test on a tiny hand-made corpus

def _lay(signs, budgets):
    """Lay signs [(gloss, id, frames array)] out as the builder does; returns (KPS, BBOX, segments)."""
    kp, bb, segs, cur = [], [], [], 0
    for j, ((g, sid, k), n_t) in enumerate(zip(signs, budgets)):
        b = k[:, 0, :2].repeat(2, 1)
        t = n_t if kp else 0
        if t:
            w = np.linspace(0, 1, t + 2)[1:-1][:, None, None]
            kp.append((1 - w) * kp[-1][-1][None] + w * k[0][None]); bb.append(np.repeat(bb[-1][-1][None], t, 0))
            cur += t
        segs.append(dict(gloss=g, source_dir="d", source_id=sid, output_frames=len(k), source_frames_original=80,
                         source_frames_after_trim=56, transition_frames_before=t, start_frame=cur,
                         end_frame=cur + len(k) - 1))
        cur += len(k); kp.append(k); bb.append(b)
    return np.concatenate(kp).astype(np.float32), np.concatenate(bb).astype(np.float32), segs


def _corpus(d, name, clips, drop):
    os.makedirs(os.path.join(d, name), exist_ok=True)
    out, aud = [], []
    for i, (text, signs, budgets) in enumerate(clips):
        if drop:
            keep = [j for j, s in enumerate(signs) if s[0] not in WS]
            if keep:
                signs, budgets = [signs[j] for j in keep], [budgets[j] for j in keep]
        K, B, segs = _lay(signs, budgets)
        p = os.path.join(d, name, f"c{i}.npz")
        np.savez(p, KPS=K, BBOX=B)
        out.append([p, text]); aud.append(dict(text=text, segments=segs))
    return out, aud


def selftest():
    r = np.random.default_rng(0)
    sg = lambda g, n: (g, g + str(n), r.normal(size=(n, 5, 3)).astype(np.float32))
    ws, ok = "good (no good)", "hello"
    clips = [("t0 .", [sg(ok, 6), sg(ws, 5), sg("yes", 7)], [0, 2, 3]),       # wrong sign in the middle
             ("t1 .", [sg(ws, 4), sg(ok, 6), sg("yes", 5), sg("can (tin)", 4)], [0, 1, 0, 2]),   # first and last
             ("t2 .", [sg(ok, 6), sg("yes", 5), sg("no", 4)], [0, 4, 1]),      # nothing to remove
             ("t3 .", [sg(ws, 4), sg(ws, 5), sg(ws, 6)], [0, 2, 2])]           # all wrong-sense: left as built
    pin = dict(clips=4, signs=13, wrong=6, removed=3, changed=2, whole=1, short=2,
               per_gloss={"can (tin)": 1, ws: 5})
    with tempfile.TemporaryDirectory() as d, Pool(2) as pool:
        O, _ = _corpus(d, "O", clips, False)
        A, aA = _corpus(d, "A", clips, False)
        W, aW = _corpus(d, "W", clips, True)
        U, aU = _corpus(d, "U", clips, False)             # a "W" that removed nothing
        assert plan(aA) == pin, plan(aA)
        assert check(A, W, aA, aW, O, pool, pin) == [], check(A, W, aA, aW, O, pool, pin)
        has = lambda sub, *a, p=pin: any(sub in b for b in check(*a, pool, p))

        def broken(path, sub, *a):                        # shift one clip's pose, expect `sub`, restore the clip
            z = np.load(path); k, b = z["KPS"], z["BBOX"]
            np.savez(path, KPS=k + 1, BBOX=b)
            assert has(sub, *a), sub
            np.savez(path, KPS=k, BBOX=b)

        assert has("minus the wrong-sense", A, U, aA, aU, O), "a W that keeps the wrong signs must fail"
        assert has("English rows differ", A, [W[0], [W[1][0], "other ."]] + W[2:], aA, aW, O)
        assert has("shares clip files", A, A, aA, aA, O)
        broken(W[0][0], "frames of sign 0 differ", A, W, aA, aW, O)
        broken(W[2][0], "although no sign was removed", A, W, aA, aW, O)
        broken(W[3][0], "although no sign was removed", A, W, aA, aW, O)      # the all-wrong clip must be A's clip
        broken(O[1][0], "not inert", A, W, aA, aW, O)
        bw = json.loads(json.dumps(aW)); bw[0]["segments"][1]["transition_frames_before"] = 1
        assert has("own transition budget", A, W, aA, bw, O)
        assert has("counts differ", A, W, aA, aW, O, p=dict(pin, removed=4))
    syn = [["p", "a b ."], ["q", "c d ."]]
    assert any("test matches" in b for b in leaks(syn, [], [], []))
    assert any("real-train" in b for b in leaks(syn, [["x", "A  b ."]], [], []))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "lexicon", "plan", "gate"])
    ap.add_argument("paths", nargs="*")
    for k in ("real", "val", "test", "control", "orig", "A", "W", "out"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "lexicon":
        return lexicon(a.paths)
    if a.mode == "plan":
        got = plan([json.loads(l) for l in open(a.paths[0], encoding="utf-8")])
        print(got)
        sys.exit(0 if got == PIN else 1)
    real, val, test, control, orig = (rows(p) for p in (a.real, a.val, a.test, a.control, a.orig))
    A, W = (rows(os.path.join(d, "posestitch_v2_train.tsv")) for d in (a.A, a.W))
    bad = []
    clip = lambda r: {os.path.basename(p) for p, _ in r}
    if (len(real), len(val), len(test)) != (12439, 800, 800) or clip(val) & clip(test) or clip(real) & (clip(val) | clip(test)):
        bad.append(f"real manifests: {len(real)}/{len(val)}/{len(test)} rows or a clip shared between splits")
    if control != real + orig[:PIN["clips"]]:
        bad.append("the adopted joint manifest is not real train + the adopted synthetic rows")
    bad += leaks(W, real, val, test)
    with Pool(min(16, os.cpu_count() or 1)) as pool:
        bad += check(A, W, audit(a.A), audit(a.W), orig, pool)
    if bad:
        print("GATE FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "joint_W.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{p}\t{t}\n" for p, t in real + W)
    print(f"joint_W.tsv: {len(real) + len(W)} rows (control = the adopted manifest, unchanged)\nW12B_PREP_GATE_PASSED")


if __name__ == "__main__":
    main()
