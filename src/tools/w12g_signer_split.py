#!/usr/bin/env python3
"""Signer-disjoint split of Auslan-Daily Communication, with an exposure control (wave 12g).
Pre-registration: results/PREREG_WAVE12G_SIGNERSPLIT_2026-10-13.md

The spreadsheet column Signer_ID names the signer of every clip. Signers 1 and 2 (the two recurring performers) are
in all 90 videos and hold 92% of the clips; the other 47 signers ("held-out signers") hold 1,087 clips that have pose.

  SD   (signers unseen)   train = clips of signers 1 and 2 outside the validation videos
                          test  = every clip of the held-out signers (folds A + B)
  EXPa (signers exposed)  train = SD train with len(A) random clips replaced by fold A      test = fold B
  EXPb (signers exposed)  train = SD train with len(B) random clips replaced by fold B      test = fold A
  validation (all arms)   = clips of signers 1 and 2 in whole videos that contain NO held-out signer, so the
                            validation set shares no video with any training or test clip and is never tested on

Folds: within each (held-out signer, video) group the clips are shuffled and dealt alternately to A and B, so a
group with two or more clips has clips in both folds (the signer is "exposed" in the same video, as on the official
split). Pooling EXPa's fold-B output with EXPb's fold-A output scores every held-out clip once with its signer in
training; SD scores the same clips with the signer absent. The three training sets have the same number of clips.

Every manifest row is copied verbatim from the official manifests (clip path on /group, subtitle), so no text is
re-tokenised. The trainer builds its vocabulary from the training manifest of each arm (train only).

  python tools/w12g_signer_split.py build --xlsx X --train T --val V --test S --out_dir D   # writes, then checks
  python tools/w12g_signer_split.py check --xlsx X --train T --val V --test S --out_dir D   # re-reads the files
  python tools/w12g_signer_split.py selftest                                                # synthetic, no data
"""
import argparse, collections, json, os, random, re, sys, tempfile

PRINCIPAL = ("1", "2")
VID = re.compile(r"(video_\d+)_")
TOK = re.compile(r"[A-Za-z']+|[.,!?;:]")     # slt/data.py::_word_re on lower-cased text: the trainer's default tokeniser
MANIFESTS = ("SD_train", "SD_test", "EXPa_train", "EXPa_test", "EXPb_train", "EXPb_test", "val")
# Corpus facts of the real data, computed on 13 Oct 2026 from the spreadsheet and the official manifests on Kaya
# (md5 e67f2eb0 / b612245b / f73404a2); `build` and `check` refuse anything else.
REAL = dict(xlsx=14041, xlsx_principal=12953, xlsx_held=1088, pool=14039, principal=12952, held=1087,
            held_signers=47, videos=90, videos_without_held=38, val_videos=6, val=928, SD_train=12024,
            fold_A=544, fold_B=543, held_exposed=1036)


def read_xlsx(path):
    import openpyxl
    it = openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True)
    ix = {h: i for i, h in enumerate(next(it))}
    info = {}
    for r in it:
        clip = r[ix["Video_Clip_Name"]]
        if clip is None:
            continue
        assert clip not in info, f"{clip} twice in the spreadsheet"
        info[str(clip)] = dict(video=str(r[ix["Video_Name"]]).lower(), text=str(r[ix["Subtitle"]]),
                               split=str(r[ix["Split"]]), signer=str(int(r[ix["Signer_ID"]])))
    return info


def read_manifest(path):
    return [l.rstrip("\n") for l in open(path, encoding="utf-8") if "\t" in l]


def clip_of(row):
    return os.path.basename(row.split("\t", 1)[0])[:-len(".npz")]


def video_of(clip):
    return VID.match(clip).group(1)


def vnum(video):
    return int(video.split("_")[1])


def norm(text):
    return " ".join(text.lower().split())


def build(rows, info, val_min=800):
    """rows = official train + val + test manifest rows. Returns ({manifest name: rows}, {clip: sidecar fields})."""
    clips = [clip_of(r) for r in rows]
    row = dict(zip(clips, rows))
    assert len(row) == len(rows), "a clip occurs twice in the pool"
    for c, r in row.items():
        assert c in info, f"{c} is not in the spreadsheet"
        assert norm(r.split("\t", 1)[1]) == norm(info[c]["text"]), f"{c}: manifest text differs from the spreadsheet"
        assert video_of(c) == info[c]["video"], f"{c}: video name differs from the spreadsheet"
    signer = {c: info[c]["signer"] for c in clips}
    held = [c for c in clips if signer[c] not in PRINCIPAL]
    prin = [c for c in clips if signer[c] in PRINCIPAL]

    # validation: whole videos without any held-out signer, in a fixed random order, until val_min clips
    free = sorted({video_of(c) for c in prin} - {video_of(c) for c in held}, key=vnum)
    per_video = collections.Counter(video_of(c) for c in prin)
    random.Random("w12g:val").shuffle(free)
    val_videos, n = set(), 0
    for v in free:
        if n >= val_min:
            break
        val_videos.add(v); n += per_video[v]
    assert n >= val_min, "not enough videos without held-out signers for the validation set"
    val = [c for c in prin if video_of(c) in val_videos]
    train = [c for c in prin if video_of(c) not in val_videos]

    # folds: deal each (signer, video) group alternately; `start` carries over so the folds stay balanced
    groups = collections.defaultdict(list)
    for c in held:
        groups[(int(signer[c]), vnum(video_of(c)))].append(c)
    rng, fold, start = random.Random("w12g:fold"), {}, 0
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        for i, c in enumerate(g):
            fold[c] = "AB"[(i + start) % 2]
        start = (start + len(g)) % 2
    A = [c for c in held if fold[c] == "A"]
    B = [c for c in held if fold[c] == "B"]

    def exposed(add, tag):          # same size as SD train: len(add) random training clips make room for the fold
        drop = set(random.Random(f"w12g:drop:{tag}").sample(train, len(add)))
        return [c for c in train if c not in drop] + add

    parts = {"SD_train": train, "SD_test": held, "EXPa_train": exposed(A, "a"), "EXPa_test": B,
             "EXPb_train": exposed(B, "b"), "EXPb_test": A, "val": val}
    part = {**{c: "train" for c in train}, **{c: "val" for c in val}, **fold}
    side = {c: dict(signer=signer[c], video=video_of(c), part=part[c], official=info[c]["split"],
                    exposed=int(c in fold and len({fold[x] for x in groups[(int(signer[c]), vnum(video_of(c)))]}) == 2))
            for c in clips}
    return {k: [row[c] for c in v] for k, v in parts.items()}, side


def write(out_dir, parts, side):
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in parts.items():
        with open(os.path.join(out_dir, f"w12g_{name}.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    with open(os.path.join(out_dir, "w12g_clips.tsv"), "w", encoding="utf-8") as f:
        f.write("clip\tsigner\tvideo\tpart\tofficial\texposed\n")
        for c, s in side.items():
            f.write(f"{c}\t{s['signer']}\t{s['video']}\t{s['part']}\t{s['official']}\t{s['exposed']}\n")


def read_side(out_dir):
    lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(out_dir, "w12g_clips.tsv"), encoding="utf-8")]
    return {l[0]: dict(zip(lines[0][1:], l[1:])) for l in lines[1:]}


def check(out_dir, rows, info):
    """Re-read the written files and verify the split from them alone. Returns the counts."""
    pool = set(rows)
    m = {n: read_manifest(os.path.join(out_dir, f"w12g_{n}.tsv")) for n in MANIFESTS}
    C = {n: [clip_of(r) for r in v] for n, v in m.items()}
    sig = lambda c: info[c]["signer"]
    for n in MANIFESTS:
        assert set(m[n]) <= pool, f"{n}: a row is not a verbatim row of the official manifests"
        assert len(set(C[n])) == len(C[n]), f"{n}: a clip occurs twice"
        assert all(r.startswith("/group/") for r in m[n]), f"{n}: a clip path is not on /group"
    held = {clip_of(r) for r in rows if sig(clip_of(r)) not in PRINCIPAL}
    val, sd_train = set(C["val"]), set(C["SD_train"])
    assert all(sig(c) in PRINCIPAL for c in val | sd_train), "a held-out signer is in SD train or in validation"
    assert set(C["SD_test"]) == held, "SD test is not exactly the clips of the held-out signers"
    assert val | sd_train == {clip_of(r) for r in rows} - held, \
        "SD train and validation together are not exactly the clips of signers 1 and 2"
    for arm in ("SD", "EXPa", "EXPb"):
        tr, te = set(C[f"{arm}_train"]), set(C[f"{arm}_test"])
        assert not tr & te and not tr & val and not te & val, f"{arm}: a clip is in two parts"
        assert not {video_of(c) for c in val} & {video_of(c) for c in tr | te}, \
            f"{arm}: a validation video also has clips in train or test"
        assert all(sig(c) not in PRINCIPAL for c in te), f"{arm}: a clip of signer 1 or 2 is in test"
        assert len(tr) == len(sd_train), f"{arm}: training size differs from SD"
    A, B = set(C["EXPb_test"]), set(C["EXPa_test"])
    assert A | B == held and not A & B, "the folds do not partition the held-out clips"
    assert abs(len(A) - len(B)) <= 1, "the folds are not balanced"
    for arm, fold in (("EXPa", A), ("EXPb", B)):
        tr = set(C[f"{arm}_train"])
        assert fold <= tr and tr - fold <= sd_train, f"{arm}: training is not SD train with clips replaced by its fold"
    groups = collections.defaultdict(set)
    for c in held:
        groups[(sig(c), video_of(c))].add(c)
    side = read_side(out_dir)
    assert set(side) == {clip_of(r) for r in rows}, "the sidecar does not list exactly the pool"
    for g in groups.values():
        both = bool(g & A) and bool(g & B)
        assert both == (len(g) >= 2), "a (signer, video) group with two or more clips is in one fold only"
        assert all(side[c]["exposed"] == str(int(both)) and side[c]["part"] == ("A" if c in A else "B") for c in g), \
            "the sidecar's fold or exposure flag differs from the manifests"
    assert all(side[c]["signer"] == sig(c) for c in side), "sidecar signer differs from the spreadsheet"
    return dict(pool=len(rows), principal=len(rows) - len(held), held=len(held),
                held_signers=len({sig(c) for c in held}), videos=len({video_of(clip_of(r)) for r in rows}),
                videos_without_held=len({video_of(clip_of(r)) for r in rows} - {video_of(c) for c in held}),
                val_videos=len({video_of(c) for c in val}), val=len(val), SD_train=len(sd_train),
                fold_A=len(A), fold_B=len(B), held_exposed=sum(len(g) for g in groups.values() if len(g) >= 2))


def assert_real(counts, info):
    x = collections.Counter("principal" if v["signer"] in PRINCIPAL else "held" for v in info.values())
    got = dict(counts, xlsx=len(info), xlsx_principal=x["principal"], xlsx_held=x["held"])
    bad = {k: (got[k], v) for k, v in REAL.items() if got[k] != v}
    assert not bad, f"counts differ from the recorded corpus facts (got, expected): {bad}"


def report(out_dir, info):
    """Corpus facts for the pre-registration: sizes, target overlap and vocabulary cover. No model output."""
    m = {n: read_manifest(os.path.join(out_dir, f"w12g_{n}.tsv")) for n in MANIFESTS}
    text = lambda rows: [r.split("\t", 1)[1] for r in rows]
    T = {n: {norm(t) for t in text(m[n])} for n in ("SD_train", "EXPa_train", "EXPb_train")}
    tok = lambda t: TOK.findall(t.lower().strip())
    V = {n: {w for t in text(m[n]) for w in tok(t)} for n in ("SD_train", "EXPa_train", "EXPb_train")}
    exp_of = {clip_of(r): "EXPa_train" for r in m["EXPa_test"]} | {clip_of(r): "EXPb_train" for r in m["EXPb_test"]}
    test = m["SD_test"]
    seen_sd = [norm(t) in T["SD_train"] for t in text(test)]
    seen_ex = [norm(r.split("\t", 1)[1]) in T[exp_of[clip_of(r)]] for r in test]
    toks = [(w, clip_of(r)) for r in test for w in tok(r.split("\t", 1)[1])]
    by_signer = collections.Counter(info[clip_of(r)]["signer"] for r in test)
    out = dict(
        sizes={n: len(m[n]) for n in MANIFESTS},
        validation_videos=sorted({video_of(clip_of(r)) for r in m["val"]}, key=vnum),
        test_clips_by_signer=dict(by_signer.most_common(5), others=sum(by_signer.values()) - sum(
            n for _, n in by_signer.most_common(5))),
        test_seen_in_SD_train=sum(seen_sd), test_unseen_under_both=sum(not a and not b for a, b in zip(seen_sd, seen_ex)),
        test_seen_only_through_the_other_fold=sum(b and not a for a, b in zip(seen_sd, seen_ex)),
        test_distinct_targets=len({norm(t) for t in text(test)}),
        val_seen_in_SD_train=sum(norm(t) in T["SD_train"] for t in text(m["val"])),
        vocabulary_word_types={n: len(V[n]) for n in V},
        test_tokens=len(toks), test_tokens_outside_SD_vocabulary=sum(w not in V["SD_train"] for w, _ in toks),
        test_tokens_outside_exposed_vocabulary=sum(w not in V[exp_of[c]] for w, c in toks),
        mean_reference_tokens=dict(test=round(sum(len(t.split()) for t in text(test)) / len(test), 2),
                                   val=round(sum(len(t.split()) for t in text(m["val"])) / len(m["val"]), 2),
                                   SD_train=round(sum(len(t.split()) for t in text(m["SD_train"])) / len(m["SD_train"]), 2)))
    json.dump(out, open(os.path.join(out_dir, "w12g_report.json"), "w"), indent=1)
    return out


def synthetic():
    """A small corpus with the real one's structure: 12 videos of signers 1 and 2, held-out signers in videos 1-6."""
    info, rows = {}, []
    def add(video, k, signer):
        c = f"video_{video}_{k}"
        info[c] = dict(video=f"video_{video}", text=f"sentence {video} {k % 7} .", split="train", signer=str(signer))
        rows.append(f"/group/x/{c}.npz\tsentence {video} {k % 7} .")
    for v in range(1, 13):
        for k in range(30):
            add(v, k, 1 + k % 2)
    for v, s, n in ((1, 10, 9), (2, 10, 4), (3, 10, 1), (3, 8, 5), (4, 8, 2), (5, 30, 1), (6, 31, 3), (6, 32, 1)):
        for k in range(n):
            add(v, 100 * s + k, s)
    random.Random(0).shuffle(rows)
    return rows, info


def selftest():
    """Synthetic corpus: the build passes its own check, is deterministic, and the check catches injected faults."""
    rows, info = synthetic()
    parts, side = build(rows, info, val_min=50)
    assert build(rows, info, val_min=50)[0] == parts, "build is not deterministic"
    with tempfile.TemporaryDirectory() as d:
        write(d, parts, side)
        c = check(d, rows, info)
        assert (c["held"], c["held_signers"], c["val_videos"], c["val"], c["SD_train"]) == (26, 5, 2, 60, 300), c
        assert (c["fold_A"], c["fold_B"], c["held_exposed"]) == (13, 13, 23), c     # 3 one-clip groups
        assert {video_of(clip_of(r)) for r in parts["val"]} <= {f"video_{v}" for v in range(7, 13)}
        rep = report(d, info)
        assert rep["sizes"]["EXPa_train"] == rep["sizes"]["EXPb_train"] == 300 and rep["test_tokens"] == 26 * 2   # the trainer's tokeniser drops digits
        good = {n: read_manifest(os.path.join(d, f"w12g_{n}.tsv")) for n in MANIFESTS}
        prin_train, val_row, b_row = good["SD_train"][0], good["val"][0], good["EXPa_test"][0]
        off_group = "/scratch" + val_row[len("/group"):]
        faults = {                                    # what is broken: ({file: rows}, the gate that must fire)
            "principal clip in test": ({"SD_test": good["SD_test"] + [prin_train]}, "SD test is not exactly"),
            "held-out clip in SD train": ({"SD_train": good["SD_train"][1:] + [good["SD_test"][0]]}, "held-out signer is in SD train"),
            "principal clip dropped": ({"SD_train": good["SD_train"][1:]}, "together are not exactly"),
            "validation clip in train": ({"EXPa_train": good["EXPa_train"][1:] + [val_row]}, "a clip is in two parts"),
            "test clip of the arm in its train": ({"EXPa_train": good["EXPa_train"][1:] + [b_row]}, "a clip is in two parts"),
            "row not verbatim": ({"SD_test": [good["SD_test"][0] + " x"] + good["SD_test"][1:]}, "not a verbatim row"),
            "duplicate row": ({"val": good["val"] + [val_row]}, "occurs twice"),
            "path off /group": ({"val": [off_group] + good["val"][1:]}, "not on /group"),
            "training size differs": ({"EXPb_train": good["EXPb_train"][1:]}, "training size differs"),
            "fold missing from its arm": ({"EXPb_test": good["EXPb_test"][1:]}, "do not partition"),
            "validation video also trained on": ({"val": good["val"][1:], "SD_train": good["SD_train"] + [val_row],
                                                  "EXPa_train": good["EXPa_train"] + [val_row],
                                                  "EXPb_train": good["EXPb_train"] + [val_row]}, "validation video also has clips"),
        }
        for what, (bad, gate) in faults.items():
            for name, bad_rows in bad.items():
                open(os.path.join(d, f"w12g_{name}.tsv"), "w").write("\n".join(bad_rows) + "\n")
            try:
                check(d, rows + [off_group], info)
            except AssertionError as e:
                assert gate in str(e), f"selftest FAILED: '{what}' was caught by the wrong gate: {e}"
            else:
                raise SystemExit(f"selftest FAILED: the check did not catch: {what}")
            for name in bad:
                open(os.path.join(d, f"w12g_{name}.tsv"), "w").write("\n".join(good[name]) + "\n")
        check(d, rows, info)
    # a video that gains a held-out signer may no longer be used for validation
    v = video_of(clip_of(parts["val"][0]))
    first = next(c for c in info if info[c]["video"] == v)
    p2, _ = build(rows, {**info, first: dict(info[first], signer="77")}, val_min=50)
    assert v not in {video_of(clip_of(r)) for r in p2["val"]}, "a video with a held-out signer was used for validation"
    print(f"selftest OK: build deterministic, check passes, {len(faults)} injected faults each caught by its gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check", "selftest"])
    for k in ("xlsx", "train", "val", "test", "out_dir"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    info = read_xlsx(a.xlsx)
    rows = read_manifest(a.train) + read_manifest(a.val) + read_manifest(a.test)
    if a.mode == "build":
        write(a.out_dir, *build(rows, info))
    counts = check(a.out_dir, rows, info)
    assert_real(counts, info)
    print(f"[w12g] {a.mode} OK:", json.dumps(counts))
    if a.mode == "build":
        print("[w12g] report:", json.dumps(report(a.out_dir, info)))


if __name__ == "__main__":
    sys.exit(main())
