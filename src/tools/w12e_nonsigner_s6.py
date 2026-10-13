"""Wave 12e: does the confidence gate of the REPORTED six-member ensemble (S6, beam 10, LP 0.3) decline people who are
not signing? Pre-registration: results/PREREG_WAVE12E_NONSIGNER_ENSEMBLE_2026-10-13.md, a dated addendum to
results/PREREG_NONSIGNER_GATE_2026-09-22.md (its Stage 2 / amendment B13).

Kaya stages (called by jobs/slurm/w12e_extract.sbatch and jobs/slurm/w12e_decode.sbatch; numpy + the decode code tree only):
  manifest   the 1,678 dev+test negatives as a decode manifest (text = crop id), the symlink lists for the feature
             extractor, and md5 / T / valid-T of every pose file that is decoded
  featcheck  every manifest row has a valid rgb_<clip>.npz (RGBDataset silently substitutes ONE ZERO FRAME otherwise)
  rgbgate    features re-extracted today for signer clips equal the stored August features
  dumpgate   a fresh n-best dump reproduces the recorded BLEU-4 / WER and, row by row, a saved dump
Local stages (need the dataset tables; statistics are tools/nonsigner_gate.analyse() itself, unchanged):
  build      the ensemble's decodes in nonsigner_gate's format (ce = -confidence) after every gate and provenance check
  analyse    single-model record reproduced first, then the ensemble analysis and the ensemble-vs-single comparison
  selftest

Confidence of an ensemble output = raw / (words + 1) of the hypothesis selected at LP 0.3 (tools/ensemble_selective.py
::load); a clip is voiced when confidence >= quantile(validation confidence, 1 - q). With ce = -confidence this is
ce <= quantile(ce_val, q), the single-model rule, so one function serves both.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# On Kaya W12E_CODE = the frozen decode snapshot, so the gates use the very modules the decode used.
sys.path[:0] = [os.environ.get("W12E_CODE") or str(HERE.parent), str(HERE)]
from slt.metrics import corpus_bleu4, corpus_wer, detokenize      # noqa: E402
from slt.rgb_fusion import _rgb_path                              # noqa: E402  (the dataset's own file-name rule)

ROOT = HERE.parents[1]
RES = ROOT / "results"
S6 = RES / "nonsigner_s6"
LP, QS, RGB_DIM, FRAMES = 0.3, (0.50, 0.75), 1152, 24
RECORDED = {"val": (12.470, 0.9164), "test": (11.802, 0.9069)}     # BLEU-4, WER of S6 at beam 10 / LP 0.3
REC_COVERAGE = {0.5: 0.51875, 0.75: 0.75625}                       # results/enssel_S6.json, test, thresholds from val
SETS = ("sig_val", "sig_test", "trk_val", "trk_test", "neg")


def read_tsv(p):
    return [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if "\t" in l]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def valid_T(kps):
    return int((np.abs(kps).reshape(len(kps), -1).sum(1) > 1e-6).sum())      # = nonsigner_gate.valid_T (asserted in build)


def die(msg):
    print("FAIL:", msg, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------- n-best dumps
def nbest_rows(dump):
    return [json.loads(l)["nbest"] for l in open(Path(dump) / "nbest.jsonl", encoding="utf-8")]


def select(dump, lp=LP):
    """Per row: the hypothesis ensemble_mbr_wer.py reports at this LP (argmax raw / words**lp) and its confidence."""
    out = []
    for l in open(Path(dump) / "nbest.jsonl", encoding="utf-8"):
        r = json.loads(l)
        raw, h = max(r["nbest"], key=lambda c: c[0] / (max(1, len(c[1].split())) ** lp))
        out.append({"ref": r["ref"], "hyp": h, "conf": raw / (len(h.split()) + 1)})
    return out


def scores(sel):
    h, r = [detokenize(x["hyp"]) for x in sel], [detokenize(x["ref"]) for x in sel]
    return corpus_bleu4(h, r, tokenize="13a"), corpus_wer(h, r)


def compare_dumps(new, saved, min_same):
    """Row-level agreement of two decodes of the same manifest. Returns a dict with 'ok'."""
    a, b = select(new), select(saved)
    if len(a) != len(b) or any(x["ref"] != y["ref"] for x, y in zip(a, b)):
        return {"ok": False, "why": "row count or references differ"}
    same = [x["hyp"] == y["hyp"] for x, y in zip(a, b)]
    d = np.array([abs(x["conf"] - y["conf"]) for x, y, s in zip(a, b, same) if s])
    out = {"n": len(a), "same_hyp": int(sum(same)), "median_abs_dconf": float(np.median(d)) if len(d) else None,
           "max_abs_dconf": float(d.max()) if len(d) else None,
           "differing_rows": [i for i, s in enumerate(same) if not s][:50]}
    # The September S6 gate as worded: the first 50 rows reproduce the saved RAW n-best scores to within 1e-4.
    # A row fails when its candidate strings differ or a raw score moved by 1e-4 or more; as many rows may fail
    # as may differ in the selected hypothesis (n - min_same).
    bad50, d50 = 0, []
    for x, y in zip(nbest_rows(new)[:50], nbest_rows(saved)[:50]):
        if [h for _, h in x] != [h for _, h in y]:
            bad50 += 1; continue
        d50.append(max(abs(p[0] - q[0]) for p, q in zip(x, y)))
        bad50 += d50[-1] >= 1e-4
    out["first50_rows_not_reproduced"], out["first50_max_abs_draw"] = int(bad50), (float(max(d50)) if d50 else None)
    out["ok"] = bool(out["same_hyp"] >= min_same and len(d) and out["max_abs_dconf"] < 1e-4
                     and bad50 <= len(a) - min_same)
    return out


def dumpgate(new, bleu, wer, saved=None, n=800, min_same=798):
    sel = select(new)
    b, w = scores(sel)
    rep = {"dump": str(new), "n": len(sel), "bleu4": b, "wer": w, "recorded": [bleu, wer],
           "ok": bool(len(sel) == n and abs(b - bleu) < 1e-3 and abs(w - wer) < 1e-4)}
    lpfile = Path(new) / f"hyps_LP={LP}.jsonl"           # the tool's own LP row must equal the re-selection
    if lpfile.exists():
        own = [json.loads(l)["hyp_beam"] for l in open(lpfile, encoding="utf-8")]
        rep["reselect_equals_tool"] = own == [x["hyp"] for x in sel]
        rep["ok"] = rep["ok"] and rep["reselect_equals_tool"]
    if saved:
        rep["vs_saved"] = compare_dumps(new, saved, min_same)
        rep["ok"] = rep["ok"] and rep["vs_saved"]["ok"]
    return rep


# ---------------------------------------------------------------- Kaya: manifest, feature checks
def stage_manifest(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = lambda p: Path(p).stem
    man = {k: read_tsv(p) for k, p in (("sig_val", a.val_manifest), ("sig_test", a.test_manifest),
                                         ("trk_val", a.trk_val), ("trk_test", a.trk_test))}
    ids = {k: [stem(r[0]) for r in v] for k, v in man.items()}
    for k in ("sig_val", "sig_test"):
        assert len(ids[k]) == 800 == len(set(ids[k])), f"{k}: {len(ids[k])} rows"
        t = k.replace("sig", "trk")           # the tracked manifest must be the same clips, order and text
        assert [(stem(p), x) for p, x in man[t]] == [(stem(p), x) for p, x in man[k]], f"{t} differs from {k}"
    assert not set(ids["sig_val"]) & set(ids["sig_test"]), "validation/test overlap"
    exp = read_tsv(a.expected)
    neg = sorted((r[0][:-4], r[1], r[2]) for r in exp if r[2] in ("dev", "test") and r[0].endswith(".mp4"))
    cids = [r[0] for r in neg]
    n_test, n_dev = sum(r[2] == "test" for r in neg), sum(r[2] == "dev" for r in neg)
    assert (n_test, n_dev) == (a.n_test, a.n_dev), f"negatives: test {n_test} dev {n_dev}"
    assert len(set(cids)) == len(cids) and not set(cids) & set(ids["sig_val"] + ids["sig_test"]), "crop id collision"
    assert {r[1] for r in neg if r[2] == "test"} <= set(ids["sig_test"]), "a test negative's parent is not a test clip"
    assert {r[1] for r in neg if r[2] == "dev"} <= set(ids["sig_val"]), "a dev negative's parent is not a validation clip"
    paths = {k: [r[0] for r in v] for k, v in man.items()}
    paths["neg"] = [str(Path(a.neg_dir) / f"{c}.npz") for c in cids]
    ids["neg"] = cids
    for k, ps in paths.items():
        bad = [p for p in ps if not p.startswith(a.require_prefix) or not os.path.exists(p)]
        assert not bad, f"{k}: {len(bad)} pose files missing or outside {a.require_prefix!r} (e.g. {bad[0]})"
    with open(out / "negatives.tsv", "w", encoding="utf-8") as f:      # text = crop id: the dump rows identify themselves
        f.writelines(f"{p}\t{c}\n" for p, c in zip(paths["neg"], cids))
    json.dump({r[0]: {"parent": r[1], "split": r[2]} for r in neg}, open(out / "negatives_meta.json", "w"))

    if a.neg_mp4_dir:                                                   # symlink lists for the unmodified extractor
        def walk(d, suffix):
            return {f[:-len(suffix)]: os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(suffix)}
        nmp4, smp4 = walk(a.neg_mp4_dir, ".mp4"), walk(a.sig_mp4_dir, "_signer.mp4")
        assert all(c in nmp4 for c in cids), "a dev/test non-signer mp4 is missing"
        assert all(c in smp4 for c in ids["sig_val"] + ids["sig_test"]), "a val/test signer mp4 is missing"
        gate = random.Random(0).sample(sorted(ids["sig_test"]), a.n_gate) + random.Random(0).sample(sorted(ids["sig_val"]), a.n_gate)
        # the extractor names a clip basename.replace("_signer.mp4", ""), so every link is <pose stem>_signer.mp4
        lists = {"neg": [(nmp4[c], c) for c in cids], "trk": [(smp4[c], c) for c in ids["sig_val"] + ids["sig_test"]],
                 "gate": [(smp4[c], c) for c in gate]}
        for k, rows in lists.items():
            assert all(f"{c}_signer.mp4".replace("_signer.mp4", "") == c for _, c in rows)
            with open(out / f"ext_{k}.tsv", "w") as f:
                f.writelines(f"{src}\t{c}_signer.mp4\n" for src, c in rows)
        print(f"extraction lists: neg {len(lists['neg'])} trk {len(lists['trk'])} gate {len(lists['gate'])}")
    if a.lists_only:
        return
    facts = {}
    for k in SETS:
        facts[k] = {}
        for c, p in zip(ids[k], paths[k]):
            z = np.load(p)
            kps = z["KPS"]
            assert kps.ndim == 3 and kps.shape[1:] == (133, 3), f"{p}: KPS {kps.shape}"
            e = {"md5": md5(p), "T": int(len(kps)), "vT": valid_T(kps)}
            assert e["vT"] >= 1, f"{p}: no valid frame (the loader raises on such a clip)"
            if "NDET" in z.files:
                e["ndet_max"] = int(z["NDET"].max()) if len(z["NDET"]) else 0
            facts[k][c] = e
    json.dump({"order": ids, "facts": facts}, open(out / "pose_facts.json", "w"))
    print(f"MANIFEST_OK negatives test {n_test} dev {n_dev} | pose files hashed "
          + " ".join(f"{k} {len(facts[k])}" for k in SETS))


def featcheck(manifest, rgb_dir, n):
    """The validity rule of slt.rgb_fusion.RGBDataset._load_feature, plus rows == min(24, pose T). Returns a report."""
    rows = read_tsv(manifest)
    bad, zero_blocks, blocks, allzero = [], 0, 0, 0
    for clip, _ in rows:
        p = _rgb_path(clip, rgb_dir)
        try:
            f = np.asarray(np.load(p)["FEATURES"], np.float32)
            T = len(np.load(clip)["KPS"])
            ok = f.ndim == 2 and f.shape[1] == RGB_DIM and f.shape[0] == min(FRAMES, T) >= 1 and np.isfinite(f).all()
        except Exception:
            ok = False
        if not ok:
            bad.append(p); continue
        z = np.abs(f).reshape(len(f), 3, RGB_DIM // 3).sum(-1) == 0
        zero_blocks += int(z.sum()); blocks += z.size; allzero += int(z.all())
    return {"manifest": str(manifest), "rgb_dir": str(rgb_dir), "rows": len(rows), "invalid": len(bad), "first_invalid": bad[:5],
            "zero_crop_blocks": zero_blocks, "crop_blocks": blocks, "files_all_zero": allzero,
            "ok": bool(len(rows) == n and not bad)}


def rgbgate(ids, new_dir, stored_dir, min_cos=0.9999):
    """Per crop (384-D block): cosine between today's and the stored feature; zero (no-crop) pattern must be identical."""
    cos, dmax, bad = [], 0.0, []
    for c in ids:
        try:
            a = np.load(Path(new_dir) / f"rgb_{c}.npz")["FEATURES"].astype(np.float64)
            b = np.load(Path(stored_dir) / f"rgb_{c}.npz")["FEATURES"].astype(np.float64)
        except Exception:
            bad.append(c); continue
        if a.shape != b.shape:
            bad.append(c); continue
        a, b = a.reshape(-1, RGB_DIM // 3), b.reshape(-1, RGB_DIM // 3)
        za, zb = np.abs(a).sum(1) == 0, np.abs(b).sum(1) == 0
        if (za != zb).any():
            bad.append(c); continue
        m = ~za
        cos.extend(((a[m] * b[m]).sum(1) / (np.linalg.norm(a[m], axis=1) * np.linalg.norm(b[m], axis=1))).tolist())
        dmax = max(dmax, float(np.abs(a - b).max()))
    rep = {"clips": len(ids), "unusable": bad[:10], "n_unusable": len(bad), "crops": len(cos),
           "cos_min": float(min(cos)) if cos else None, "cos_median": float(np.median(cos)) if cos else None,
           "max_abs_diff": dmax, "min_cos_required": min_cos}
    rep["ok"] = bool(ids and not bad and cos and rep["cos_min"] >= min_cos)
    return rep


def write_gate(out, name, rep):
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(Path(out) / f"{name}.json", "w"), indent=1)
    print(f"GATE {name}: {'PASS' if rep['ok'] else 'FAIL'} " + json.dumps({k: v for k, v in rep.items() if k != 'ok'})[:900],
          flush=True)
    if not rep["ok"]:
        sys.exit(1)


# ---------------------------------------------------------------- local: build + analyse
def manifest_ids(p):
    return [(Path(r[0]).stem, r[1]) for r in read_tsv(p)]


def build(kaya, out_dir=S6, single=RES / "nonsigner", val_dump=RES / "final_ensemble_validation_decodes", test_dump=RES / "final_ensemble_test_decodes",
          sig_dirs=None):
    """Assemble the ensemble decodes. Thresholds and test coverage of arm (i) come from the SAVED dumps; the Kaya
    re-decodes of the same clips are only checked against them. Nothing here looks at a statistic of the negatives."""
    kaya, out_dir = Path(kaya), Path(out_dir)
    sig_dirs = sig_dirs or {"sig_val": RES / "val_clips_dwpose", "sig_test": RES / "test_clips_dwpose"}
    need = ["rgbgate", "feat_sig_val", "feat_sig_test", "feat_neg", "feat_trk_val", "feat_trk_test", "dump_sig_val", "dump_sig_test"]
    for g in need:                                               # 1. every Kaya gate passed
        assert json.load(open(kaya / "gates" / f"{g}.json"))["ok"] is True, f"Kaya gate {g} did not pass"
    for k in ("sig_val", "sig_test", "trk_val", "trk_test", "neg"):
        assert (kaya / "dumps" / k / "DONE").exists(), f"decode {k} has no DONE marker"
    T = json.load(open(single / "decodes.json"))                 # pose facts and ids of the single-model analysis
    P = json.load(open(kaya / "pose_facts.json"))
    val, test = manifest_ids(RES / "_local_val_manifest.tsv"), manifest_ids(RES / "TEST_manifest_official.tsv")
    order = {"sig_val": [c for c, _ in val], "sig_test": [c for c, _ in test]}
    order["trk_val"], order["trk_test"] = order["sig_val"], order["sig_test"]
    order["neg"] = [r[1] for r in read_tsv(kaya / "negatives.tsv")]
    text = {"sig_val": [t for _, t in val], "sig_test": [t for _, t in test]}
    text["trk_val"], text["trk_test"], text["neg"] = text["sig_val"], text["sig_test"], order["neg"]
    rep = {}
    for k in SETS:                                               # 2. Kaya decoded the pose files that are on this disk
        assert P["order"][k] == order[k] and set(order[k]) == set(T[k]), f"{k}: clip ids or order differ"
        for c in order[k]:
            f = P["facts"][k][c]
            assert (f["T"], f["vT"]) == (T[k][c]["T"], T[k][c]["vT"]), (k, c)
            local = T[k][c].get("md5") or md5(Path(sig_dirs[k]) / f"{c}.npz")
            assert f["md5"] == local, f"{k}/{c}: the pose file decoded on Kaya is not the local one"
    for name, dump in (("val", val_dump), ("test", test_dump)):   # 3. the saved dumps are the recorded system
        g = dumpgate(dump, *RECORDED[name])
        assert g["ok"] and g.get("reselect_equals_tool") is True, f"saved {name} dump does not reproduce the record: {g}"
        rep[f"saved_{name}"] = g
        c = compare_dumps(kaya / "dumps" / f"sig_{name}", dump, 798)   # 4. the Kaya re-decode reproduces them row by row
        assert c["ok"], f"Kaya re-decode of {name} differs from the saved dump: {c}"
        rep[f"redecode_{name}_vs_saved"] = c
        rep[f"tracked_{name}"] = dict(zip(("bleu4", "wer"), scores(select(kaya / "dumps" / f"trk_{name}"))))  # descriptive
    rep["features"] = {g: {f: v for f, v in json.load(open(kaya / "gates" / f"{g}.json")).items() if f != "first_invalid"}
                       for g in need if not g.startswith("dump_")}                                             # descriptive
    from ensemble_selective import load                           # 5. my selection = the recorded tool's selection
    _, hy, cf = load(val_dump, LP)
    mine = select(val_dump)
    assert hy == [x["hyp"] for x in mine] and np.allclose(cf, [x["conf"] for x in mine], rtol=0, atol=0)
    src = {"sig_val": val_dump, "sig_test": test_dump, "trk_val": kaya / "dumps/trk_val", "trk_test": kaya / "dumps/trk_test",
           "neg": kaya / "dumps/neg"}
    D = {}
    for k in SETS:
        sel = select(src[k])
        assert [x["ref"] for x in sel] == text[k], f"{k}: dump rows are not in manifest order"
        D[k] = {c: {**{f: v for f, v in T[k][c].items() if f not in ("hyp", "ce", "frozen")},
                    "md5": P["facts"][k][c]["md5"], "hyp": x["hyp"], "ce": -x["conf"]} for c, x in zip(order[k], sel)}
    D["zero"] = {"hyp": None, "ce": None, "note": "not decoded: the current loader raises on a clip with no valid frame; "
                                                  "no crop has zero valid frames"}
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(D, open(out_dir / "decodes.json", "w"))
    json.dump(rep, open(out_dir / "build_checks.json", "w"), indent=1)
    print("BUILD_OK", json.dumps({k: {f: v for f, v in r.items() if f in ("bleu4", "wer", "same_hyp", "max_abs_dconf")}
                                  for k, r in rep.items()}))
    return D


def reproduce_single(g, single=RES / "nonsigner"):
    """Harness: nonsigner_gate.analyse() on the single-model decodes must reproduce the recorded results file."""
    rec = json.load(open(single / "nonsigner_gate_results.json"))
    with tempfile.TemporaryDirectory() as t:
        g.analyse(decodes=single / "decodes.json", out=Path(t) / "r.json")
        new = json.load(open(Path(t) / "r.json"))
    new["B3"].pop("flagged_ids", None)

    def diff(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            return [d for k in sorted(set(a) | set(b)) for d in diff(a.get(k), b.get(k), f"{path}/{k}")]
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
        if isinstance(a, float) and isinstance(b, float):
            return [] if abs(a - b) <= 1e-9 else [f"{path}: {b} -> {a}"]
        return [] if a == b else [f"{path}: {b!r} -> {a!r}"]
    d = diff(new, rec)
    assert not d, f"single-model record not reproduced ({len(d)} differences), e.g. {d[:5]}"
    print("single-model record reproduced: results/nonsigner/nonsigner_gate_results.json, every value")
    return rec


def pose_side(r):
    """Everything in a results dict that depends on the pose files only (not on any model)."""
    out = {"B3": {k: v for k, v in r["B3"].items() if k != "flagged_ids"}, "G4": r["G4_length_mismatch"],
           "no_person": r["no_person_detected"]}
    for arm in ("i_production", "ii_tracked"):
        for q in QS:
            x = r[arm][f"q{q}"]
            out[f"{arm}/{q}"] = {"n": [x["test"]["n_sig"], x["test"]["n_neg"], x["dev"]["n_neg"]],
                                 "strata_n": {k: v[0] for k, v in x["strata_FV"].items()},
                                 "quartile_n": {k: [v["n_sig"], v["n_neg"]] for k, v in x["R_by_vT_quartile"].items()}}
    return json.loads(json.dumps(out, default=str))


def analyse(out_dir=S6, single=RES / "nonsigner", skip_single=False):
    import nonsigner_gate as g
    out_dir = Path(out_dir)
    rec = json.load(open(single / "nonsigner_gate_results.json")) if skip_single else reproduce_single(g, single)
    D = json.load(open(out_dir / "decodes.json"))
    rep = g.analyse(decodes=out_dir / "decodes.json", out=out_dir / "nonsigner_gate_s6_results.json")
    rep = json.loads(json.dumps(rep, default=str))
    ex = {}
    # (a) the pre-registered wording of the rule gives the same decisions as ce <= quantile(ce_val, q)
    for arm, sv, st in (("i_production", "sig_val", "sig_test"), ("ii_tracked", "trk_val", "trk_test")):
        for q in QS:
            tau = rep[arm][f"q{q}"]["tau"]
            thr = float(np.quantile([-e["ce"] for e in D[sv].values()], 1 - q))
            for k in (st, "neg"):
                assert all((-e["ce"] >= thr) == (e["ce"] <= tau) for e in D[k].values()), (arm, q, k)
    # (b) arm (i) test coverage is the recorded selective-prediction coverage of S6
    for q in QS:
        c = rep["i_production"][f"q{q}"]["test"]["C"]
        assert abs(c - REC_COVERAGE[q]) < 1e-9, f"C({q}) = {c}, recorded {REC_COVERAGE[q]}"
    ex["coverage_reproduces_enssel_S6"] = True
    # (c) the populations and strata are those of the single-model analysis
    assert pose_side(rep) == pose_side(rec), "pose-side quantities differ from the single-model analysis"
    ex["pose_side_equals_single_model"] = True
    # (d) ensemble against single model on the same crops (primary test population)
    S = json.load(open(single / "decodes.json"))
    neg = g.negatives()
    vid = dict(zip(neg.Person_Video_Clip.str[:-4], neg.Video_Name))
    flagged = set(rep["B3"]["flagged_ids"])
    pop = [c for c, s in zip(neg.Person_Video_Clip.str[:-4], neg.Split) if s == "test" and c not in flagged]
    ex["vs_single"] = {}
    for q in QS:
        p6 = {c: D["neg"][c]["ce"] <= rep["i_production"][f"q{q}"]["tau"] for c in pop}
        p1 = {c: S["neg"][c]["ce"] <= rec["i_production"][f"q{q}"]["tau"] for c in pop}
        by = {}
        for c in pop:
            by.setdefault(vid[c], []).append(float(p6[c]) - float(p1[c]))
        d, lo, hi = g.cluster_boot(list(by), lambda vs: float(np.mean([x for v in vs for x in by[v]])))
        ex["vs_single"][f"q{q}"] = {
            "n": len(pop), "FV_ensemble": float(np.mean(list(p6.values()))), "FV_single": float(np.mean(list(p1.values()))),
            "diff": d, "diff_lo": lo, "diff_hi": hi,
            "both": sum(p6[c] and p1[c] for c in pop), "ensemble_only": sum(p6[c] and not p1[c] for c in pop),
            "single_only": sum(p1[c] and not p6[c] for c in pop), "neither": sum(not p6[c] and not p1[c] for c in pop),
            "R_ensemble": [rep["i_production"][f"q{q}"]["test"][k] for k in ("R", "R_lo", "R_hi")],
            "R_single": [rec["i_production"][f"q{q}"]["test"][k] for k in ("R", "R_lo", "R_hi")]}
    ex["same_reading_as_single_model"] = rep["final_reading"] == rec["final_reading"]
    json.dump(ex, open(out_dir / "w12e_extras.json", "w"), indent=1)
    print("\n=== wave 12e: six-member ensemble (beam 10, LP 0.3), thresholds from the saved validation dump ===")
    for arm in ("i_production", "ii_tracked"):
        for q in QS:
            r = rep[arm][f"q{q}"]
            print(f"{arm:13s} q={q:<4} test C {r['test']['C']:.3f} FV {r['test']['FV']:.3f} R {r['test']['R']:.2f} "
                  f"[{r['test']['R_lo']:.2f}, {r['test']['R_hi']:.2f}] {r['test']['reading']:14s} | dev FV {r['dev']['FV']:.3f} "
                  f"R {r['dev']['R']:.2f} [{r['dev']['R_lo']:.2f}, {r['dev']['R_hi']:.2f}] {r['dev']['reading']:14s} | "
                  f"quiescent n {r['strata_FV']['quiescent'][0]} FV {r['strata_FV']['quiescent'][1]:.3f}")
        print(f"{arm:13s} AUROC {rep[arm]['auroc'][0]:.3f} [{rep[arm]['auroc'][1]:.3f}, {rep[arm]['auroc'][2]:.3f}]")
    print("FINAL READING (weaker of arms and of test/dev, q = 0.50):", rep["final_reading"],
          "| single model:", rec["final_reading"])
    for q in QS:
        v = ex["vs_single"][f"q{q}"]
        print(f"q={q}: non-signers voiced, ensemble {v['FV_ensemble']:.3f} vs single {v['FV_single']:.3f}, difference "
              f"{v['diff']:+.3f} [{v['diff_lo']:+.3f}, {v['diff_hi']:+.3f}] (n {v['n']}; both {v['both']}, ensemble only "
              f"{v['ensemble_only']}, single only {v['single_only']}, neither {v['neither']})")
    return rep, ex


# ---------------------------------------------------------------- self-test (no dataset needed)
def selftest():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)

        def dump(name, rows):
            (t / name).mkdir()
            with open(t / name / "nbest.jsonl", "w") as f:
                f.writelines(json.dumps({"idx": i, "ref": r, "nbest": nb}) + "\n" for i, (r, nb) in enumerate(rows))
            return t / name
        rows = [("a b .", [[-2.0, "a b ."], [-1.0, "a ."]]), ("c d e .", [[-9.0, "c d e ."], [-8.0, "x y z w ."]])]
        d = dump("d", rows)
        s = select(d)
        # row 0: -2/3**.3 = -1.44 vs -1/2**.3 = -0.81 -> "a ."; confidence -1 / (2 + 1)
        assert s[0]["hyp"] == "a ." and abs(s[0]["conf"] + 1 / 3) < 1e-12
        # row 1: -9/4**.3 = -5.94 vs -8/5**.3 = -4.94 -> the longer one
        assert s[1]["hyp"] == "x y z w ." and abs(s[1]["conf"] + 8 / 6) < 1e-12
        assert compare_dumps(d, d, 2)["ok"] and compare_dumps(d, d, 2)["max_abs_dconf"] == 0.0
        e = dump("e", [rows[0], ("c d e .", [[-9.0, "c d e ."], [-18.0, "x y z w ."]])])
        assert compare_dumps(e, d, 2)["ok"] is False and compare_dumps(e, d, 1)["ok"] is True      # one hypothesis differs
        assert compare_dumps(dump("f", [rows[0], ("OTHER", rows[1][1])]), d, 0)["ok"] is False      # references differ
        shifted = dump("h", [(r, [[sc - 1e-3, h] for sc, h in nb]) for r, nb in rows])
        assert compare_dumps(shifted, d, 2)["ok"] is False                                           # confidence drifted
        tail = compare_dumps(dump("g", [("a b .", [[-2.5, "a b ."], [-1.0, "a ."]]), rows[1]]), d, 2)
        assert tail["same_hyp"] == 2 and tail["max_abs_dconf"] == 0.0                                # same outputs, but a
        assert tail["first50_rows_not_reproduced"] == 1 and tail["ok"] is False                      # raw n-best score moved
        b, w = scores(select(dump("p", [("a b c d e .", [[-1.0, "a b c d e ."]])])))
        assert abs(b - 100) < 1e-6 and w == 0.0
        assert dumpgate(t / "p", 100.0, 0.0, n=1)["ok"] and not dumpgate(t / "p", 99.0, 0.0, n=1)["ok"]
        assert not dumpgate(t / "p", 100.0, 0.0, n=2)["ok"]
        # ce = -confidence: the two wordings of the threshold rule decide alike
        rng = np.random.default_rng(0)
        conf = -rng.gamma(2.0, 0.4, 801)
        for q in QS:
            assert ((conf >= np.quantile(conf, 1 - q)) == (-conf <= np.quantile(-conf, q))).all()
        # features: validity, row count, cosine gate
        clip = t / "video_1_2_3_nonsigner.npz"
        np.savez(clip, KPS=np.ones((30, 133, 3), np.float32), BBOX=np.zeros((30, 4), np.float32))
        man = t / "m.tsv"; man.write_text(f"{clip}\tvideo_1_2_3_nonsigner\n")
        fd = t / "feats"; fd.mkdir()
        assert featcheck(man, fd, 1)["ok"] is False                                                  # missing file
        good = rng.normal(size=(24, RGB_DIM)).astype(np.float32); good[3, :384] = 0
        for bad in (good[:23], good[:, :1151], np.where(np.arange(RGB_DIM) == 5, np.nan, good)):
            np.savez(fd / "rgb_video_1_2_3_nonsigner.npz", FEATURES=bad)
            assert featcheck(man, fd, 1)["ok"] is False
        np.savez(fd / "rgb_video_1_2_3_nonsigner.npz", FEATURES=good)
        r = featcheck(man, fd, 1)
        assert r["ok"] and r["zero_crop_blocks"] == 1 and featcheck(man, fd, 2)["ok"] is False
        assert Path(_rgb_path(str(clip), str(fd))).name == "rgb_video_1_2_3_nonsigner.npz"
        assert "video_1_2_3_nonsigner_signer.mp4".replace("_signer.mp4", "") == "video_1_2_3_nonsigner"   # extractor's rule
        fd2 = t / "feats2"; fd2.mkdir()
        ids = ["video_1_2_3_nonsigner"]
        np.savez(fd2 / "rgb_video_1_2_3_nonsigner.npz", FEATURES=good * (1 + 1e-7))
        assert rgbgate(ids, fd2, fd)["ok"]
        noisy = good + 0.05 * rng.normal(size=good.shape).astype(np.float32); noisy[3, :384] = 0
        np.savez(fd2 / "rgb_video_1_2_3_nonsigner.npz", FEATURES=noisy)
        assert rgbgate(ids, fd2, fd)["ok"] is False                                                  # cosine ~0.9988
        moved = good.copy(); moved[3, :384] = 1; moved[4, :384] = 0
        np.savez(fd2 / "rgb_video_1_2_3_nonsigner.npz", FEATURES=moved)
        assert rgbgate(ids, fd2, fd)["ok"] is False and rgbgate([], fd2, fd)["ok"] is False          # zero pattern moved
        assert valid_T(np.zeros((4, 133, 3))) == 0 and valid_T(np.ones((4, 133, 3))) == 4
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    m = sub.add_parser("manifest")
    for k in ("expected", "neg_dir", "val_manifest", "test_manifest", "trk_val", "trk_test", "out_dir"):
        m.add_argument(f"--{k}", required=True)
    m.add_argument("--neg_mp4_dir", default=""); m.add_argument("--sig_mp4_dir", default="")
    m.add_argument("--lists_only", action="store_true")
    m.add_argument("--require_prefix", default="/group/")
    m.add_argument("--n_test", type=int, default=847); m.add_argument("--n_dev", type=int, default=831)
    m.add_argument("--n_gate", type=int, default=50)
    f = sub.add_parser("featcheck")
    f.add_argument("--manifest", required=True); f.add_argument("--rgb_dir", required=True)
    f.add_argument("--n", type=int, required=True); f.add_argument("--name", required=True); f.add_argument("--gates", default="")
    r = sub.add_parser("rgbgate")
    r.add_argument("--list", required=True, help="ext_gate.tsv written by the manifest stage")
    r.add_argument("--new", required=True); r.add_argument("--stored", required=True); r.add_argument("--gates", default="")
    d = sub.add_parser("dumpgate")
    d.add_argument("--new", required=True); d.add_argument("--split", choices=("val", "test"), required=True)
    d.add_argument("--saved", default=""); d.add_argument("--name", required=True); d.add_argument("--gates", default="")
    b = sub.add_parser("build"); b.add_argument("--kaya", default=str(S6 / "kaya")); b.add_argument("--out", default=str(S6))
    an = sub.add_parser("analyse"); an.add_argument("--out", default=str(S6)); an.add_argument("--skip_single", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.stage == "manifest":
        stage_manifest(a)
    elif a.stage == "featcheck":
        write_gate(a.gates, a.name, featcheck(a.manifest, a.rgb_dir, a.n))
    elif a.stage == "rgbgate":
        ids = [r[1].replace("_signer.mp4", "") for r in read_tsv(a.list)]
        write_gate(a.gates, "rgbgate", rgbgate(ids, a.new, a.stored))
    elif a.stage == "dumpgate":
        write_gate(a.gates, a.name, dumpgate(a.new, *RECORDED[a.split], saved=a.saved or None))
    elif a.stage == "build":
        build(a.kaya, a.out)
    elif a.stage == "analyse":
        analyse(a.out, skip_single=a.skip_single)
    else:
        selftest()


if __name__ == "__main__":
    main()
