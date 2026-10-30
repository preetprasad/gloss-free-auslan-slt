"""Non-signer hard negatives for the confidence gate. Pre-registration (binding; amendments B1-B15 win):
results/PREREG_NONSIGNER_GATE_2026-09-22.md.   Stages:  selftest | decode | analyse

The model and decoder are the speech chain's own: tools/speech_pipeline_bench.translate() on
comm_dwpose_vlp_ctc.pt, beam 5, LP 0.6. Nothing about the negatives fits a threshold, stratum or setting.
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE)]
from config import Config  # noqa: E402
from slt.data import normalize_clip  # noqa: E402
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402
from slt.reference_model import build_model  # noqa: E402
from speech_pipeline_bench import translate  # noqa: E402
from content_failure_analysis import content, stoplist  # noqa: E402

RES = ROOT / "results"
NS = RES / "nonsigner"
DS = Path.home() / "Library/CloudStorage/GoogleDrive-<account>/My Drive/Honours Datasets/Auslan-Daily/Dataset Split Table"
CKPT = RES / "ckpts/comm_dwpose_vlp_ctc.pt"
BEAM, LP, QS, B, CONF = 5, 0.6, (0.50, 0.75), 2000, 0.3
BODY_HANDS = np.r_[0:17, 91:133]
RANK = ["INVERTED", "FAILS", "PARTIAL", "WORKS", "WORKS (strong)"]      # weakest -> strongest (B11)


def cid(p):
    return Path(p).stem


def manifest(p):
    return [(l.split("\t")[0], l.rstrip("\n").split("\t")[1]) for l in open(p) if "\t" in l]


def valid_T(kps):
    return int((np.abs(kps).reshape(len(kps), -1).sum(1) > 1e-6).sum())       # B2: drop_zero_frames' own rule


def negatives():
    """Communication Non_Signer crops, dev + test. Signer_Detection.xlsx also lists News clips (21,242 Non_Signer rows;
    14,803 Communication), so the join on the Communication translation table is required, not optional."""
    sd = pd.read_excel(DS / "Signer Detection/Signer_Detection.xlsx")
    comm = set(pd.read_excel(ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx").Video_Clip_Name)
    neg = sd[(sd.Label == "Non_Signer") & sd.Split.isin(["dev", "test"]) & sd.Video_Clip_Name.isin(comm)]
    assert (neg.Split == "test").sum() == 847 and (neg.Split == "dev").sum() == 831, neg.Split.value_counts()
    return neg


# ---------------------------------------------------------------- decode
def frozen(kps):
    """B7: the middle VALID frame repeated for the clip's full length (keeps the person, removes motion)."""
    ok = np.flatnonzero(np.abs(kps).reshape(len(kps), -1).sum(1) > 1e-6)
    return np.repeat(kps[ok[len(ok) // 2]][None], len(kps), 0) if len(ok) else kps


def decode():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config().to_dict()})
    vocab, test, val = ck["vocab"], manifest(RES / "TEST_manifest_official.tsv"), manifest(RES / "_local_val_manifest.tsv")
    model = build_model(399, len(vocab), cfg)
    model.load_state_dict(ck["model"]); model.eval()

    def run(kps, bbox=None):
        f, m = normalize_clip(kps, np.zeros((len(kps), 4), np.float32) if bbox is None else bbox, cfg)
        h, ce = translate(model, f, m, vocab, BEAM, LP)
        return {"hyp": h, "ce": ce, "T": int(len(kps)), "vT": valid_T(kps)}

    neg = negatives()
    sets = {"sig_test": [(cid(p), p) for p, _ in test], "sig_val": [(cid(p), p) for p, _ in val],
            "trk_test": [(cid(p), NS / "tracked" / f"{cid(p)}.npz") for p, _ in test],
            "trk_val": [(cid(p), NS / "tracked" / f"{cid(p)}.npz") for p, _ in val],
            "neg": [(r.Person_Video_Clip[:-4], NS / "npz" / (r.Person_Video_Clip[:-4] + ".npz")) for r in neg.itertuples()]}
    cache = json.load(open(NS / "decodes.json")) if (NS / "decodes.json").exists() else {}
    for name, items in sets.items():
        if name in cache:
            continue
        missing = [str(p) for _, p in items if not Path(p).exists()]
        if missing:
            print(f"{name}: {len(missing)} files missing (e.g. {missing[0]}) -- skipped"); continue
        out = {}
        for k, p in items:
            d = np.load(p)
            out[k] = run(d["KPS"], d["BBOX"])
            out[k]["md5"] = fingerprint(p)
            if name in ("sig_test", "sig_val"):
                out[k]["frozen"] = run(frozen(d["KPS"]))["ce"]
            if "NDET" in d:
                out[k]["ndet_max"] = int(d["NDET"].max()) if len(d["NDET"]) else 0
        cache = json.load(open(NS / "decodes.json")) if (NS / "decodes.json").exists() else {}
        cache[name] = out
        json.dump(cache, open(NS / "decodes.json", "w"))
        print(f"decoded {name}: {len(out)}")
    if "zero" not in cache:                                   # B1: a crop with no detected person
        cache["zero"] = run(np.zeros((50, 133, 3), np.float32))
        json.dump(cache, open(NS / "decodes.json", "w"))
    print("zero-person input ->", cache["zero"])
    # G2: the chain reproduces itself
    bench = {r["clip"]: r for r in json.load(open(RES / "speech_pipeline_bench.json"))["records"]}
    st = cache["sig_test"]
    same = sum(st[k]["hyp"] == bench[k]["hyp"] and abs(st[k]["ce"] - bench[k]["ce"]) < 1e-5 for k in bench)
    bl = corpus_bleu4([detokenize(st[k]["hyp"]) for k in bench], [detokenize(bench[k]["ref"]) for k in bench])
    print(f"G2: {same}/800 hyps+ce reproduce the bench | BLEU-4 {bl:.4f}")
    assert same == 800 and abs(bl - 10.2691) < 0.0015, "G2 FAILED"


# ---------------------------------------------------------------- skeleton geometry
def norm_skel(kps):
    """Per frame: shoulder midpoint to origin, shoulder width to 1. Frames without confident shoulders -> None."""
    out = []
    for f in kps:
        if f[5, 2] < CONF or f[6, 2] < CONF:
            out.append(None); continue
        mid, w = (f[5, :2] + f[6, :2]) / 2, np.linalg.norm(f[5, :2] - f[6, :2])
        out.append(None if w < 1e-3 else ((f[:, :2] - mid) / w, f[:, 2]))
    return out


def skel_dist(a, b):
    """B3: median over frames of the mean L2 over body+hand keypoints confident in both. None if untestable."""
    A, Bs, per = norm_skel(a), norm_skel(b), []
    for x, y in zip(A, Bs):
        if x is None or y is None:
            continue
        m = (x[1][BODY_HANDS] >= CONF) & (y[1][BODY_HANDS] >= CONF)
        if m.any():
            per.append(float(np.linalg.norm(x[0][BODY_HANDS][m] - y[0][BODY_HANDS][m], axis=1).mean()))
    return float(np.median(per)) if per else None


def wrist_speed(kps):
    """Per-crop median wrist speed (kpts 9, 10; conf >= 0.3), in shoulder widths per frame."""
    sp = []
    for t in range(1, len(kps)):
        a, b = kps[t - 1], kps[t]
        if min(a[5, 2], a[6, 2], b[5, 2], b[6, 2]) < CONF:
            continue
        w = np.linalg.norm(b[5, :2] - b[6, :2])
        for j in (9, 10):
            if a[j, 2] >= CONF and b[j, 2] >= CONF and w > 1e-3:
                sp.append(np.linalg.norm(b[j, :2] - a[j, :2]) / w)
    return float(np.median(sp)) if sp else float("nan")


def tracked_share(kps):
    return float(np.mean((kps[:, 5, 2] >= CONF) & (kps[:, 6, 2] >= CONF))) if len(kps) else 0.0


# ---------------------------------------------------------------- statistics
def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 ties, score = -ce (higher = more confident)."""
    s = np.r_[-np.asarray(pos), -np.asarray(neg)]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot(videos, stat, seed=0):
    """Resample Video_Name with replacement (all clips and negatives of a drawn video come along)."""
    rng, uniq = np.random.default_rng(seed), sorted(set(videos))
    full, draws = stat(uniq), []
    for _ in range(B):
        v = stat(list(rng.choice(uniq, len(uniq))))
        if v is not None and np.isfinite(v):
            draws.append(v)
    return full, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def category(r, lo, hi):
    if lo > 1:
        return "INVERTED"
    if hi >= 1:
        return "FAILS"
    return "WORKS (strong)" if r <= 0.20 else "WORKS" if r <= 0.50 else "PARTIAL"


def gate_split(sig_ce, sig_vid, neg_ce, neg_vid, tau):
    """R = FV/C with a video-cluster CI; the signers and negatives of one resampled video travel together."""
    by_s, by_n = {}, {}
    for c, v in zip(sig_ce, sig_vid):
        by_s.setdefault(v, []).append(c <= tau)
    for c, v in zip(neg_ce, neg_vid):
        by_n.setdefault(v, []).append(c <= tau)

    def R(vs):
        s = [x for v in vs for x in by_s.get(v, [])]
        n = [x for v in vs for x in by_n.get(v, [])]
        return None if not s or not n or not np.mean(s) else float(np.mean(n) / np.mean(s))
    r, lo, hi = cluster_boot(list(by_s) + list(by_n), R)
    return {"C": float(np.mean([c <= tau for c in sig_ce])), "FV": float(np.mean([c <= tau for c in neg_ce])),
            "n_sig": len(sig_ce), "n_neg": len(neg_ce), "R": r, "R_lo": lo, "R_hi": hi, "reading": category(r, lo, hi)}


def gate_fixed_q(neg_ce, neg_vid, tau, q):
    """Dev replication (§5): R_dev = FV_dev / q with tau fixed from validation, so the denominator is a constant and the
    video-cluster bootstrap runs over the dev negatives only."""
    by_n = {}
    for c, v in zip(neg_ce, neg_vid):
        by_n.setdefault(v, []).append(c <= tau)
    R = lambda vs: float(np.mean([x for v in vs for x in by_n[v]]) / q)
    r, lo, hi = cluster_boot(list(by_n), R)
    return {"C": q, "FV": float(np.mean([c <= tau for c in neg_ce])), "n_neg": len(neg_ce),
            "R": r, "R_lo": lo, "R_hi": hi, "reading": category(r, lo, hi)}


def concurrent_other_signer():
    """B8: parent clip -> True if its span overlaps an alignment row whose signer is known and DIFFERENT; None if the clip
    is outside the unambiguous (video, subtitle) join, or an overlapping row's signer cannot be determined; else False.
    Overlaps are checked against ALL alignment rows, not only the joined ones."""
    tr = pd.read_excel(ROOT / "data/auslan-daily/translation/Auslan-Daily_Communication.xlsx")
    al = pd.read_excel(DS / "Sign Language Alignment/Auslan_Daily_Communication.xlsx")
    key = ["Video_Name", "Subtitle"]
    sig_of = tr.groupby(key).Signer_ID.agg(lambda x: set(x)).to_dict()      # (video, subtitle) -> set of signer ids
    tu = tr[~tr.duplicated(key, keep=False)]
    au = al[~al.duplicated(key, keep=False)]
    j = tu.merge(au, on=key)[["Video_Name", "Subtitle", "Video_Clip_Name", "Signer_ID", "Begin_Frame", "End_Frame"]]
    rows = {v: g[["Subtitle", "Begin_Frame", "End_Frame"]].to_numpy() for v, g in al.groupby("Video_Name")}
    out = {}
    for v, sub, c, s, b, e in j.to_numpy():
        state = False
        for sub2, b2, e2 in rows[v]:
            if sub2 == sub and b2 == b and e2 == e:
                continue                                                     # its own row
            if b2 < e and b < e2:
                ids = sig_of.get((v, sub2), set())
                if len(ids) == 1 and s not in ids:
                    state = True; break
                if len(ids) != 1:
                    state = None
        out[c] = state
    return out


def fingerprint(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def analyse(decodes=None, out=None):
    """decodes / out default to the single-model files. Wave 12e (tools/w12e_nonsigner_s6.py) passes the six-member
    ensemble's decodes in the same format (ce = -confidence) so that every statistic is computed by this one function."""
    decodes, out = Path(decodes or NS / "decodes.json"), Path(out or NS / "nonsigner_gate_results.json")
    D = json.load(open(decodes))
    sd = pd.read_excel(DS / "Signer Detection/Signer_Detection.xlsx")
    neg = negatives().copy()
    neg["cid"] = neg.Person_Video_Clip.str[:-4]
    pmap = dict(zip(neg.cid, neg.Video_Clip_Name))
    split_of = dict(zip(neg.cid, neg.Split))
    vid = dict(zip(sd.Video_Clip_Name, sd.Video_Name))
    test_ids = [cid(p) for p, _ in manifest(RES / "TEST_manifest_official.tsv")]
    val_ids = [cid(p) for p, _ in manifest(RES / "_local_val_manifest.tsv")]
    refs = {cid(p): t for p, t in manifest(RES / "TEST_manifest_official.tsv")}
    G, rep = D["neg"], {"zero_person_input": D["zero"]}
    npz = lambda c: np.load(NS / "npz" / f"{c}.npz")["KPS"]
    trk = lambda c: np.load(NS / "tracked" / f"{c}.npz")["KPS"]

    # cache integrity: the decodes must belong to the files on disk now
    for name, d_ in (("neg", NS / "npz"), ("trk_test", NS / "tracked"), ("trk_val", NS / "tracked")):
        stale = [k for k, v in D.get(name, {}).items() if v.get("md5") != fingerprint(d_ / f"{k}.npz")]
        assert not stale, f"{name}: {len(stale)} cached decodes do not match the npz on disk (e.g. {stale[:3]})"

    # G3/G4
    nt, nd = neg[neg.Split == "test"], neg[neg.Split == "dev"]
    assert len(nt) == 847 and len(nd) == 831 and set(nt.Video_Clip_Name) <= set(test_ids) \
        and set(nd.Video_Clip_Name) <= set(val_ids) and not set(neg.cid) & set(test_ids + val_ids), "G3 FAILED"
    assert set(G) == set(neg.cid), "decoded negatives != the 1,678 registered crops"
    shapes = {npz(c).shape[1:] for c in neg.cid}
    assert shapes == {(133, 3)}, f"G4 FAILED {shapes}"
    sigT = {**{c: D["trk_test"][c] for c in test_ids}, **{c: D["trk_val"][c] for c in val_ids}}
    mis_vT = [c for c in neg.cid if abs(G[c]["vT"] - sigT[pmap[c]]["vT"]) > 2]
    mis_T = [c for c in neg.cid if abs(G[c]["T"] - sigT[pmap[c]]["T"]) > 2]
    rep["G4_length_mismatch"] = {"n_pairs": len(neg), "vT_gt2": len(mis_vT), "vT_share": len(mis_vT) / len(neg),
                                 "rawT_gt2": len(mis_T), "rawT_share": len(mis_T) / len(neg),
                                 "paired_claim_valid": len(mis_vT) / len(neg) <= 0.05}

    # B1: every no-person crop reaches the model as the same single zero frame, so it must decode like D["zero"]
    nope = {c for c in neg.cid if G[c]["vT"] == 0}
    for c in nope:
        assert G[c]["hyp"] == D["zero"]["hyp"] and abs(G[c]["ce"] - D["zero"]["ce"]) < 1e-6, c
    rep["no_person_detected"] = {s: sum(c in nope for c in neg[neg.Split == s].cid) for s in ("test", "dev")}
    rep["no_person_outcome"] = D["zero"]

    # B3 contamination test, on tracked signer skeletons
    rng = random.Random(0)
    null = []
    while len(null) < 1000:                 # reading: 1,000 random unordered pairs of distinct test signer clips
        a, b = rng.sample(test_ids, 2)
        x, y = trk(a), trk(b)
        n = min(len(x), len(y))
        d = skel_dist(x[:n], y[:n])
        if d is not None:
            null.append(d)
    thr = float(np.percentile(null, 1))
    pos = [skel_dist(trk(c), np.load(p)["KPS"]) for c, p in ((cid(p), p) for p, _ in manifest(RES / "TEST_manifest_official.tsv"))]
    pos = [d for d in pos if d is not None]
    pos_ok = float(np.mean([d < thr for d in pos]))
    usable = pos_ok >= 0.90
    flag, why = {}, {}
    for c in neg.cid:
        if c in nope:
            continue
        x, y = npz(c), trk(pmap[c])
        if abs(len(x) - len(y)) > 2:
            why[c] = "length_mismatch"; continue
        n = min(len(x), len(y))
        d = skel_dist(x[:n], y[:n])
        if d is None:
            why[c] = "no_confident_frames"; continue
        flag[c] = usable and d < thr
    rep["B3"] = {"null_p1_threshold": thr, "null_median": float(np.median(null)), "positive_control_below": pos_ok,
                 "test_usable": usable, "flagged_ids": sorted(c for c in flag if flag[c]),
                 "per_split": {s: {"flagged_same_person": sum(flag.get(c, False) for c in neg[neg.Split == s].cid),
                                   "untestable": dict(Counter(why[c] for c in neg[neg.Split == s].cid if c in why))}
                               for s in ("test", "dev")},
                 "limitation": "the positive control compares the tracked and production skeletons of the SAME crop, which "
                               "are the same extraction on almost every frame, so it passes trivially; it cannot show "
                               "detection of the same person seen through a different crop window and re-estimated"}

    # populations (B1, B15)
    prim = lambda split: [c for c in neg[neg.Split == split].cid if c not in nope and not flag.get(c, False)]
    concur, stop = concurrent_other_signer(), stoplist("stitch_lexicon")
    train_sents = {t.strip() for _, t in manifest(RES / "_kaya_train_manifest.tsv")}
    parents = sorted(set(nt.Video_Clip_Name))
    wrist_val = {"sig_val": [wrist_speed(np.load(p)["KPS"]) for p, _ in manifest(RES / "_local_val_manifest.tsv")],
                 "trk_val": [wrist_speed(trk(c)) for c in val_ids]}
    for arm, sig_t, sig_v in (("i_production", "sig_test", "sig_val"), ("ii_tracked", "trk_test", "trk_val")):
        if sig_t not in D:
            continue
        A = rep.setdefault(arm, {})
        ce_val = [D[sig_v][c]["ce"] for c in val_ids]
        tq = np.quantile([D[sig_v][c]["vT"] for c in val_ids], [0.25, 0.5, 0.75])   # B2, fixed on val signers
        qt = lambda t: int(np.searchsorted(tq, t, side="right"))
        ws = np.nanpercentile(wrist_val[sig_v], 10)                                 # arm's own val pose
        pt, pd_ = prim("test"), prim("dev")
        for q in QS:
            tau = float(np.quantile(ce_val, q))
            res = {"tau": tau, "tau_fitted_on": sig_v}
            res["test"] = gate_split([D[sig_t][c]["ce"] for c in test_ids], [vid[c] for c in test_ids],
                                     [G[c]["ce"] for c in pt], [vid[pmap[c]] for c in pt], tau)
            res["dev"] = gate_fixed_q([G[c]["ce"] for c in pd_], [vid[pmap[c]] for c in pd_], tau, q)
            res["dev"]["C_val_realised"] = float(np.mean([x <= tau for x in ce_val]))
            res["reading"] = min(res["test"]["reading"], res["dev"]["reading"], key=RANK.index)
            passes = lambda cs: float(np.mean([G[c]["ce"] <= tau for c in cs])) if cs else None
            res["FV_variants_test"] = {"all_847": passes(list(nt.cid)), "no_person_only": passes([c for c in nt.cid if c in nope]),
                                       "b3_flagged_only": passes([c for c in nt.cid if flag.get(c)])}
            res["FV_variants_dev"] = {"all_831": passes(list(nd.cid)), "b3_flagged_only": passes([c for c in nd.cid if flag.get(c)])}
            res["C_706_parents"] = float(np.mean([D[sig_t][c]["ce"] <= tau for c in parents]))
            wrong = [c for c in test_ids if content(refs[c], stop)
                     and not set(content(refs[c], stop)) & set(content(D[sig_t][c]["hyp"], stop))]
            res["C_err"] = float(np.mean([D[sig_t][c]["ce"] <= tau for c in wrong])); res["n_err"] = len(wrong)
            if arm == "i_production" and all("frozen" in D[k][c] for k, ids in (("sig_test", test_ids), ("sig_val", val_ids))
                                             for c in ids):      # B7 frozen-frame control (absent from the ensemble decodes)
                res["frozen_pass"] = {s: float(np.mean([D[k][c]["frozen"] <= tau for c in ids]))
                                      for s, k, ids in (("test", "sig_test", test_ids), ("val", "sig_val", val_ids))}
            # R within valid-length quartiles (§6 + B2): signer coverage and negative FV in the same bucket
            res["R_by_vT_quartile"] = {}
            for k in range(4):
                sk = [c for c in test_ids if qt(D[sig_t][c]["vT"]) == k]
                nk = [c for c in pt if qt(G[c]["vT"]) == k]
                Ck = float(np.mean([D[sig_t][c]["ce"] <= tau for c in sk])) if sk else None
                FVk = passes(nk)
                res["R_by_vT_quartile"][k] = {"n_sig": len(sk), "n_neg": len(nk), "C": Ck, "FV": FVk,
                                              "R": FVk / Ck if Ck and FVk is not None else None}
            strata = {"single_person": [c for c in pt if G[c].get("ndet_max", 9) <= 1],
                      "quiescent": [c for c in pt if wrist_speed(npz(c)) < ws],
                      "untracked": [c for c in pt if tracked_share(npz(c)) < 0.5],
                      "no_concurrent_other_signer": [c for c in pt if concur.get(pmap[c]) is False],
                      "concurrent_other_signer": [c for c in pt if concur.get(pmap[c]) is True],
                      "concurrent_unknown": [c for c in pt if concur.get(pmap[c]) is None]}
            res["strata_FV"] = {k: (len(v), passes(v)) for k, v in strata.items()}
            kept = [c for c in pt if tracked_share(npz(c)) >= 0.5]                    # untracked-exclusion sensitivity
            res["test_excl_untracked"] = gate_split([D[sig_t][c]["ce"] for c in test_ids], [vid[c] for c in test_ids],
                                                    [G[c]["ce"] for c in kept], [vid[pmap[c]] for c in kept], tau)
            sig_pass = [D[sig_t][c]["hyp"] for c in test_ids if D[sig_t][c]["ce"] <= tau]
            neg_pass = [G[c]["hyp"] for c in pt if G[c]["ce"] <= tau]
            res["outputs"] = {k: {"n": len(h), "words_hist": dict(sorted(Counter(len(x.split()) for x in h).items())),
                                  "verbatim_train": float(np.mean([x in train_sents for x in h])) if h else None,
                                  "top10": Counter(h).most_common(10)} for k, h in (("signers", sig_pass), ("negatives", neg_pass))}
            A[f"q{q}"] = res
        # paired delta and AUROC (test, primary population); pairs matched on valid length (B2)
        pairs = [(c, G[c]["ce"] - D[sig_t][pmap[c]]["ce"]) for c in pt if abs(G[c]["vT"] - D[sig_t][pmap[c]]["vT"]) <= 2]
        pv = [vid[pmap[c]] for c, _ in pairs]
        by_v = {}
        for (c, x), v in zip(pairs, pv):
            by_v.setdefault(v, []).append(x)
        A["paired"] = {"n": len(pairs), "excluded_length_mismatch": len(pt) - len(pairs),
                       "claim_valid": rep["G4_length_mismatch"]["paired_claim_valid"],
                       "share_neg_less_confident": cluster_boot(pv, lambda vs: float(np.mean([x > 0 for v in vs for x in by_v[v]]))),
                       "median_delta": cluster_boot(pv, lambda vs: float(np.median([x for v in vs for x in by_v[v]])))}
        ts = {v: [] for v in set(vid[c] for c in test_ids) | set(vid[pmap[c]] for c in pt)}
        tn = {v: [] for v in ts}
        for c in test_ids:
            ts[vid[c]].append(D[sig_t][c]["ce"])
        for c in pt:
            tn[vid[pmap[c]]].append(G[c]["ce"])
        A["auroc"] = cluster_boot(list(ts), lambda vs: auroc([x for v in vs for x in ts[v]], [x for v in vs for x in tn[v]])
                                  if any(tn[v] for v in vs) else None)
    rep["final_reading"] = min((rep[a]["q0.5"]["reading"] for a in ("i_production", "ii_tracked") if a in rep), key=RANK.index)
    json.dump(rep, open(out, "w"), indent=1, default=str)   # before anything that can fail
    # B4 audit lists (rendering is done separately, by eye); a short pool is reported, never a crash
    tau = rep["i_production"]["q0.5"]["tau"]
    pt = prim("test")
    pools = {"passing": [c for c in pt if G[c]["ce"] <= tau], "declined": [c for c in pt if G[c]["ce"] > tau]}
    rep["B4_audit"] = {k: random.Random(0).sample(v, min(30, len(v))) for k, v in pools.items()}
    rep["B4_audit_shortfall"] = {k: max(0, 30 - len(v)) for k, v in pools.items()}
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("zero_person_input", "no_person_detected", "G4_length_mismatch", "B3",
                                          "final_reading")}, indent=1, default=str))
    for arm in ("i_production", "ii_tracked"):
        if arm in rep:
            for q in QS:
                r = rep[arm][f"q{q}"]
                print(arm, q, {s: {k: r[s][k] for k in ("C", "FV", "R", "R_lo", "R_hi", "reading")} for s in ("test", "dev")},
                      "C_err", r["C_err"])
    return rep


def selftest():
    k = np.zeros((5, 133, 3), np.float32)
    k[:, 5, :2], k[:, 6, :2], k[:, :, 2] = (10, 0), (20, 0), 1.0
    k[:, 9, :2] = k[:, 10, :2] = [[0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]
    assert skel_dist(k, k) == 0.0 and valid_T(k) == 5 and tracked_share(k) == 1.0
    assert abs(wrist_speed(k) - 0.1) < 1e-6                          # 1 px/frame over a 10 px shoulder width
    k2 = k.copy(); k2[..., :2] = k2[..., :2] * 3 + 7                  # same person, other crop window and scale
    assert abs(skel_dist(k, k2)) < 1e-5
    z = k.copy(); z[2] = 0
    assert valid_T(z) == 4 and frozen(z).shape == z.shape and np.allclose(frozen(z)[0], z[3])
    assert auroc([0.1, 0.2], [0.9, 1.0]) == 1.0 and auroc([1.0], [0.1]) == 0.0 and auroc([0.5], [0.5]) == 0.5
    assert category(0.1, 0.05, 0.3) == "WORKS (strong)" and category(0.4, 0.2, 0.9) == "WORKS"
    assert category(0.7, 0.4, 0.95) == "PARTIAL" and category(0.9, 0.6, 1.2) == "FAILS" and category(1.5, 1.1, 2) == "INVERTED"
    assert min("WORKS", "INVERTED", key=RANK.index) == "INVERTED"
    g = gate_split([0.1, 0.2, 0.9, 1.0], ["a", "a", "b", "b"], [0.1, 2.0], ["a", "b"], 0.5)
    assert g["C"] == 0.5 and g["FV"] == 0.5 and g["R"] == 1.0
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("selftest", "decode", "analyse"))
    {"selftest": selftest, "decode": decode, "analyse": analyse}[ap.parse_args().stage]()
