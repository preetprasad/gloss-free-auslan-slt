"""Wave 12d: does a power mean of the members' probabilities decode better than the geometric mean? VALIDATION ONLY.

Pre-registration: results/PREREG_WAVE12D_POWERMEAN_2026-10-13.md (grid, gates, criterion and decision rule are fixed
there; the constants below are that grid). Decoding is the reported ensemble's own (ensemble_mbr_wer.ensemble_nbest,
candidate chosen by raw / words**LP); the only thing that moves is ensemble_nbest's `power` argument.

  decode one rule:  python tools/w12d_powermean.py decode --ckpts a.pt,..,f.pt --manifest val.tsv --rgb_feat_dir DIR \
                        --out OUT --power 1.0            (writes OUT/nbest_p1.json; computes no score)
  read-out:         python tools/w12d_powermean.py report --out OUT --prior_dir DIR_OF_JOB_120153 [--train_manifest F]
  self-test:        python tools/w12d_powermean.py self_test

The read-out prints one gate line and stops unless the control (p = 0) reproduces the recorded validation score.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

POWERS = [0.0, -1.0, 0.5, 1.0, 2.0]                      # 0 = geometric mean, the control
LPS = [round(0.1 * i, 1) for i in range(11)] + [1.2, 1.4, 1.6, 1.8, 2.0]
LP_REPORTED, EXPECT_BLEU, EXPECT_WER = 0.3, 12.470, 0.9164
ALPHA = 0.05 / (len(POWERS) - 1)                         # Bonferroni over the four comparisons


def nbest_file(out, power):
    return os.path.join(out, f"nbest_p{power:g}.json")


def pick(cands, lp):
    """The reported tool's own choice (ensemble_mbr_wer.py, LP sweep): first candidate wins ties."""
    return max(cands, key=lambda c: c[0] / (max(1, len(c[1].split())) ** lp))[1]


# ---------------------------------------------------------------------------------------------------- decode
def decode(a):
    import torch
    from torch.utils.data import DataLoader
    from config import Config
    from slt.data import Vocab
    from slt.preproc_guard import assert_preproc_matches_checkpoints
    from slt.rgb_fusion import RGBDataset, _rgb_path, collate_rgb
    from ensemble_decode import Member
    from ensemble_mbr_wer import _align_vocabs, ensemble_nbest

    base = os.path.basename(a.manifest)
    assert "val" in base and "test" not in base, f"validation only: refusing manifest {base!r}"
    assert a.power in POWERS, f"p = {a.power} is not in the pre-registered grid {POWERS}"
    assert os.path.isdir(a.rgb_feat_dir), f"rgb_feat_dir {a.rgb_feat_dir!r} is not a directory"
    cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame", rgb_feat_dir=a.rgb_feat_dir,
                 rgb_feature_dim=1152, test_manifest=a.manifest, device="cpu")
    dev = torch.device("cpu")
    paths = a.ckpts.split(",")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    assert_preproc_matches_checkpoints(cfg, cks, False)
    vocab = Vocab(cks[0]["vocab"])
    members = _align_vocabs(cks, [Member(c, cfg, Vocab(c["vocab"]), dev) for c in cks],
                            list(cks[0]["vocab"]), dev, False)
    ds = RGBDataset(a.manifest, vocab, cfg, "")
    if a.limit: ds.samples = ds.samples[:a.limit]
    # the loader replaces a missing or unreadable RGB file by one zero frame without a word: refuse instead
    bad = [c for c, _ in ds.samples if ds._load_feature(_rgb_path(c, ds.rgb_dir), ds.rgb_dim) is None]
    assert not bad, f"{len(bad)} clips have no usable RGB feature file (a fusion member would see a zero frame): {bad[:3]}"
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_rgb)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    print(f"[w12d] p = {a.power:g} | members {len(members)} fusion={[m.fusion for m in members]} | clips {len(ds)} "
          f"| beam {a.beam} | threads {torch.get_num_threads()}", flush=True)

    nbest, refs = [], []
    for bi, (src, src_pad, rgb, rgb_pad, _t, texts) in enumerate(dl):
        with torch.no_grad():
            mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in members]
            for cands in ensemble_nbest(members, mems, mx, a.beam, dev, None, a.power):
                nbest.append([[raw, vocab.decode(seq)] for raw, seq in cands])
        refs += list(texts)
        if bi % 10 == 0: print(f"[w12d] batch {bi + 1}/{len(dl)}", flush=True)
    # a power mean of probabilities is a probability, so every summed log score is finite and <= 0
    assert all(c and all(np.isfinite(r) and r <= 1e-3 for r, _ in c) for c in nbest), "empty or non-finite candidate list"

    os.makedirs(a.out, exist_ok=True)
    f = nbest_file(a.out, a.power)
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    json.dump({"power": a.power, "ckpts": paths, "ckpt_md5": [md5(p) for p in paths],
               # the read-out refuses rules decoded by different code (a partial re-run after a code change)
               "code_md5": [md5(ROOT / f) for f in ("tools/ensemble_mbr_wer.py", "ensemble_decode.py", "tools/w12d_powermean.py")],
               "manifest": a.manifest, "beam": a.beam, "limit": a.limit, "refs": refs, "nbest": nbest}, open(f, "w"))
    print(f"[w12d] wrote {f} ({len(nbest)} clips, {sum(map(len, nbest)) / len(nbest):.1f} candidates per clip)", flush=True)


# ---------------------------------------------------------------------------------------------------- scoring
def sent_stats(hyp, ref, cache={}):
    """sacreBLEU's per-sentence BLEU statistics for one detokenised pair: [hyp_len, ref_len, 4 matches, 4 totals]."""
    if (hyp, ref) not in cache:
        import sacrebleu
        s = sacrebleu.sentence_bleu(hyp, [ref], tokenize="13a")
        cache[hyp, ref] = [s.sys_len, s.ref_len, *s.counts, *s.totals]
    return cache[hyp, ref]


def bleu_from_stats(S):
    """Corpus BLEU-4 from statistics summed over sentences (last axis = the 10 numbers of sent_stats).

    Where some n-gram order has no match the result is 0 (BLEU without smoothing; sacreBLEU would smooth instead).
    The read-out compares with sacreBLEU only where every order has a match and names any other cell."""
    c, t = S[..., 2:6], S[..., 6:10]
    with np.errstate(divide="ignore", invalid="ignore"):
        b = 100 * np.exp(np.log(c / t).mean(-1) + np.minimum(0.0, 1.0 - S[..., 1] / S[..., 0]))
    return np.where((c > 0).all(-1), b, 0.0)


def criteria(ST, n_boot, n_halv, seed=0):
    """ST[rule, lp, sentence, 10], rule 0 = control. Returns the pre-registered quantities for rules 1.. .

    'chosen LP' = argmax of BLEU over the LP axis on the clips in question (np.argmax: lowest LP wins ties).
    """
    R, L, n, _ = ST.shape
    full = bleu_from_stats(ST.sum(2))                                   # (R, L)
    rng = np.random.default_rng(seed)
    B = np.empty((n_boot, R, L))
    for b in range(n_boot):                                             # paired: one resample for every rule and LP
        B[b] = bleu_from_stats(np.tensordot(ST, np.bincount(rng.integers(0, n, n), minlength=n), axes=([2], [0])))
    rng = np.random.default_rng(seed)
    H = np.empty((n_halv, R)); CH = np.empty((n_halv, R), dtype=int)
    for k in range(n_halv):
        perm = rng.permutation(n)
        CH[k] = bleu_from_stats(ST[:, :, perm[:n // 2]].sum(2)).argmax(1)       # LP chosen on half A
        H[k] = bleu_from_stats(ST[:, :, perm[n // 2:]].sum(2))[np.arange(R), CH[k]]   # scored on half B
    i03 = LPS.index(LP_REPORTED) if L == len(LPS) else 0
    out = []
    for r in range(1, R):
        d = float(full[r].max() - full[0].max())
        dist = B[:, r].max(1) - B[:, 0].max(1)                          # LP re-chosen inside every resample
        lo, hi = np.percentile(dist, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        lo95, hi95 = np.percentile(dist, [2.5, 97.5])
        hd = H[:, r] - H[:, 0]
        agree = float(np.mean(np.sign(hd) == np.sign(d))) if d else 0.0
        fixed = B[:, r, i03] - B[:, 0, i03]
        out.append({"delta": d, "bonf": [float(lo), float(hi)], "ci95": [float(lo95), float(hi95)],
                    "p_ahead_boot": float(np.mean(dist > 0)),
                    "half_mean": float(hd.mean()), "half_pct": [float(x) for x in np.percentile(hd, [2.5, 97.5])],
                    "half_agree": agree, "half_ahead": float(np.mean(hd > 0)),
                    "delta_fixed": float(full[r, i03] - full[0, i03]),
                    "ci95_fixed": [float(x) for x in np.percentile(fixed, [2.5, 97.5])],
                    # the interval must exclude zero on D_p's own side (pre-registration, section 5)
                    "different": bool(((lo > 0 and d > 0) or (hi < 0 and d < 0)) and agree >= 0.95)})
    return full, out, CH


def report(a):
    from ensemble_mbr_wer import corpus_scores
    from slt.metrics import corpus_bleu4, detokenize
    from arm_readouts import readouts
    from stitch_lexicon import STOP

    D = {}
    for p in POWERS:
        assert os.path.exists(nbest_file(a.out, p)), f"missing decode for p = {p:g}: {nbest_file(a.out, p)}"
        D[p] = json.load(open(nbest_file(a.out, p)))
        assert D[p]["power"] == p and not D[p]["limit"], f"p = {p:g}: wrong or truncated decode file"
        for k in ("refs", "ckpt_md5", "code_md5", "beam", "manifest"):
            assert D[p][k] == D[0.0][k], f"p = {p:g}: {k} differs from the control's"
    refs = D[0.0]["refs"]; n = len(refs)
    assert "val" in os.path.basename(D[0.0]["manifest"]) and "test" not in os.path.basename(D[0.0]["manifest"])
    hyps = {p: {lp: [pick(c, lp) for c in D[p]["nbest"]] for lp in LPS} for p in POWERS}

    # ---- gates 1 and 2: nothing else is printed unless the control is the reported ensemble ----
    bleu, wer, *_ = corpus_scores(hyps[0.0][LP_REPORTED], refs)
    ok = abs(bleu - a.expect) < 0.0005 and abs(wer - a.expect_wer) < 0.00005
    same = "not checked"
    if a.prior_dir:
        prior = [d for d in map(lambda f: json.load(open(f)), sorted(glob.glob(os.path.join(a.prior_dir, "shard*.json"))))
                 if "63" in d["hyps"] and d["n"] == 6 and d["beam"] == D[0.0]["beam"] and d["lp"] == LP_REPORTED]
        k = -1
        if len(prior) == 1 and prior[0]["refs"] == refs and prior[0]["ckpt_md5"] == D[0.0]["ckpt_md5"]:
            k = sum(x == y for x, y in zip(prior[0]["hyps"]["63"], hyps[0.0][LP_REPORTED]))
        same = f"{k}/{n}"; ok = ok and k == n
    print(f"GATE {'PASSED' if ok else 'FAILED'}: control (p = 0) at LP {LP_REPORTED}: BLEU-4 {bleu:.3f} WER {wer:.4f} | "
          f"recorded {a.expect:.3f} {a.expect_wer:.4f} | clips {n} | beam {D[0.0]['beam']} | hypotheses identical to job 120153: {same}")
    if not ok: sys.exit(1)

    # ---- gate 4: the bootstrap's BLEU is sacreBLEU's ----
    R_ = [detokenize(r) for r in refs]
    ST = np.array([[[sent_stats(detokenize(h), r) for h, r in zip(hyps[p][lp], R_)] for lp in LPS] for p in POWERS], float)
    exact = np.array([[corpus_bleu4([detokenize(h) for h in hyps[p][lp]], R_, "13a") for lp in LPS] for p in POWERS])
    stat = bleu_from_stats(ST.sum(2)); zero = stat == 0
    assert not zero[0].any(), "the control has a length penalty with no 4-gram match: not the reported ensemble"
    gap = np.abs(stat - exact)[~zero].max()
    assert gap < 1e-6, f"BLEU from summed statistics differs from sacreBLEU by {gap}"
    full, crit, CH = criteria(ST, a.boot, a.halvings)
    for i, j in zip(*np.nonzero(zero)):
        print(f"NOTE: p = {POWERS[i]:g} at LP {LPS[j]:g} has an n-gram order without a match on the {n} clips: scored 0 "
              f"here (sacreBLEU with smoothing: {exact[i, j]:.3f})")

    name = lambda p: "geometric (p = 0)" if p == 0 else f"p = {p:g}"
    print(f"\nValidation BLEU-4 by length penalty (rows) and averaging rule (columns); {n} clips, no test clip decoded")
    print(f"{'LP':>5s} " + " ".join(f"{name(p):>18s}" for p in POWERS))
    for j, lp in enumerate(LPS):
        print(f"{lp:5.1f} " + " ".join(f"{full[i, j]:18.3f}" for i in range(len(POWERS))))
    best = full.argmax(1)
    edge = [("  (END OF GRID)" if b in (0, len(LPS) - 1) else "") for b in best]

    train = None
    if a.train_manifest:
        train = {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split())
                 for l in open(a.train_manifest, encoding="utf-8") if "\t" in l}
    sec = {}
    for i, p in enumerate(POWERS):
        for tag, lp in (("own", LPS[best[i]]), ("fixed", LP_REPORTED)):
            h = hyps[p][lp]; r = readouts(h, refs, STOP, train); _b, _w, S, Dl, I, _u = corpus_scores(h, refs)
            ref0 = hyps[0.0][LPS[best[0]] if tag == "own" else LP_REPORTED]
            sec[p, tag] = {**r, "lp": lp, "S": S, "D": Dl, "I": I, "same_as_geometric": sum(x == y for x, y in zip(h, ref0)) / n}
    if train is not None:
        print(f"\n'unseen' = the {sec[0.0, 'own']['unseen_n']} of {n} validation clips whose reference is not a training target")
    for tag, title in (("own", "each rule at its validation-chosen length penalty"), ("fixed", f"every rule at LP {LP_REPORTED}")):
        print(f"\nRead-outs, {title}:")
        print(f"{'rule':>18s} {'LP':>4s} {'BLEU-4':>7s} {'WER':>7s} {'S':>5s} {'D':>5s} {'I':>5s} {'len':>5s} {'uniq':>5s} "
              f"{'c-prec':>6s} {'c-rec':>6s} {'unseen':>7s} {'=geo':>6s}")
        for i, p in enumerate(POWERS):
            s = sec[p, tag]
            print(f"{name(p):>18s} {s['lp']:4.1f} {s['bleu4']:7.3f} {s['wer']:7.4f} {s['S']:5d} {s['D']:5d} {s['I']:5d} "
                  f"{s['len_ratio']:5.2f} {s['distinct']:5d} {s['content_precision']:6.2f} {s['content_recall']:6.2f} "
                  f"{s.get('bleu4_unseen', float('nan')):7.3f} {100 * s['same_as_geometric']:5.1f}%"
                  + (edge[i] if tag == "own" else ""))

    print(f"\nPrimary criterion: BLEU-4 at own chosen LP minus the geometric mean's at its own chosen LP")
    print(f"  (i) {100 * (1 - ALPHA):g}% interval, {a.boot} paired resamples, LP re-chosen inside each; "
          f"(ii) LP chosen on half A, scored on half B, {a.halvings} halvings")
    for p, c in zip(POWERS[1:], crit):
        print(f"  {name(p):>8s}: {c['delta']:+.3f}  (i) [{c['bonf'][0]:+.3f}, {c['bonf'][1]:+.3f}] (95%: [{c['ci95'][0]:+.3f}, "
              f"{c['ci95'][1]:+.3f}])  (ii) half B {c['half_mean']:+.3f} [{c['half_pct'][0]:+.3f}, {c['half_pct'][1]:+.3f}], "
              f"same sign in {100 * c['half_agree']:.1f}% of halvings  ->  "
              + ("MEASURABLY DIFFERENT from the geometric mean on validation" if c["different"]
                 else "not distinguishable from the geometric mean on validation"))
    print(f"\nSecondary, both rules at LP {LP_REPORTED} (95% interval, not corrected):")
    for p, c in zip(POWERS[1:], crit):
        print(f"  {name(p):>8s}: {c['delta_fixed']:+.3f} [{c['ci95_fixed'][0]:+.3f}, {c['ci95_fixed'][1]:+.3f}]")
    print("\nLength penalty chosen on half A (count over the halvings):")
    for i, p in enumerate(POWERS):
        cnt = np.bincount(CH[:, i], minlength=len(LPS))
        print(f"  {name(p):>18s}: " + ", ".join(f"{LPS[j]:g}: {c}" for j, c in enumerate(cnt) if c))
    print("\nThe six members and LP 0.3 were chosen on this validation set under the geometric mean (pre-registration, section 6).")

    json.dump({"clips": n, "powers": POWERS, "lps": LPS, "bleu4": full.tolist(), "chosen_lp": [LPS[b] for b in best],
               "gate": {"bleu4": bleu, "wer": wer, "identical_to_120153": same},
               "criteria": {f"{p:g}": c for p, c in zip(POWERS[1:], crit)},
               "readouts": {f"{p:g}|{tag}": v for (p, tag), v in sec.items()},
               "boot": a.boot, "halvings": a.halvings, "seed": 0, "alpha_per_comparison": ALPHA},
              open(os.path.join(a.out, "w12d_scores.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(a.out, 'w12d_scores.json')}")


# ---------------------------------------------------------------------------------------------------- self-test
def self_test(_):
    import torch
    from slt.data import EOS
    from slt.metrics import corpus_bleu4
    from ensemble_mbr_wer import ensemble_nbest, power_mean_logp

    # 1. the formula, in float64, against a direct computation; limits and inequalities
    g = torch.Generator().manual_seed(0)
    lps = [torch.log_softmax(torch.randn(4, 50, generator=g, dtype=torch.float64) * 3, -1) for _ in range(6)]
    q = torch.stack(lps).exp()
    geo = sum(lps) / 6
    for w in ([1.0] * 6, [3.0, 1.0, 1.0, 0.5, 2.0, 1.0]):
        wt = torch.tensor(w, dtype=torch.float64).view(6, 1, 1) / sum(w)
        for p in (-1.0, 0.5, 1.0, 2.0):
            direct = ((wt * q ** p).sum(0)).log() / p
            assert torch.allclose(power_mean_logp(lps, w, p), direct, atol=1e-10), p
    assert torch.allclose(power_mean_logp(lps, [1.0] * 6, 1.0).exp().sum(-1), torch.ones(4, dtype=torch.float64))
    assert torch.allclose(power_mean_logp(lps, [1.0] * 6, 1e-6), geo, atol=1e-4)            # p -> 0 is the geometric mean
    chain = [power_mean_logp(lps, [1.0] * 6, -1.0), geo] + [power_mean_logp(lps, [1.0] * 6, p) for p in (0.5, 1.0, 2.0)]
    assert all((b >= a - 1e-12).all() for a, b in zip(chain, chain[1:])), "power means must not decrease with p"
    assert all((c <= 1e-12).all() for c in chain), "a mean of probabilities is at most 1"
    hole = [l.clone() for l in lps]; hole[0][:, 3] = float("-inf")                          # one member cannot say word 3
    assert all(not torch.isnan(power_mean_logp(hole, [1.0] * 6, p)).any() for p in (-1.0, 0.5, 1.0, 2.0))
    assert torch.isinf(power_mean_logp(hole, [1.0] * 6, -1.0)[:, 3]).all() and torch.isfinite(power_mean_logp(hole, [1.0] * 6, 1.0)).all()

    # 2. through the beam search, with stand-in members
    class Fake:
        fusion = False
        def __init__(self, seed):
            t = torch.randn(12, 12, generator=torch.Generator().manual_seed(seed)); t[:, EOS] += 1.0
            self.t = torch.log_softmax(t, -1)
        def step_logp(self, ys, mem, pad): return self.t[ys[:, -1]]
    mem = [(torch.zeros(2, 1, 1), torch.zeros(2, 1, dtype=torch.bool))]
    dev = torch.device("cpu")
    one = ensemble_nbest([Fake(1)], mem, 8, 4, dev)
    for p in POWERS[1:]:                                    # the power mean of ONE distribution is that distribution
        alt = ensemble_nbest([Fake(1)], mem, 8, 4, dev, None, p)
        assert [[s for _, s in c] for c in alt] == [[s for _, s in c] for c in one], p
        assert all(abs(x[0] - y[0]) < 1e-4 for c, d in zip(alt, one) for x, y in zip(c, d)), p
    two = [Fake(1), Fake(2)]
    assert ensemble_nbest(two, mem * 2, 8, 4, dev) == ensemble_nbest(two, mem * 2, 8, 4, dev, None, 0.0)   # 0 = off
    g0 = dict((tuple(s), r) for r, s in ensemble_nbest(two, mem * 2, 8, 4, dev)[0])
    a1 = dict((tuple(s), r) for r, s in ensemble_nbest(two, mem * 2, 8, 4, dev, None, 1.0)[0])
    assert a1 != g0 and all(a1[s] >= g0[s] - 1e-5 for s in a1 if s in g0), "arithmetic >= geometric on every shared path"

    # 3. candidate choice, statistics, criteria
    c = [(-1.0, "a"), (-1.5, "a b c"), (-1.0, "z")]
    assert pick(c, 0.0) == "a" and pick(c, 1.0) == "a b c"
    rng = np.random.default_rng(1); words = "the cat dog sat ran on mat a big red".split()
    refs = [" ".join(rng.choice(words, 8)) + "." for _ in range(80)]
    worse = [r if i % 3 else r.replace(" ", " x ", 1) for i, r in enumerate(refs)]
    st = lambda hyps: [sent_stats(h, r) for h, r in zip(hyps, refs)]
    short = [" ".join(r.split()[:6]) for r in refs]                                         # brevity penalty below 1
    for h in (worse, short):
        assert abs(bleu_from_stats(np.array(st(h), float).sum(0)) - corpus_bleu4(h, refs, "13a")) < 1e-9
    assert corpus_bleu4(short, refs, "13a") < 0.8 * bleu_from_stats(np.array(st(short), float).sum(0) * [1, 0, *[1] * 8])
    z = np.array([st(worse), st(refs)], float).sum(1); z[0, [5, 9]] = 0                     # row 0: no 4-gram at all (0/0)
    assert bleu_from_stats(z)[0] == 0 and abs(bleu_from_stats(z)[1] - 100) < 1e-9
    full, crit, _ = criteria(np.array([[st(worse)] * 2, [st(worse)] * 2, [st(refs)] * 2], float), 200, 100)
    assert crit[0]["delta"] == 0 and crit[0]["bonf"] == [0.0, 0.0] and not crit[0]["different"]     # same rule: no difference
    assert crit[1]["delta"] > 0 and crit[1]["bonf"][0] > 0 and crit[1]["half_agree"] == 1.0 and crit[1]["different"]
    assert abs(full[2, 0] - 100.0) < 1e-9
    _, neg, _ = criteria(np.array([[st(refs)] * 2, [st(worse)] * 2], float), 200, 100)
    assert neg[0]["delta"] < 0 and neg[0]["bonf"][1] < 0 and neg[0]["different"]                    # a worse rule is found too
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--ckpts", required=True); d.add_argument("--manifest", required=True)
    d.add_argument("--rgb_feat_dir", required=True); d.add_argument("--out", required=True)
    d.add_argument("--power", type=float, required=True); d.add_argument("--beam", type=int, default=10)
    d.add_argument("--limit", type=int, default=0, help="smoke tests only; the read-out refuses a truncated decode")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--prior_dir", default="", help="decode files of job 120153; the control must equal its six-member hypotheses")
    r.add_argument("--train_manifest", default="", help="for BLEU-4 on references that do not occur in training")
    r.add_argument("--expect", type=float, default=EXPECT_BLEU); r.add_argument("--expect_wer", type=float, default=EXPECT_WER)
    r.add_argument("--boot", type=int, default=2000); r.add_argument("--halvings", type=int, default=1000)
    sub.add_parser("self_test")
    a = ap.parse_args()
    {"decode": decode, "report": report, "self_test": self_test}[a.cmd](a)
