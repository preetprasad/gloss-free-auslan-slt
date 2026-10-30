"""Wave-8 learning-curve mechanism (results/PREREG_LEARNING_CURVE_MECHANISM_2026-09-28.md), zero GPU.

Reuses arm_readouts.content / readouts (the Counter& hit rule) and stitch_lexicon.STOP. GATE first: per-word pooled
recall must equal readouts()['content_recall'] to 1e-6 and match arm_table.txt c_rec (printed to 3 dp), and BLEU must
equal each dump's beam_result, for all 20 test runs (and all 20 val runs for Q4). Aborts on any failure.

  cd src && ../.venv/bin/python tools/learning_curve_mechanism.py [--selftest]
"""
from __future__ import annotations

import json, os, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_readouts import content, readouts, KEY  # noqa: E402
from stitch_lexicon import STOP                  # noqa: E402

W8 = Path(__file__).resolve().parents[2] / "results" / "real_data_learning_curve"
ARMS, SEEDS = ["R025", "R050", "R100", "V050"], [42, 101, 202, 303, 404]
BUCKETS = [(0, 0, "0"), (1, 2, "1-2"), (3, 10, "3-10"), (11, 50, "11-50"), (51, 200, "51-200"), (201, 10**12, "201+")]
NBOOT, BOOT_SEED, THR = 2000, 0, 1.0
norm = lambda t: " ".join(t.lower().split())


# ---------------------------------------------------------------- core statistics
def word_stats(hyps, refs, keep=None):
    """Per content word: hits (sum over clips of min(count hyp, count ref)) and reference occurrences."""
    hits, occ = Counter(), Counter()
    for i, (h, r) in enumerate(zip(hyps, refs)):
        if keep is not None and i not in keep:
            continue
        ch, cr = Counter(content(h, STOP)), Counter(content(r, STOP))
        hits.update(ch & cr); occ.update(cr)
    return hits, occ


def pooled(hits, occ):
    return 100 * sum(hits.values()) / max(sum(occ.values()), 1)


def wcorr(x, y, w):
    w = w / w.sum(); mx, my = (w * x).sum(), (w * y).sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def fit_logistic(x, succ, trials, iters=100):
    """Binomial logistic regression hit ~ 1 + x by Newton/IRLS (sklearn absent in .venv)."""
    X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        step = np.linalg.solve(X.T @ (X * (trials * p * (1 - p))[:, None]), X.T @ (succ - trials * p))
        b += step
        if np.abs(step).max() < 1e-12:
            break
    return b


def predict_recall(b, counts, occ):
    ws = list(occ); x = np.log1p([counts.get(w, 0) for w in ws]); n = np.array([occ[w] for w in ws], float)
    return 100 * (n / (1 + np.exp(-(b[0] + b[1] * x)))).sum() / n.sum()


# ---------------------------------------------------------------- data
def train_counts(path):
    c = Counter()
    for l in open(path, encoding="utf-8"):
        if "\t" in l:
            c.update(content(l.rstrip("\n").split("\t", 1)[1], STOP))
    return c


def recorded_crec():
    rows = {}
    for l in open(W8 / "arm_table.txt"):
        p = l.split()
        if p and "_s" in p[0] and p[0].split("_s")[0] in ARMS:
            rows[p[0]] = float(p[10])          # columns: val val_soup test test_soup unseen seen_own wer len c_prec c_rec
    return rows


def load(split):
    full = {norm(l.rstrip("\n").split("\t", 1)[1]) for l in open(W8 / "train.tsv", encoding="utf-8") if "\t" in l}
    rec = recorded_crec() if split == "test" else {}
    runs, fails = {}, []
    for a in ARMS:
        for s in SEEDS:
            j = json.load(open(W8 / "runs" / f"{a}_s{s}" / f"beam5_{split}_best.json"))
            h, r = j["beam_hyps"], j["refs"]
            ro = readouts(h, r, STOP, full)
            hits, occ = word_stats(h, r)
            pr, tag = pooled(hits, occ), f"{a}_s{s}"
            if abs(ro["bleu4"] - j["beam_result"]["metrics"][KEY]) > 1e-6:
                fails.append(f"{split} {tag}: BLEU {ro['bleu4']} != recorded {j['beam_result']['metrics'][KEY]}")
            if abs(pr - ro["content_recall"]) > 1e-6:
                fails.append(f"{split} {tag}: pooled recall {pr} != readouts {ro['content_recall']}")
            if split == "test" and abs(pr - rec[tag]) > 5e-4 + 1e-9:
                fails.append(f"test {tag}: pooled recall {pr:.6f} != arm_table.txt {rec[tag]:.3f}")
            unseen = {i for i, x in enumerate(r) if norm(x) not in full}
            runs[(a, s)] = {"hyps": h, "refs": r, "unseen": unseen, "recall": pr, "bleu": ro["bleu4"],
                            "counts": train_counts(W8 / "manifests" / f"{tag}.tsv")}
    return runs, fails


# ---------------------------------------------------------------- analyses
def q1(runs, keep_unseen=False):
    out = {}
    for a in ARMS:
        H, N = Counter(), Counter()
        for s in SEEDS:
            R = runs[(a, s)]; hits, occ = word_stats(R["hyps"], R["refs"], R["unseen"] if keep_unseen else None)
            for w, n in occ.items():
                c = R["counts"].get(w, 0); lab = next(b[2] for b in BUCKETS if b[0] <= c <= b[1])
                H[lab] += hits[w]; N[lab] += n
        tot = sum(N.values())
        out[a] = {lab: {"recall": 100 * H[lab] / N[lab] if N[lab] else None, "ref_tokens": N[lab],
                        "share": 100 * N[lab] / tot} for _, _, lab in BUCKETS}
    return out


def q2a(runs, keep_unseen=False):
    rows = []                                   # (word, seed, d_count, d_recall, weight)
    per_seed = {}
    for s in SEEDS:
        V, R = runs[("V050", s)], runs[("R050", s)]
        kv = V["unseen"] if keep_unseen else None
        hv, ov = word_stats(V["hyps"], V["refs"], kv); hr, orr = word_stats(R["hyps"], R["refs"], kv)
        assert ov == orr, "refs differ between V050 and R050"
        ws = sorted(ov)
        dc = np.array([np.log1p(V["counts"].get(w, 0)) - np.log1p(R["counts"].get(w, 0)) for w in ws])
        dr = np.array([hv[w] / ov[w] - hr[w] / ov[w] for w in ws]); wt = np.array([ov[w] for w in ws], float)
        per_seed[s] = {"r": float(wcorr(dc, dr, wt)), "n_words": len(ws),
                       "frac_words_count_changed": float((dc != 0).mean())}
        rows += [(w, s, c, d, x) for w, c, d, x in zip(ws, dc, dr, wt)]
    words = sorted({r[0] for r in rows}); idx = {w: i for i, w in enumerate(words)}
    wi = np.array([idx[r[0]] for r in rows]); x, y, wt = (np.array([r[k] for r in rows], float) for k in (2, 3, 4))
    groups = [np.flatnonzero(wi == i) for i in range(len(words))]
    est = float(wcorr(x, y, wt)); rng = np.random.default_rng(BOOT_SEED); bs = []
    for _ in range(NBOOT):
        sel = np.concatenate([groups[i] for i in rng.integers(0, len(words), len(words))])
        bs.append(wcorr(x[sel], y[sel], wt[sel]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"r": est, "ci95": [float(lo), float(hi)], "n_words": len(words), "n_obs": len(rows), "per_seed": per_seed}


def q2b(runs, keep_unseen=False):
    xs, ys, ns, stats = [], [], [], {}
    for (a, s), R in runs.items():
        hits, occ = word_stats(R["hyps"], R["refs"], R["unseen"] if keep_unseen else None)
        stats[(a, s)] = (hits, occ)
        if a in ("R025", "R050"):
            for w, n in occ.items():
                xs.append(np.log1p(R["counts"].get(w, 0))); ys.append(hits[w]); ns.append(n)
    b = fit_logistic(np.array(xs), np.array(ys, float), np.array(ns, float))
    per = {}
    for (a, s), (hits, occ) in stats.items():
        per.setdefault(a, {})[s] = {"obs": pooled(hits, occ), "pred": predict_recall(b, runs[(a, s)]["counts"], occ)}
    mean = {a: {k: float(np.mean([per[a][s][k] for s in SEEDS])) for k in ("obs", "pred")} for a in ARMS}
    r100 = {s: {**per["R100"][s], "resid": per["R100"][s]["obs"] - per["R100"][s]["pred"]} for s in SEEDS}
    resid = float(np.mean([v["resid"] for v in r100.values()]))
    return {"beta": b.tolist(), "per": {a: {str(s): v for s, v in d.items()} for a, d in per.items()}, "mean": mean,
            "R100": {str(s): v for s, v in r100.items()}, "resid_mean": resid,
            "resid_sd": float(np.std([v["resid"] for v in r100.values()], ddof=1))}


def reading(a, b):
    pos = a["ci95"][0] > 0; res = b["resid_mean"]
    if pos and abs(res) <= THR:
        return "COUNT-DRIVEN"
    if res > THR:
        return "MIXED" if pos else "GENERAL"
    if not (a["ci95"][0] <= 0 <= a["ci95"][1]) or res < -THR:
        return "UNMATCHED (no pre-stated reading covers this case)"
    return "NEITHER"


def split_change(b):
    m = b["mean"]; obs = m["R100"]["obs"] - m["R050"]["obs"]; pr = m["R100"]["pred"] - m["R050"]["pred"]
    return {"observed": obs, "count_predicted": pr, "residual_part": obs - pr}


def q3(b):
    m = b["mean"]; beta = b["beta"]
    curve = {c: 100 / (1 + np.exp(-(beta[0] + beta[1] * np.log1p(c)))) for c in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)}
    return {"pred_R025_R050": m["R050"]["pred"] - m["R025"]["pred"], "pred_R050_R100": m["R100"]["pred"] - m["R050"]["pred"],
            "obs_R025_R050": m["R050"]["obs"] - m["R025"]["obs"], "obs_R050_R100": m["R100"]["obs"] - m["R050"]["obs"],
            "p_at_count": {str(k): float(v) for k, v in curve.items()}}


def median_token_count(runs, arm):
    v = []
    for s in SEEDS:
        R = runs[(arm, s)]; _, occ = word_stats(R["hyps"], R["refs"])
        for w, n in occ.items():
            v += [R["counts"].get(w, 0)] * n
    return float(np.median(v))


# ---------------------------------------------------------------- selftest
def selftest():
    h, o = word_stats(["cat cat dog .", "bird ."], ["cat dog dog .", "fish ."])
    assert h == Counter({"cat": 1, "dog": 1}) and o == Counter({"cat": 1, "dog": 2, "fish": 1}), (h, o)
    assert abs(pooled(h, o) - 50.0) < 1e-12
    assert abs(readouts(["cat cat dog .", "bird ."], ["cat dog dog .", "fish ."], STOP)["content_recall"] - 50.0) < 1e-12
    x = np.arange(5.0); assert abs(wcorr(x, 2 * x + 1, np.ones(5)) - 1) < 1e-12
    assert abs(wcorr(x, -x, np.array([1, 2, 3, 4, 5.])) + 1) < 1e-12
    xs = np.linspace(0, 6, 25); n = np.full(25, 1e4); p = 1 / (1 + np.exp(-(-3 + 0.7 * xs)))
    b = fit_logistic(xs, n * p, n); assert np.allclose(b, [-3, 0.7], atol=1e-8), b
    assert abs(predict_recall(np.array([0.0, 0.0]), Counter(), Counter({"a": 3, "b": 1})) - 50) < 1e-12
    print("selftest passed")


# ---------------------------------------------------------------- main
def f(v):
    return " / ".join(f"{x:+.3f}" if isinstance(x, float) else str(x) for x in v)


def main():
    out, lines = {}, []
    P = lambda *a: (lines.append(" ".join(str(x) for x in a)), print(*a))
    runs, fails = load("test"); vruns, vfails = load("val")
    fails += vfails
    if fails:
        print("GATE FAILED:\n  " + "\n  ".join(fails)); sys.exit(1)
    assert all(len(R["unseen"]) == 570 for R in runs.values())
    P("GATE PASSED: 40 dumps (20 test + 20 val) -- per-word pooled recall == readouts content_recall to 1e-6,")
    P("  test recall == arm_table.txt c_rec (3 dp), BLEU == beam_result to 1e-6; unseen_n == 570 in every test run.")
    out["gate"] = {"passed": True, "test_recall": {f"{a}_s{s}": runs[(a, s)]["recall"] for a, s in runs},
                   "test_bleu": {f"{a}_s{s}": runs[(a, s)]["bleu"] for a, s in runs}}

    P("\n=== Q1 pooled per-word recall (%) by own training-count bucket, pooled over 5 seeds (ref-token share %) ===")
    out["Q1"] = q1(runs)
    P("arm   " + "".join(f"{lab:>18s}" for _, _, lab in BUCKETS))
    for a in ARMS:
        P(f"{a}  " + "".join(f"{(v['recall'] or 0):8.2f} ({v['share']:5.1f}%)  " for v in out["Q1"][a].values()))
    for a in ARMS:
        P(f"  median own-train count of a test reference content token, {a}: {median_token_count(runs, a):.0f}")

    def q2_block(name, R, keep_unseen):
        a = q2a(R, keep_unseen); b = q2b(R, keep_unseen); rd = reading(a, b)
        P(f"\n=== Q2 [{name}] ===")
        P(f"(a) weighted corr(d log(1+count), d recall), V050 - R050, pooled over seeds: r = {a['r']:+.4f}  "
          f"95% word-bootstrap [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]  ({a['n_words']} words, {a['n_obs']} word-seed obs, {NBOOT} resamples, seed {BOOT_SEED})")
        P("    per seed 42/101/202/303/404: r " + f([a["per_seed"][s]["r"] for s in SEEDS])
          + " | share of words whose count changed " + f([round(a["per_seed"][s]["frac_words_count_changed"], 3) for s in SEEDS]))
        P(f"(b) logistic hit ~ log(1+count), fitted on R025+R050: b0 {b['beta'][0]:+.4f}, b1 {b['beta'][1]:+.4f}")
        for arm in ARMS:
            P(f"    {arm}: obs " + f([b['per'][arm][str(s)]['obs'] for s in SEEDS]) + f" (mean {b['mean'][arm]['obs']:.3f})"
              + " | pred " + f([b['per'][arm][str(s)]['pred'] for s in SEEDS]) + f" (mean {b['mean'][arm]['pred']:.3f})")
        P("    R100 residual (obs - pred) per seed: " + f([b['R100'][str(s)]['resid'] for s in SEEDS])
          + f" | mean {b['resid_mean']:+.3f} (sd {b['resid_sd']:.3f})")
        sc = split_change(b)
        P(f"    R050 -> R100 change: observed {sc['observed']:+.3f} = count-predicted {sc['count_predicted']:+.3f} + residual {sc['residual_part']:+.3f}")
        P(f"READING [{name}]: {rd}")
        return {"a": a, "b": b, "reading": rd, "split_R050_R100": sc}

    out["Q2"] = q2_block("primary: official test, all 800 clips", runs, False)
    q = q3(out["Q2"]["b"]); out["Q3"] = q
    P("\n=== Q3 why the second doubling moved more (Q2(b) model, mean over seeds, recall points) ===")
    P(f"predicted R025->R050 {q['pred_R025_R050']:+.3f} vs observed {q['obs_R025_R050']:+.3f}")
    P(f"predicted R050->R100 {q['pred_R050_R100']:+.3f} vs observed {q['obs_R050_R100']:+.3f}")
    P("fitted P(hit) at training count: " + ", ".join(f"{k}:{v:.2f}%" for k, v in q["p_at_count"].items()))
    out["Q4_unseen"] = q2_block("Q4 robustness: unseen-570 test clips", runs, True)
    out["Q4_val"] = q2_block("Q4 robustness: validation dumps, all 800 clips", vruns, False)

    (W8 / "mechanism_output.txt").write_text("\n".join(lines) + "\n")
    (W8 / "mechanism_output.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
