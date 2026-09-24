#!/usr/bin/env python3
"""Decision readout for results/PREREG_SEEDENS_2026-09-24.md (job 21736, jobs/slurm/seedens.sbatch).

For every system (single seed sK, or ensemble ensN) in every batch directory under --out, reads the dumped LP sweep
(<sys>_<split>/hyps_LP=<lp>.jsonl), re-scores every LP through slt/metrics.py and asserts it equals the BLEU printed in
<sys>_<split>.log (gate 4). Picks each system's LP as its VALIDATION argmax (first maximum in grid order), then reads
test at that LP. Estimand per batch: ens - (member with the highest validation BLEU). Applies the pre-registered rule.

  python tools/seedens_analysis.py --out <dir with pa_aug/ w1A/ ... vdB/> --train_manifest results/_kaya_train_manifest.tsv
  python tools/seedens_analysis.py --selftest
"""
import argparse, json, os, re, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from slt.metrics import corpus_bleu4, detokenize   # noqa: E402

LPS = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
OFFICIAL = ["pa_aug", "w1A", "w3C", "r3_111", "a3ctl", "w4A", "w5C"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
norm = lambda s: " ".join(s.lower().split())


def load(path):
    rows = sorted((json.loads(l) for l in open(path, encoding="utf-8")), key=lambda r: r["idx"])
    # detokenise both sides before 13a, exactly as slt.metrics.corpus_scores does for the logged numbers
    return [detokenize(r["hyp_beam"]) for r in rows], [detokenize(r["ref"]) for r in rows]


def log_scores(path):
    out = {}
    for l in open(path):
        m = re.match(r"\s+LP=([0-9.]+)\s+BLEU-4=\s*([0-9.]+)", l)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def system(bdir, name):
    """Return dict(lp, val, test, test_hyps, refs) for one system, after gate-4 re-scoring."""
    res = {}
    for split in ("val", "test"):
        logged = log_scores(os.path.join(bdir, f"{name}_{split}.log"))
        scores = {}
        for lp in LPS:
            h, r = load(os.path.join(bdir, f"{name}_{split}", f"hyps_LP={lp}.jsonl"))
            b = corpus_bleu4(h, r, "13a")
            assert abs(b - logged[lp]) < 0.0015, f"gate 4: {bdir}/{name}_{split} LP {lp} rescored {b:.3f} vs log {logged[lp]}"
            scores[lp] = (b, h, r)
        res[split] = scores
    lp = max(LPS, key=lambda x: (res["val"][x][0], -LPS.index(x)))      # validation argmax, first in grid order
    return dict(lp=lp, val=res["val"][lp][0], test=res["test"][lp][0], hyps=res["test"][lp][1], refs=res["test"][lp][2])


def interval(d):
    m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5
    return m, m - hw, m + hw


def batch(bdir):
    names = sorted({f[:-8] for f in os.listdir(bdir) if f.endswith("_val.log")})
    ens = [n for n in names if n.startswith("ens")]; singles = [n for n in names if n.startswith("s")]
    assert len(ens) == 1 and len(singles) >= 3, f"{bdir}: systems {names}"
    S = {n: system(bdir, n) for n in singles + ens}
    best = max(singles, key=lambda n: S[n]["val"])
    return S, ens[0], best, singles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--train_manifest", default="")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        h, r = load(os.path.join(HERE, "..", "results", "final_ensemble_test_decodes", "hyps_LP=0.3.jsonl"))
        b = corpus_bleu4(h, r, "13a"); assert abs(b - 11.802) < 0.001, b
        assert interval([1.0, 2.0, 3.0])[1] < 1.0 < 2.0 < interval([1.0, 2.0, 3.0])[2]
        print(f"selftest OK: dump reader + metric reproduce S6 {b:.3f}"); return
    from scipy import stats
    train = {norm(detokenize(l.split("\t", 1)[1])) for l in open(a.train_manifest) if "\t" in l} if a.train_manifest else set()
    rows, keep = [], {}
    for b in OFFICIAL:
        S, e, best, singles = batch(os.path.join(a.out, b))
        mean_member = st.mean(S[n]["test"] for n in singles)
        rows.append(dict(b=b, ens=e, best=best, lp_ens=S[e]["lp"], lp_best=S[best]["lp"],
                         d_val=S[e]["val"] - S[best]["val"], d_test=S[e]["test"] - S[best]["test"],
                         ens_test=S[e]["test"], best_test=S[best]["test"], d_mean_member=S[e]["test"] - mean_member,
                         singles={n: (round(S[n]["val"], 3), round(S[n]["test"], 3), S[n]["lp"]) for n in singles}))
        keep[b] = (S[e]["hyps"], S[best]["hyps"], S[e]["refs"])
        print(f"{b:7s} ens {S[e]['val']:.3f}/{S[e]['test']:.3f}@LP{S[e]['lp']} | best-on-val {best} "
              f"{S[best]['val']:.3f}/{S[best]['test']:.3f}@LP{S[best]['lp']} | Δval {rows[-1]['d_val']:+.3f} "
              f"Δtest {rows[-1]['d_test']:+.3f} | singles {rows[-1]['singles']}")
    refs0 = keep[OFFICIAL[0]][2]
    assert all(keep[b][2] == refs0 for b in OFFICIAL), "test references differ across batches"
    dv = [r["d_val"] for r in rows]; dt = [r["d_test"] for r in rows]
    t, p2 = stats.ttest_1samp(dt, 0.0); p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    m, lo, hi = interval(dt)
    print(f"\nΔ val  per batch {' '.join(f'{x:+.3f}' for x in dv)} | mean {st.mean(dv):+.3f}")
    print(f"Δ test per batch {' '.join(f'{x:+.3f}' for x in dt)} | mean {m:+.3f} [95% t df6 {lo:+.3f}, {hi:+.3f}] | "
          f"positive {sum(x > 0 for x in dt)}/7 | one-sided p {p1:.4f}")
    # (d) pooled paired sentence bootstrap: one resample of the 800 sentences per draw, applied to every batch
    rng = np.random.default_rng(a.seed); N = len(refs0); boot = np.empty(a.n)
    for k in range(a.n):
        ii = rng.integers(0, N, size=N); R = [refs0[i] for i in ii]
        boot[k] = np.mean([corpus_bleu4([keep[b][0][i] for i in ii], R, "13a")
                           - corpus_bleu4([keep[b][1][i] for i in ii], R, "13a") for b in OFFICIAL])
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"pooled sentence bootstrap (test sampling only) mean Δ {np.mean(dt):+.3f} [{blo:+.3f}, {bhi:+.3f}]")
    fut = st.mean(dv) <= 0
    conds = dict(a=p1 < 0.05, b=st.mean(dv) > 0, c=sum(x > 0 for x in dt) >= 6, d=blo > 0)
    print("FUTILITY (val mean <= 0):", fut, "| conditions", conds)
    print("READING:", "DOES NOT REPLICATE (futility)" if fut else
          ("SAME-RECIPE SEED ENSEMBLING REPLICATES" if all(conds.values()) else "BOUNDED ESTIMATE (rule not met)"))
    # secondary, labelled
    dm = [r["d_mean_member"] for r in rows]
    print(f"\n[secondary, descriptive] ens - mean member test: {' '.join(f'{x:+.3f}' for x in dm)} | mean {st.mean(dm):+.3f}")
    print(f"[secondary] across-batch sd of test: ens {st.stdev([r['ens_test'] for r in rows]):.3f} vs "
          f"best-on-val single {st.stdev([r['best_test'] for r in rows]):.3f}")
    if train:
        uns = [norm(r) not in train for r in refs0]
        du = [corpus_bleu4([h for h, u in zip(keep[b][0], uns) if u], [r for r, u in zip(refs0, uns) if u], "13a")
              - corpus_bleu4([h for h, u in zip(keep[b][1], uns) if u], [r for r, u in zip(refs0, uns) if u], "13a")
              for b in OFFICIAL]
        mu, ulo, uhi = interval(du)
        print(f"[secondary] unseen-{sum(uns)} Δ: {' '.join(f'{x:+.3f}' for x in du)} | mean {mu:+.3f} [{ulo:+.3f}, {uhi:+.3f}]")
    # VD, descriptive
    vd = os.path.join(a.out, "vdB")
    if os.path.isdir(vd) and os.path.exists(os.path.join(vd, "DONE")):
        S, e, best, singles = batch(vd)
        print(f"\n[VD descriptive] {e} {S[e]['val']:.3f}/{S[e]['test']:.3f}@LP{S[e]['lp']} vs best-on-vd_val {best} "
              f"{S[best]['val']:.3f}/{S[best]['test']:.3f} -> Δtest {S[e]['test'] - S[best]['test']:+.3f}")
        eh, bh, R0 = S[e]["hyps"], S[best]["hyps"], S[e]["refs"]; N = len(R0); rng = np.random.default_rng(a.seed); bb = []
        for _ in range(a.n):
            ii = rng.integers(0, N, size=N); R = [R0[i] for i in ii]
            bb.append(corpus_bleu4([eh[i] for i in ii], R, "13a") - corpus_bleu4([bh[i] for i in ii], R, "13a"))
        singles_txt = ", ".join(f"{n}: {S[n]['val']:.3f}/{S[n]['test']:.3f}" for n in singles)
        print(f"[VD descriptive] paired sentence bootstrap 95% [{np.percentile(bb, 2.5):+.3f}, "
              f"{np.percentile(bb, 97.5):+.3f}]; singles val/test {singles_txt}")
    json.dump(rows, open(os.path.join(a.out, "seedens_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
