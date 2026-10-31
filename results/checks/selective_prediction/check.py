"""Selective prediction: single model (DWPose+CTC, beam 5) and the S6 ensemble with validation-fixed thresholds.

Single model: per-clip beam-5 hypotheses, references and the model's self-CE / self-reliance on its own output
(saved by tools/self_reliance.py), 800 official test and 800 validation clips. Retained-set BLEU-4 by coverage
follows tools/selective_prediction.py (stable argsort, top round(c*n), sacreBLEU 13a on detokenised text).
Ensemble: saved n-best of the six-member S6 ensemble; tools/selective_ci.py's bootstrap (2000 resamples, seed 0).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize
from tools.ensemble_selective import load, bleu

dig = set(open(ROOT / "results/train_target_hashes.txt").read().split())
sha = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()


def single(split):
    recs = [json.loads(l) for l in open(HERE / f"single_{split}.jsonl")]
    man = ROOT / "comm_run" / "manifests" / f"{'test' if split == 'test' else 'val'}.tsv"
    if man.exists():
        m = [l.rstrip("\n").split("\t", 1)[1] for l in open(man) if "\t" in l]
        assert [r["ref"] for r in recs] == m, f"references differ from {man.name}"
        print(f"references match comm_run/manifests/{man.name}: {len(m)}/{len(m)}")
    ce = np.array([r["self_ce"] for r in recs]); rel = np.array([r["self_rel"] for r in recs])
    hy = [detokenize(r["hyp"]) for r in recs]; rf = [detokenize(r["ref"]) for r in recs]
    seen = np.array([sha(r["ref"].strip()) in dig for r in recs])
    verb = np.array([sha(r["hyp"].strip()) in dig for r in recs])
    z = (-ce - (-ce).mean()) / ce.std() + (rel - rel.mean()) / rel.std()
    return dict(n=len(recs), ce=ce, z=z, hy=hy, hr=[r["hyp"] for r in recs], rf=rf, seen=seen, verb=verb,
                length=np.array([len(h.split()) for h in hy], float))


def cur(S, rank, idx, c):
    idx = np.asarray(idx); o = idx[np.argsort(-rank[idx], kind="stable")][:int(round(c * len(idx)))]
    return corpus_bleu4([S["hy"][i] for i in o], [S["rf"][i] for i in o], tokenize="13a"), o


T, V = single("test"), single("val")
# GATE: the full-set beam-5 test score is the recorded 9.52 of the 40-epoch DWPose+CTC model (and validation 9.64)
g_t = cur(T, T["z"], range(T["n"]), 1.0)[0]; g_v = cur(V, V["z"], range(V["n"]), 1.0)[0]
print(f"GATE full-set beam-5 BLEU-4: test {g_t:.2f} (recorded 9.52), validation {g_v:.2f} (recorded 9.64)")
if round(g_t, 2) != 9.52 or round(g_v, 2) != 9.64:
    sys.exit("GATE FAILED")
COVS = (1.0, 0.9, 0.75, 0.5, 0.25)
allT, allV = list(range(T["n"])), list(range(V["n"]))
unT = [i for i in allT if not T["seen"][i]]; unV = [i for i in allV if not V["seen"][i]]
print(f"\nSetting: single model, beam 5 hypotheses (field hyp_beam of the saved decodes), n test {T['n']}, "
      f"n validation {V['n']}, unseen-target test clips {len(unT)} (validation {len(unV)})  [c16-044]")
print(f"Coverage = clips answered / clips, from 0% to 100%; here 100% = all {T['n']} answered  [c06-010]")
print(f"At 50% coverage the system answers {int(round(.5 * T['n']))} of {T['n']} test clips (its more confident half, coverage 0.5)  [c01-029 c02-029 c18-036]")
print("Coverages evaluated: 90%, 75%, 50% and 25% (plus 100%)  [c07-073]")

rowT = {c: cur(T, T["z"], allT, c)[0] for c in COVS}
rowV = {c: cur(V, V["z"], allV, c)[0] for c in COVS}
ceT = {c: cur(T, -T["ce"], allT, c)[0] for c in COVS}
ceV = {c: cur(V, -V["ce"], allV, c)[0] for c in COVS}
print("\nRetained-set BLEU-4 by coverage (ranking: self-CE + self-reliance, z-sum)")
print(f"{'':22s}" + "".join(f"{int(c*100):>8d}%" for c in COVS) + "  [c16-046]")
print(f"{'Test':22s}" + "".join(f"{rowT[c]:9.2f}" for c in COVS) + "  [c16-046]")
print(f"{'Validation':22s}" + "".join(f"{rowV[c]:9.2f}" for c in COVS) + "  [c16-047]")
print(f"{'Test, self-CE alone':22s}" + "".join(f"{ceT[c]:9.2f}" for c in COVS))
print(f"{'Val, self-CE alone':22s}" + "".join(f"{ceV[c]:9.2f}" for c in COVS))
print(f"\nTest, all {T['n']} clips, 100% -> 50%: {rowT[1.0]:.2f} -> {rowT[.5]:.2f}  [c01-030 c02-030 c13-001 c13-003 c13-008]")
print(f"Validation 100% -> 50%: {rowV[1.0]:.2f} -> {rowV[.5]:.2f}  [c13-005]")
print(f"Self-CE term alone at 50% coverage: test {ceT[.5]:.2f}, validation {ceV[.5]:.2f}; two-term score {rowT[.5]:.2f} and {rowV[.5]:.2f}  [c13-002 c18-035]")
print("Beam width 5 (beam-5 decodes; the full-set score matches the recorded beam-5 value above)  [c13-004]")

# unseen targets: clips retained by the all-clip ranking whose reference is unseen
def unseen_ret(S, c):
    o = cur(S, S["z"], range(S["n"]), c)[1]; u = [i for i in o if not S["seen"][i]]
    return corpus_bleu4([S["hy"][i] for i in u], [S["rf"][i] for i in u], tokenize="13a"), len(u)
u100 = corpus_bleu4([T["hy"][i] for i in unT], [T["rf"][i] for i in unT], tokenize="13a")
u50, nu = unseen_ret(T, .5)
uv100 = corpus_bleu4([V["hy"][i] for i in unV], [V["rf"][i] for i in unV], tokenize="13a"); uv50, _ = unseen_ret(V, .5)
print(f"\nUnseen-target test clips ({len(unT)}): all {u100:.2f}; the more confident half of the 800 clips, unseen ones only ({nu}): {u50:.2f}  [c13-007 c16-048 c02-030]")
print(f"Unseen-target validation: all {uv100:.2f}, confident half {uv50:.2f}")

# longest-output control
L = {c: cur(T, T["length"], allT, c)[0] for c in (.5, .25)}
print(f"\nControl, keep the longest outputs: 50% coverage {L[.5]:.2f}  [c13-006]; 25% coverage {L[.25]:.2f}  [c18-037]")

# composition of the answered set at 25% coverage
o25 = cur(T, T["z"], allT, .25)[1]
print(f"Baseline model, 25% coverage: {T['verb'][o25].mean():.1%} of the answered outputs are verbatim training sentences "
      f"(all 800: {T['verb'].mean():.1%})  [c13-018]")
print(f"Signer repeats {1 - 0.5:.0%} of utterances at 50% coverage, i.e. one in two  [c14-092]")

# ---------- ensemble ----------
print("\nS6 ensemble, beam-10 n-best, length penalty 0.3; thresholds from validation, applied unchanged to test")
vr, _, vc = load((ROOT / "results/final_ensemble_validation_decodes/nbest.jsonl").parent, 0.3)
r, h, c = load((ROOT / "results/final_ensemble_test_decodes/nbest.jsonl").parent, 0.3)
tseen = np.array([sha(x.strip()) in dig for x in r]); vseen = np.array([sha(x.strip()) in dig for x in vr])
N = len(r); rng = np.random.default_rng(0)
full = bleu(h, r)
print(f"GATE ensemble test BLEU-4 {full:.3f} (recorded 11.802)")
if round(full, 3) != 11.802:
    sys.exit("GATE FAILED")
RECORD = {0.9: (12.90, 0.894), 0.75: (14.61, 0.756), 0.5: (19.07, 0.519)}
out = {}
for subset in ("all", "unseen"):
    keep = np.ones(N, bool) if subset == "all" else ~tseen
    vcs = vc if subset == "all" else vc[~vseen]
    idx0 = np.flatnonzero(keep)
    for cov in (0.9, 0.75, 0.5):
        tau = np.quantile(vcs, 1 - cov)
        sel = idx0[c[idx0] >= tau]
        b = bleu([h[i] for i in sel], [r[i] for i in sel]); got = len(sel) / len(idx0)
        if subset == "all":
            rb, rc = RECORD[cov]
            if abs(b - rb) >= 0.006 or abs(got - rc) >= 0.0006:
                sys.exit(f"GATE FAILED {cov}")
        bs, ds = [], []
        for _ in range(2000):
            ii = rng.choice(idx0, size=len(idx0), replace=True)
            s_ = ii[c[ii] >= tau]
            rnd = rng.choice(ii, size=len(s_), replace=False)
            bsel = bleu([h[i] for i in s_], [r[i] for i in s_])
            bs.append(bsel); ds.append(bsel - bleu([h[i] for i in rnd], [r[i] for i in rnd]))
        dlo, dhi = np.percentile(ds, [2.5, 97.5])
        out[(subset, cov)] = (b, got, np.mean(ds), dlo, dhi, len(sel))
for (subset, cov), (b, got, m, lo, hi, ns) in out.items():
    tag = {("all", 0.5): "  [c13-012 c18-038]", ("unseen", 0.5): "  [c18-039]"}.get((subset, cov), "")
    print(f"{subset:6s} cov {cov:.2f}: test {b:6.2f} @ {got:5.1%} ({ns} clips); selected - random at matched coverage "
          f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{tag}")
b, got = out[("all", 0.5)][:2]
print(f"Ensemble answers {got:.1%} of the {N} official test clips with the validation threshold for 50% coverage  [c13-012]")
