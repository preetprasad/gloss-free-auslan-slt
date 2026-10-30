"""Non-signer negative control: single model, six-member ensemble (12e) and the grounding score (wave 18).
Recomputes every ratio, interval and area under the curve from the saved per-clip scores.
Run from src/:  python3 ../results/checks/nonsigner_confidence_gate/check.py"""
import csv, hashlib, json, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
B, SEED = 2000, 0
J = lambda n: json.load(open(HERE / n))


# ---------------------------------------------------------------- statistics (nonsigner_gate conventions: video-cluster bootstrap, 2,000 draws, seed 0)
class Boot:
    def __init__(self, videos):
        self.uniq = sorted(set(videos)); self.pos = {v: i for i, v in enumerate(self.uniq)}
        rng, n = np.random.default_rng(SEED), len(self.uniq)
        self.idx = np.array([rng.choice(np.arange(n), n) for _ in range(B)])           # same stream as rng.choice(sorted videos, n)
        self.W = np.stack([np.bincount(r, minlength=n) for r in self.idx]).astype(float)

    def table(self, videos, flags):
        n, k = np.zeros(len(self.uniq)), np.zeros(len(self.uniq))
        for v, f in zip(videos, flags):
            n[self.pos[v]] += 1; k[self.pos[v]] += bool(f)
        return n, k


def pct(draws):
    d = np.asarray(draws, float); d = d[np.isfinite(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ratio_cell(boot, ce_sig, v_sig, ce_neg, v_neg, tau):
    sn, sk = boot.table(v_sig, np.asarray(ce_sig) <= tau); nn, nk = boot.table(v_neg, np.asarray(ce_neg) <= tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss, sv, ns, nv = boot.W @ sn, boot.W @ sk, boot.W @ nn, boot.W @ nk
        dr = (nv / ns) / (sv / ss); dr[(ss == 0) | (ns == 0) | (sv == 0)] = np.nan
    C, FV = sk.sum() / sn.sum(), nk.sum() / nn.sum()
    lo, hi = pct(dr)
    return dict(C=C, FV=FV, R=FV / C, lo=lo, hi=hi, n_sig=len(ce_sig), n_neg=len(ce_neg)), dr


def dev_cell(boot, ce_neg, v_neg, tau, q):
    nn, nk = boot.table(v_neg, np.asarray(ce_neg) <= tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = (boot.W @ nk) / (boot.W @ nn) / q
    lo, hi = pct(dr)
    return dict(FV=nk.sum() / nn.sum(), R=nk.sum() / nn.sum() / q, lo=lo, hi=hi, n_neg=len(ce_neg))


def auroc(pos, neg):
    r = rankdata(np.r_[pos, neg]); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_boot(boot, sig_by_v, neg_by_v, other=None):
    """AUROC (signers above non-signers) with a video-cluster interval; with `other` = a second score table, the paired difference."""
    def stat(vs, tabs):
        out = []
        for ts, tn in tabs:
            s = np.concatenate([ts[v] for v in vs]); n = np.concatenate([tn[v] for v in vs])
            out.append(auroc(s, n) if len(n) and len(s) else np.nan)
        return out
    tabs = [(sig_by_v, neg_by_v)] + ([other] if other else [])
    f = (lambda vs: stat(vs, tabs)[0] - stat(vs, tabs)[1]) if other else (lambda vs: stat(vs, tabs)[0])
    full = f(boot.uniq)
    return float(full), *pct([f([boot.uniq[i] for i in r]) for r in boot.idx])


# ---------------------------------------------------------------- world: videos, populations
vid = lambda c: re.match(r"(video_\d+)_", c).group(1)
parent = lambda c: re.sub(r"_\d+_nonsigner$", "", c)
SINGLE, ENS = J("decodes_single.json"), J("decodes_ens.json")
GS, GE = J("ground_single.json"), J("ground_ens.json")
test_ids, val_ids = list(SINGLE["sig_test"]), list(SINGLE["sig_val"])
assert test_ids == list(ENS["sig_test"]) and val_ids == list(ENS["sig_val"]) and len(test_ids) == len(val_ids) == 800
man = ROOT / "comm_run" / "manifests"                                                    # written by `make data`
for nm, ids in (("test", test_ids), ("val", val_ids)):
    if (man / f"{nm}.tsv").exists():
        stems = {Path(l.split("\t")[0]).stem for l in open(man / f"{nm}.tsv", encoding="utf-8") if "\t" in l}
        assert stems == set(ids), f"{nm}: saved clip ids differ from comm_run/manifests/{nm}.tsv"
neg_ids = sorted(SINGLE["neg"]); assert set(neg_ids) == set(ENS["neg"]) and len(neg_ids) == len(set(neg_ids)) == 1678      # the two records list the crops in different orders
split = {c: "test" if parent(c) in set(test_ids) else "dev" if parent(c) in set(val_ids) else None for c in neg_ids}
assert None not in split.values()
n_test_all, n_dev_all = sum(s == "test" for s in split.values()), sum(s == "dev" for s in split.values())
flagged = set(J("b3.json")["flagged_ids"])                                               # read from the saved job summary (needs pose arrays to recompute)
prim = {s: [c for c in neg_ids if split[c] == s and c not in flagged] for s in ("test", "dev")}
v_sig = {c: vid(c) for c in test_ids + val_ids}; v_neg = {c: vid(parent(c)) for c in neg_ids}
BT = Boot([v_sig[c] for c in test_ids] + [v_neg[c] for c in prim["test"]])               # test universe: signers + primary crops
BD = Boot([v_neg[c] for c in prim["dev"]])                                              # dev universe: crops only


def gate_cells(sc_val, sc_test, sc_neg, q):
    """sc_*: {clip: confidence} (higher = voiced). ce = -score, voiced iff ce <= tau, tau = q-quantile of the 800 validation signers."""
    ce_val = [-sc_val[c] for c in val_ids]; tau = float(np.quantile(ce_val, q))
    t, dr = ratio_cell(BT, [-sc_test[c] for c in test_ids], [v_sig[c] for c in test_ids], [-sc_neg[c] for c in prim["test"]], [v_neg[c] for c in prim["test"]], tau)
    d = dev_cell(BD, [-sc_neg[c] for c in prim["dev"]], [v_neg[c] for c in prim["dev"]], tau, q)
    d["C_val"] = float(np.mean([x <= tau for x in ce_val]))
    return tau, t, dr, d


def auc_tabs(sc_test, sc_neg):
    s = {v: [] for v in BT.uniq}; n = {v: [] for v in BT.uniq}
    for c in test_ids: s[v_sig[c]].append(sc_test[c])
    for c in prim["test"]: n[v_neg[c]].append(sc_neg[c])
    return {v: np.array(x) for v, x in s.items()}, {v: np.array(x) for v, x in n.items()}


neg_conf = lambda D: {c: -D["neg"][c][0] for c in neg_ids}
CE = {k: {"sig_val": {c: -v for c, v in D["sig_val"].items()}, "sig_test": {c: -v for c, v in D["sig_test"].items()},
          "neg": neg_conf(D)} for k, D in (("single", SINGLE), ("ens", ENS))}
for k, D in (("ens", ENS),):
    CE[k]["trk_val"] = {c: -v for c, v in D["trk_val"].items()}; CE[k]["trk_test"] = {c: -v for c, v in D["trk_test"].items()}
GR = {k: {"sig_val": {c: v for c, v in G["sig_val"].items()}, "sig_test": {c: v[0] for c, v in G["sig_test"].items()},
          "neg": {c: v[0] for c, v in G["neg"].items()}} for k, G in (("single", GS), ("ens", GE))}
f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{100 * x:.1f}"; f3 = lambda x: f"{x:.3f}"


# ---------------------------------------------------------------- gate: the recorded confidence-gate values (tools/nonsigner_gate.py, tools/w12e_nonsigner_s6.py) must be reproduced exactly
REC = {"single": dict(R=1.0076813207823025, lo=0.9166523452518263, hi=1.099691549265032, dev=1.0211180124223602, auc=(0.5094930806257522, 0.4810407729443646, 0.5387469767703404)),
       "ens": dict(R=0.8443883838603511, lo=0.7514442871020933, hi=0.9375265522875815, dev=0.8496894409937888, auc=(0.5646495186522262, 0.5347288231928803, 0.5966205060941199))}
RES = {}
for k in ("single", "ens"):
    tau, t, dr, d = gate_cells(CE[k]["sig_val"], CE[k]["sig_test"], CE[k]["neg"], 0.5)
    a = auroc_boot(BT, *auc_tabs(CE[k]["sig_test"], CE[k]["neg"]))
    r = REC[k]
    ok = all(abs(x - y) < 1e-9 for x, y in ((t["R"], r["R"]), (t["lo"], r["lo"]), (t["hi"], r["hi"]))) and abs(d["R"] - r["dev"]) < 1e-9 \
        and all(abs(x - y) < 1e-9 for x, y in zip(a, r["auc"]))
    if not ok:
        print(f"GATE FAILED ({k}): {t['R']} {t['lo']} {t['hi']} dev {d['R']} auc {a} != recorded {r}"); sys.exit(1)
print("GATE PASSED: confidence-gate ratio, interval and AUROC of both systems equal the values recorded by the original jobs")
print(f"Crops: {n_test_all} test + {n_dev_all} validation non-signer crops registered; {len(prim['test'])} test and {len(prim['dev'])} validation crops after removing the "
      f"{len(flagged)} flagged as the signer's own person (flag list read from the saved job summary)")
print(f"Test videos: signers {len(test_ids)} clips, crops {len(prim['test'])}; threshold fitted on the {len(val_ids)} validation signers\n")


# ---------------------------------------------------------------- (1) single model, 40-epoch pre-trained, production pose
S = SINGLE
tau, t, dr, d = gate_cells(CE["single"]["sig_val"], CE["single"]["sig_test"], CE["single"]["neg"], 0.5)
a = auroc_boot(BT, *auc_tabs(CE["single"]["sig_test"], CE["single"]["neg"]))
print(f"== Single model (40-epoch pre-trained), threshold fitted on validation for 50% coverage (realised {f1(d['C_val'])}% on validation)")
print(f"  voiced: test signers {f1(t['C'])}% ; non-signers {f1(t['FV'])}% ; ratio {f2(t['R'])} (95% CI {f2(t['lo'])} to {f2(t['hi'])}, clustered by video)")
print(f"  AUROC {f2(a[0])} [{f2(a[1])}, {f2(a[2])}] (0.50 = chance)")
fz = float(np.mean([S["frozen"]["sig_test"][c] <= tau for c in test_ids]))
ws = {r["id"]: float(r["wrist_speed"]) for r in csv.DictReader(open(HERE / "wrist_speed.csv"))}
thr = np.nanpercentile([ws[c] for c in val_ids], 10)
quiet = [c for c in prim["test"] if ws[c] < thr]
fq = float(np.mean([-CE["single"]["neg"][c] <= tau for c in quiet]))
print(f"  non-signers who hardly move (wrist speed below the 10th percentile of validation signers; {len(quiet)} crops) voiced: {f1(fq)}%")
print(f"  a signer's pose frozen at one frame voiced: {f1(fz)}% of test signers")

# ---------------------------------------------------------------- (2) six-member ensemble (12e)
E = ENS
tau, t, dr, d = gate_cells(CE["ens"]["sig_val"], CE["ens"]["sig_test"], CE["ens"]["neg"], 0.5)
a = auroc_boot(BT, *auc_tabs(CE["ens"]["sig_test"], CE["ens"]["neg"]))
print("\n== Final six-member ensemble, production pose, threshold for 50% coverage")
print(f"  voiced: test signers {f1(t['C'])}% ; non-signers {f1(t['FV'])}% ; ratio {f2(t['R'])} (95% CI {f2(t['lo'])} to {f2(t['hi'])})")
print(f"  more than four non-signing people in ten still voiced: {t['FV'] > 0.4}")
print(f"  AUROC {f2(a[0])} [{f2(a[1])}, {f2(a[2])}]")
print(f"  validation videos ({d['n_neg']} crops): ratio {f2(d['R'])} ({f2(d['lo'])} to {f2(d['hi'])})")
tau75, t75, _, _ = gate_cells(CE["ens"]["sig_val"], CE["ens"]["sig_test"], CE["ens"]["neg"], 0.75)
print(f"  threshold for 75% coverage, test: ratio {f2(t75['R'])} ({f2(t75['lo'])} to {f2(t75['hi'])})")
_, tt, _, _ = gate_cells(CE["ens"]["trk_val"], CE["ens"]["trk_test"], CE["ens"]["neg"], 0.5)
print(f"  person-tracked pose, test: ratio {f2(tt['R'])} ({f2(tt['lo'])} to {f2(tt['hi'])})")
tr = {x.strip() for x in open(ROOT / "results" / "train_target_hashes.txt")}
voiced = [c for c in prim["test"] if -CE["ens"]["neg"][c] <= tau]
vb = float(np.mean([hashlib.sha1(E["neg"][c][1].encode("utf-8")).hexdigest() in tr for c in voiced]))
print(f"  outputs of the {len(voiced)} voiced non-signer crops that are verbatim training sentences: {100 * vb:.0f}%")

# ---------------------------------------------------------------- (3) grounding score (wave 18), primary arm, threshold for 50% coverage
print("\n== Grounding score in place of confidence (wave 18), production pose, test, threshold for 50% coverage")
for k, lab in (("ens", "final ensemble"), ("single", "40-epoch pre-trained model")):
    _, g, _, _ = gate_cells(GR[k]["sig_val"], GR[k]["sig_test"], GR[k]["neg"], 0.5)
    _, c, _, _ = gate_cells(CE[k]["sig_val"], CE[k]["sig_test"], CE[k]["neg"], 0.5)
    line = f"  {lab}: non-signers voiced / signers voiced, grounding {f2(g['R'])} ({f2(g['lo'])} to {f2(g['hi'])}) against {f2(c['R'])} for confidence; {g['n_neg']} test crops"
    ag = auroc_boot(BT, *auc_tabs(GR[k]["sig_test"], GR[k]["neg"])); ac = auroc_boot(BT, *auc_tabs(CE[k]["sig_test"], CE[k]["neg"]))
    ad = auroc_boot(BT, *auc_tabs(GR[k]["sig_test"], GR[k]["neg"]), other=auc_tabs(CE[k]["sig_test"], CE[k]["neg"]))
    print(line)
    print(f"    AUROC grounding {f2(ag[0])} against {f2(ac[0])} for confidence (difference {ad[0]:+.2f}, {ad[1]:+.2f} to {ad[2]:+.2f})")
    x_s = np.array([GS_[1] for GS_ in (GS if k == "single" else GE)["sig_test"].values()]); x_n = np.array([v[1] for c_, v in (GS if k == "single" else GE)["neg"].items() if c_ in set(prim["test"])])
    print(f"    true video beats a mismatched one (same output words) on {f1(np.mean(x_s > 0))}% of test signer clips and {f1(np.mean(x_n > 0))}% of non-signer crops")
    REC_G = {"ens": (0.9501930079547407, 0.8601534122037303, 1.040999805593259), "single": (1.006808281517089, 0.9112923533128309, 1.103097531991896)}[k]
    assert all(abs(x - y) < 1e-9 for x, y in zip((g["R"], g["lo"], g["hi"]), REC_G)), "grounding-score ratio differs from the wave-18 record"
print("\nGATE PASSED: grounding-score ratios and intervals equal the wave-18 record")
