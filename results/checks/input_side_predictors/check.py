"""Input-side predictors of translation quality (dissertation Results, 'Training-free measurements').

Recomputes, from per-clip data (13 pose descriptors of the 800 official-test clips; references and beam-5 hypotheses of
three single models), the Pearson r of every descriptor against four per-clip outcomes, the family-wise shuffled-outcome
null band, the video-grouped cross-validated AUC, and the video-identity permutation test. The procedure follows
tools/translation_predictors.py (same permutation count, seed and random-number order); its saved output
translation_predictors.json is the recorded run used as the gate.
"""
import json, math, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize
from tools.stitch_lexicon import content_words
import sacrebleu

R = [json.loads(l) for l in open(HERE / "per_clip.jsonl")]
MODELS = ["dwpose_ctc", "vlp", "rgbfusion"]
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():  # references must be the official test references (when `make data` has written the manifest)
    m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
    assert all(m[r["id"]] == r["ref"].strip() for r in R), "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv")
fnames = list(R[0]["feat"])
X = np.array([[r["feat"][k] for k in fnames] for r in R], float)
videos = np.array([r["video"] for r in R])
refs = [r["ref"] for r in R]


def outcomes(hyp, ref):
    h, r = detokenize(hyp), detokenize(ref)
    cw_r, cw_h = content_words(ref), set(content_words(hyp))
    return {"content_recall": float(np.mean([w in cw_h for w in cw_r])) if cw_r else np.nan,
            "any_content": float(any(w in cw_h for w in cw_r)) if cw_r else np.nan,
            "chrf": float(sacrebleu.sentence_chrf(h, [r]).score),
            "exact": float(h.strip() == r.strip())}


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


onames = ["content_recall", "any_content", "chrf", "exact"]
oc = {m: [outcomes(r["hyp_" + m], r["ref"]) for r in R] for m in MODELS}
Y = {o: np.nanmean(np.array([[x[o] for x in oc[m]] for m in MODELS], float), axis=0) for o in onames}

# permutation null: same random-number order as the tool (one pass of 500 shuffles per outcome that the tool discards,
# then the 500 shuffles per outcome that give the band)
PERM = 500
rng = np.random.default_rng(0)
for o in onames:
    for _ in range(PERM):
        rng.shuffle(Y[o].copy())
null = {}
for o in onames:
    mx = []
    for _ in range(PERM):
        yp = Y[o].copy(); rng.shuffle(yp)
        mx.append(max(abs(corr(X[:, j], yp)) for j in range(X.shape[1])))
    null[o] = float(np.percentile(mx, 95))
C = {fn: {o: corr(X[:, j], Y[o]) for o in onames} for j, fn in enumerate(fnames)}

# video-grouped 5-fold logistic model: 'gets >= 1 content word'
yb = (Y["any_content"] > 0.5).astype(int)
Xz = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1)
uv = np.unique(videos); rng.shuffle(uv)
folds = np.array_split(uv, 5)
aucs, aucs_shuf = [], []
yshuf = yb.copy(); rng.shuffle(yshuf)
for target, store in ((yb, aucs), (yshuf, aucs_shuf)):
    for f in folds:
        te = np.isin(videos, f); tr = ~te
        if te.sum() < 15 or len(np.unique(target[tr])) < 2 or len(np.unique(target[te])) < 2:
            continue
        w = np.zeros(Xz.shape[1] + 1)
        Xt = np.hstack([Xz[tr], np.ones((tr.sum(), 1))])
        for _ in range(600):
            pr = 1 / (1 + np.exp(-Xt @ w))
            w -= 0.5 * (Xt.T @ (pr - target[tr]) / tr.sum() + 0.01 * w)
        s = np.hstack([Xz[te], np.ones((te.sum(), 1))]) @ w
        pos, neg = s[target[te] == 1], s[target[te] == 0]
        store.append(float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
auc, auc_s = float(np.mean(aucs)), float(np.mean(aucs_shuf))

# GATE: the recomputed correlations, null band and AUCs must equal the recorded run (saved job output)
rec = json.load(open(HERE / "translation_predictors.json"))
bad = []
for fn in fnames:
    for o in onames:
        a, b = C[fn][o], rec["corr"][fn][o]
        if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > 1e-6:
            bad.append((fn, o, a, b))
bad += [("null", o, null[o], rec["null_band"][o]) for o in onames if abs(null[o] - rec["null_band"][o]) > 1e-6]
bad += [("auc", "", x, y) for x, y in ((auc, rec["auc"]), (auc_s, rec["auc_shuffled"])) if abs(x - y) > 1e-6]
if len(R) != rec["n"] or fnames != rec["features"] or bad:
    sys.exit(f"GATE FAILED: {bad[:3]}")
print(f"GATE: {len(R)} clips, {len(fnames)} descriptors x {len(onames)} outcomes, null bands and AUCs reproduce the recorded run (tolerance 1e-6)")

feats = [f for f in fnames if not all(math.isnan(C[f][o]) for o in onames)]
print(f"\nclips {len(R)}; descriptors {len(fnames)}; constant on all clips: " + ", ".join(f for f in fnames if f not in feats))
print(f"Non-degenerate clip-level pose descriptors: {len(feats)}  [c10-047]")
exceed = [(f, o) for f in feats for o in onames if abs(C[f][o]) > null[o]]
print(f"descriptor x outcome pairs exceeding the shuffled-outcome null band: {len(exceed)} of {len(feats)*len(onames)}  [c10-047]")
print("null band (95th percentile of max|r| over shuffles): " + ", ".join(f"{o} {null[o]:.3f}" for o in onames))
r, f, o = max((abs(C[f][o]), f, o) for f in feats for o in onames)
print(f"largest |r| = {r:.3f} ({f}, {o})  [c10-048]")
print(f"largest variance explained r^2 = {r*r*100:.1f}% (about 1%)  [c10-048]")
print(f"joint logistic model, video-grouped 5-fold AUC {auc:.3f} vs shuffled-outcome control {auc_s:.3f}  [c10-047]")
cw = np.array([len(content_words(x)) for x in refs], float)
print(f"positive control (reference side, not available at inference): n_content_words vs any_content r = "
      f"{corr(cw, Y['any_content']):.3f}, exceeds band {null['any_content']:.3f}: {abs(corr(cw, Y['any_content'])) > null['any_content']}")

# video identity: spread of the per-video success rate against random assignment of clips to videos
y = Y["any_content"]; ok = np.isfinite(y)
vs = sorted(v for v in set(videos[ok]) if (videos[ok] == v).sum() >= 5)   # videos with at least 5 scored clips
mk = np.isin(videos, vs) & ok
yy, vv = y[mk], videos[mk]
idx = [np.where(vv == v)[0] for v in vs]
sd = lambda z: float(np.std([z[i].mean() for i in idx]))
obs = sd(yy)
rg = np.random.default_rng(0)
nl = np.array([sd(rg.permutation(yy)) for _ in range(1000)])
p = float((nl >= obs).mean())
print(f"\nvideo identity: {len(vs)} videos with >= 5 scored clips hold {mk.sum()} of {ok.sum()} scored clips; "
      f"per-video success rates run {min(yy[i].mean() for i in idx):.3f} to {max(yy[i].mean() for i in idx):.3f}")
print(f"between-video sd of success rate {obs:.4f} vs shuffled null mean {nl.mean():.4f} "
      f"(95th percentile {np.percentile(nl, 95):.4f}), permutation p = {p:.2f} (1000 shuffles, seed 0)  [c10-047]")
print(f"observed spread is within the shuffled null (p > 0.05): {p > 0.05}  [c10-047]")
