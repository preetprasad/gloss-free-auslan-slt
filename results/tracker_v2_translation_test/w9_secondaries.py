"""Wave 9 (job 43506) secondaries fixed in results/tracker_v2/PREREG_TRACKER_V2_2026-09-25.md.

(1) Mandatory: per-seed paired deltas (V2 - C0, V1 - C0, V2 - V1) for unseen-570 BLEU and content precision/recall,
    read from readouts_{val,test}.txt (tools/arm_readouts.py output).
(2) Mandatory: heavy-switch-bin DiD with the bins of results/switch_video_89_193/tracking_dose_response.py
    (frac_chosen_nonzero from results/ndet_per_clip.csv): delta(bin) - delta(none).
(3) Pre-specified sensitivity (amendment of 26 Sep): V2 - C0 and V1 - C0 with the clips where V2 keeps < half its
    frames (kept = T - v2_masked; must reproduce the pre-registered 13 val / 9 test) removed from every arm.
Descriptive only: the decision is Stage 1 on beam-5 validation (V2 - C0 mean -0.023 -> stopped).
"""
import csv, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

SEEDS, T975 = (42, 101, 202), 4.303   # two-sided 95%, df = 2
PAIRS = (("v2", "c0"), ("v1", "c0"), ("v2", "v1"))
KEY = "tokenize=13a;detok=True"


def ci(d):
    m = st.mean(d); hw = T975 * st.stdev(d) / len(d) ** 0.5
    return f"{' '.join(f'{x:+.3f}' for x in d)} | {sum(x > 0 for x in d)}/3 | mean {m:+.3f} [{m - hw:+.3f}, {m + hw:+.3f}]"


def bleu(h, r):
    return corpus_bleu4([detokenize(x) for x in h], [detokenize(x) for x in r], "13a")


def clips(p):
    return [l.split("\t")[0].rsplit("/", 1)[1][:-4] for l in open(p) if "\t" in l]


dumps = {}
for split in ("val", "test"):
    order = clips(os.path.join(HERE, "manifests", f"c0_{split}.tsv"))
    for arm in ("v1", "v2"):
        assert clips(os.path.join(HERE, "manifests", f"{arm}_{split}.tsv")) == order, f"{arm} {split} order differs"
    for arm in ("c0", "v1", "v2"):
        for s in SEEDS:
            j = json.load(open(os.path.join(HERE, "runs", f"{arm}_s{s}", f"beam5_{split}_best.json")))
            assert j["manifest"].endswith(f"/{arm}_{split}.tsv"), j["manifest"]
            dumps[split, arm, s] = (j["beam_hyps"], j["refs"], j["beam_result"]["metrics"][KEY])
    dumps[split, "order"] = order

print("== (1) unseen BLEU and content P/R, paired per seed")
for split in ("val", "test"):
    rd = {}
    for l in open(os.path.join(HERE, f"readouts_{split}.txt")):
        name, js = l.split(" ", 1); rd[name] = json.loads(js)
    for a, b in PAIRS:
        for k in ("bleu4_unseen", "content_precision", "content_recall"):
            d = [rd[f"{a}_s{s}"][k] - rd[f"{b}_s{s}"][k] for s in SEEDS]
            print(f"  {split:4s} {a}-{b} {k:18s} {ci(d)}")

print("== (2) heavy-switch-bin DiD (bins from ndet_per_clip.csv, as tracking_dose_response.py)")
FRAC = {r["clip"]: float(r["frac_chosen_nonzero"]) for r in csv.DictReader(open(os.path.join(ROOT, "results/ndet_per_clip.csv")))}
BINS = {"none": lambda f: f == 0, "light": lambda f: 0 < f <= 0.1, "medium": lambda f: 0.1 < f <= 0.3, "heavy": lambda f: f > 0.3}
for split in ("val", "test"):
    frac = [FRAC[c] for c in dumps[split, "order"]]
    idx = {b: [i for i, x in enumerate(frac) if f(x)] for b, f in BINS.items()}
    print(f"  {split}: n per bin " + ", ".join(f"{b} {len(v)}" for b, v in idx.items()))
    for a, b in PAIRS:
        per = {}
        for bn, ii in idx.items():
            per[bn] = []
            for s in SEEDS:
                (ha, ra, _), (hb, rb, _) = dumps[split, a, s], dumps[split, b, s]
                assert ra == rb
                per[bn].append(bleu([ha[i] for i in ii], [ra[i] for i in ii]) - bleu([hb[i] for i in ii], [rb[i] for i in ii]))
        for bn in ("light", "medium", "heavy"):
            did = [x - y for x, y in zip(per[bn], per["none"])]
            print(f"    {a}-{b} {bn:6s} DiD {ci(did)}")

print("== (3) sensitivity: V2 keeps < half its frames -> clip removed from every arm")
stats = json.load(open(os.path.join(HERE, "manifests", "arm_stats.json")))
degen = {r["clip"] for r in stats if (r["T"] - r["v2_masked"]) / r["T"] < 0.5}
for split, expect in (("val", 13), ("test", 9)):
    order = dumps[split, "order"]
    drop = [i for i, c in enumerate(order) if c in degen]
    assert len(drop) == expect, f"{split}: {len(drop)} degenerate clips, pre-registered {expect}"
    keep = [i for i in range(len(order)) if i not in set(drop)]
    for (arm, s), _ in [((a, s), 0) for a in ("c0", "v1", "v2") for s in SEEDS]:
        h, r, rec = dumps[split, arm, s]
        assert abs(bleu(h, r) - rec) < 1e-6, f"gate: {split} {arm} s{s}"
    for a, b in (("v2", "c0"), ("v1", "c0")):
        d = []
        for s in SEEDS:
            (ha, ra, _), (hb, rb, _) = dumps[split, a, s], dumps[split, b, s]
            d.append(bleu([ha[i] for i in keep], [ra[i] for i in keep]) - bleu([hb[i] for i in keep], [rb[i] for i in keep]))
        print(f"  {split:4s} n={len(keep)} (-{len(drop)}) {a}-{b} {ci(d)}")
