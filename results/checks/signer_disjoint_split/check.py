#!/usr/bin/env python3
"""Signer-disjoint split (wave 12g): rebuilds the split from the dataset spreadsheet, re-scores the saved beam-5 decodes
of the three arms (three seeds) and recomputes every number the dissertation quotes for it.
Run from src/:  python3 ../results/checks/signer_disjoint_split/check.py"""
import collections, hashlib, json, os, random, statistics as st, sys, tempfile
import numpy as np
from sacrebleu.metrics import BLEU
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # repository root (results/checks/<gid>/check.py)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "tools")]
import w12g_readout as ro                                # noqa: E402  (the original read-out tool)
from slt.metrics import detokenize                  # noqa: E402
import w12g_signer_split as sp                           # noqa: E402

XLSX = ROOT / "data" / "auslan-daily" / "translation" / "Auslan-Daily_Communication.xlsx"
MAN = ROOT / "comm_run" / "manifests"
# The two spreadsheet clips that have no pose file (the pool of the experiment is 14,039 clips, not 14,041).
NO_POSE = {"video_12_62", "video_68_179"}
SEEDS = [42, 101, 202]
for p in (XLSX, MAN / "train.tsv"):
    if not p.exists():
        sys.exit(f"{p} is missing: run `make data` first")

info = sp.read_xlsx(str(XLSX))
rows = [r for n in ("train", "val", "test") for r in sp.read_manifest(str(MAN / f"{n}.tsv"))]
assert len(info) == 14041 and len(rows) == 14041
assert {sp.clip_of(r) for r in rows} == set(info)

def boot_fast(H, ref, clips, seeds, n_boot):
    """Same draws as ro.bootstrap (random.Random(0), rng.choices over the clip list, 1,000 draws, percentile 2.5/97.5),
    but BLEU-4 is formed from per-clip sacreBLEU sufficient statistics summed over the resample."""
    b = BLEU(tokenize="13a")
    R = [detokenize(ref[c]) for c in clips]
    S = {arm: {s: np.array([b._extract_corpus_statistics([detokenize(H[s][arm][c])], [[r]])[0] for c, r in zip(clips, R)], float)
               for s in seeds} for arm in ("SD", "EXP")}
    rng, n, out = random.Random(0), len(clips), []
    for _ in range(n_boot):
        pick = rng.choices(range(n), k=n)
        m = {arm: st.mean(b._compute_score_from_stats(list(S[arm][s][pick].sum(0))).score for s in seeds) for arm in S}
        out.append((m["SD"], m["EXP"] - m["SD"]))
    return [(ro.pct([o[i] for o in out], 0.025), ro.pct([o[i] for o in out], 0.975)) for i in (0, 1)]


# ---- 1. official split: which signers are in test, and are they in training? (spreadsheet Split and Signer_ID columns)
by_split = collections.defaultdict(set)
for c, v in info.items():
    by_split[v["split"]].add(v["signer"])
test_sg, train_sg = by_split["test"], by_split["train"]
print(f"official split: {len(test_sg)} signers in test, {len(test_sg & train_sg)} of them also in training "
      f"(train has {len(train_sg)} signers, 'all nine test signers appear in training': {test_sg <= train_sg})  [c11-084 c14-097]")
assert len(test_sg) == 9 and test_sg <= train_sg

# ---- 2. the split: rebuilt with the original tool, then tied to the saved decodes row for row
pool = [r for r in rows if sp.clip_of(r) not in NO_POSE]
parts, side = sp.build(pool, info)
held = [sp.clip_of(r) for r in parts["SD_test"]]
cnt = collections.Counter(info[c]["signer"] for c in held)
top = cnt.most_common(2)
print(f"\nheld-out clips (signers other than 1 and 2, with pose): {len(held)}; held-out signers {len(cnt)}; "
      f"SD training clips of two signers (signers 1 and 2, outside the validation videos): {len(parts['SD_train'])}  [c18-028 c21-017]")
print(f"signers {top[0][0]} and {top[1][0]} supply {top[0][1]} + {top[1][1]} = {top[0][1] + top[1][1]} clips: "
      f"{100 * (top[0][1] + top[1][1]) / len(held):.1f}% of the {len(held)} clips with pose, "
      f"{100 * (top[0][1] + top[1][1]) / (len(held) + 1):.1f}% of the {len(held) + 1} in the spreadsheet  [c18-027]")
share = 100 * top[0][1] / len(held)
print(f"signer {top[0][0]} alone: {top[0][1]} of {len(held)} clips = {share:.1f}%, rounded {share:.0f}% of the held-out clips  [c18-034]")
assert f"{share:.0f}" == "59"      # the dissertation's rounded figure, checked against the computed share
assert (len(held), len(cnt), len(parts["SD_train"])) == (1087, 47, 12024)

with tempfile.TemporaryDirectory() as tmp:
    man = os.path.join(tmp, "man")
    sp.write(man, parts, side)
    runs = os.path.join(tmp, "runs")
    # data/ holds a subset of each saved decode (hypotheses, SHA-1 of the references, recorded BLEU-4; see SOURCES.md). The
    # decode files the read-out tool expects are rebuilt here, with the references read from the rebuilt manifests; each
    # decode's reference digests must agree with them row for row.
    for a in ro.ARMS:
        for s in SEEDS:
            for split in ("test", "val"):
                d = json.load(open(HERE / "data" / f"{a}_s{s}" / f"beam5_{split}.json"))
                refs = [r.split("\t", 1)[1] for r in sp.read_manifest(os.path.join(man, d["manifest_name"]))]
                assert [hashlib.sha1(r.encode("utf-8")).hexdigest() for r in refs] == d["refs_sha1"], \
                    f"{a} seed {s} {split}: references differ from {d['manifest_name']}"
                os.makedirs(os.path.join(runs, f"{a}_s{s}"), exist_ok=True)
                json.dump(dict(manifest=d["manifest_name"], refs=refs, beam_hyps=d["beam_hyps"],
                               beam_result=dict(metrics={ro.KEY: d["recorded_bleu4"]})),
                          open(os.path.join(runs, f"{a}_s{s}", f"beam5_{split}_best.json"), "w"))
    # gate: every decode is tied to its manifest (same file, references equal row for row, recorded BLEU-4 reproduced)
    for a in ro.ARMS:
        for s in SEEDS:
            ro.load(runs, man, a, s, "test"), ro.load(runs, man, a, s, "val")
    print(f"gate: 18 saved beam-5 decodes (3 arms x 3 seeds, test and validation) match the rebuilt manifests row for row "
          f"and reproduce their recorded BLEU-4")
    H, val, ref = {}, {}, None
    for s in SEEDS:
        H[s], ref, val[s] = ro.one_seed(runs, man, s)
    sub, _ = ro.subsets(man)
    res = {}
    for name in ("all held-out signers", "target unseen (in neither arm's training text)"):
        r = {arm: [ro.score(H[s][arm], ref, sub[name]) for s in SEEDS] for arm in ("SD", "EXP")}
        res[name] = dict(n=len(sub[name]), SD=r["SD"], EXP=r["EXP"],
                         diff=[e["bleu4"] - x["bleu4"] for e, x in zip(r["EXP"], r["SD"])])
    blo, bhi = boot_fast(H, ref, sub["all held-out signers"], SEEDS, 1000)[1]     # clips resampled, 1,000 draws, seed 0
    res["boot"] = [blo, bhi]
    const, fv, ft = ro.floor(man)
    verdict = ro.reading(res["all held-out signers"]["diff"], (blo, bhi), None)

A = res["all held-out signers"]
sd = [x["bleu4"] for x in A["SD"]]
ex = [x["bleu4"] for x in A["EXP"]]
d = A["diff"]
_, tlo, thi = ro.interval(d)
U = res["target unseen (in neither arm's training text)"]
print(f"\nbaseline model, three seeds {SEEDS}, beam width 5, trained on {len(parts['SD_train']):,} clips of signers 1 and 2  [c11-085 c18-028]")
print(f"signers absent from training (SD), held-out clips n = {A['n']:,} of {len(cnt)} other signers [c11-085 c18-029 c21-017]\n  per seed " + " ".join(f"{x:.2f}" for x in sd) + f" | mean {st.mean(sd):.2f} BLEU-4  [c11-085 c18-029]")
print(f"  of which {U['n']} clips have a reference that no training text of either arm contains: "
      f"mean {st.mean(x['bleu4'] for x in U['SD']):.2f} BLEU-4  [c18-029]")
print(f"other clips of the same signers in training (EXP), same {A['n']:,} clips: per seed " + " ".join(f"{x:.2f}" for x in ex)
      + f" | mean {st.mean(ex):.2f}  [c11-085 c18-030]")
print(f"difference EXP - SD: per seed " + " ".join(f"{x:+.2f}" for x in d) + f" | mean {st.mean(d):+.2f}  [c11-086 c18-031]")
print(f"  bootstrap over clips (1,000 draws, test sampling only): [{blo:+.2f}, {bhi:+.2f}]  [c18-031]")
print(f"  t-interval over the three seeds: [{tlo:+.2f}, {thi:+.2f}]  [c18-031]")
r1 = min(m for m in (k / 100 for k in range(300)) if ro.reading([m] * 3, (0.1, 2.0), None).startswith("R1"))
r2 = max(m for m in (k / 100 for k in range(300)) if ro.reading([m] * 3, (-1.4, 1.4), None).startswith("R2"))
print("pre-registered reading (tools/w12g_readout.py, reading()): " + verdict)
print(f"  its thresholds, found by probing reading(): R1 needs a mean difference of at least {r1:.1f} (all seeds positive, interval above 0); "
      f"R2 needs a mean difference below {r2 + 0.01:.1f} (interval inside -1.5 .. +1.5)  [c18-032]")
print(f"input-blind constant output chosen on validation: {ft['bleu4']:.2f} BLEU-4 on the held-out clips "
      f"(validation {fv['bleu4']:.2f})  [c18-033]")
