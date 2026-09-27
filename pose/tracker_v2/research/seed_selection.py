"""Root cause of v2-tuned's long masked gaps: the SEED tracklet (the one v2 scores best and builds the path from).

v2-tuned scores tracklets by mean box area only (w_cov = 0), so a tracklet of any length can seed the path. This script
  A. lists, for every clip with a masked run > 5 frames (all 60 clips; held-out is DESCRIBED, never used to choose),
     the seed tracklet (length, coverage, mean area), the longest tracklet, and what the labels in the run say;
  B. accounts every masked frame of the 60 clips by the labels inside its run: signer-absent (all panels 'none'),
     signer-present (some panel has the signer), or unlabelled -- the Q2 split into correct masks vs misses;
  C. DEV ONLY: two seed rules, both written into this file BEFORE either was run, scored on DEV labelled panels with
     the acceptance rule of residuals.py (WRONG <= v2-tuned's, HIT >= v2-tuned's, OKMASK unchanged):
       seed=cov*area      score = coverage x mean area  (= summed box area over the clip / T)
       seed cov>=0.25     area-only score, but only tracklets covering >= 25% of the clip may seed (fallback: all)
     DISCLOSURE: these were suggested by the held-out description (video_39_13, video_24_137) in cuts_describe_output.txt,
     so they are NOT scored on held-out here -- any held-out gain would be circular. DEV numbers only.
Usage: seed_selection.py            (writes seed_selection_output.txt)
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residuals as R  # noqa: E402

v2 = R.v2


def feats(d, trk):
    W, H = (int(x) for x in d["WH"])
    return v2.features(trk, d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], W, H, len(d["NDET"]))


def choose_seed(d, rule):
    """v2.choose (TUNED) with only the seed choice replaced; join/extension logic copied verbatim from v2.choose."""
    KA, SA, DB, ND = d["KP_ALL"], d["SC_ALL"], d["DET_BOX"], d["NDET"]
    W, H = (int(x) for x in d["WH"]); T = len(ND)
    tr = v2.tracklets(DB, [v2.candidates(SA[t], DB[t], int(ND[t])) for t in range(T)])
    chosen = np.full(T, -1, int)
    if not tr:
        return chosen
    base = {k: v2.score(tr[k], KA, SA, DB, W, H, T, v2.TUNED) for k in range(len(tr))}
    if rule == "tuned":
        sc = base
    elif rule == "cov*area":
        sc = {k: len(tr[k]) / T * base[k] for k in base}
    elif rule == "cov>=0.25":
        ok = [k for k in base if len(tr[k]) >= 0.25 * T] or list(base)
        sc = {k: (base[k] if k in ok else -np.inf) for k in base}
    best = max(range(len(tr)), key=lambda k: sc[k])
    path = dict(tr[best])
    # extension order stays v2's (by TUNED score), so only the seed differs from v2-tuned
    for k in sorted((k for k in range(len(tr)) if k != best), key=lambda k: -base[k]):
        shared = [t for t in tr[k] if t in path]
        fresh = {t: v for t, v in tr[k].items() if t not in path}
        if not fresh:
            continue
        if shared:
            same = float(np.mean([v2.iou(DB[t, path[t]], DB[t, tr[k][t]]) for t in shared])) >= v2.EXT_IOU
        else:
            t0, t1 = min(fresh), max(fresh)
            before = [t for t in path if t < t0]; after = [t for t in path if t > t1]
            links = []
            if before: tb = max(before); links.append(v2.iou(DB[tb, path[tb]], DB[t0, fresh[t0]]))
            if after: ta = min(after); links.append(v2.iou(DB[ta, path[ta]], DB[t1, fresh[t1]]))
            same = bool(links) and max(links) >= v2.EXT_IOU
        if same:
            path.update(fresh)
    for t, det in path.items():
        chosen[t] = det
    return chosen


def main():
    data, clips, panels = R.load()
    out = []; P = lambda s="": (print(s), out.append(s))
    chos = {}
    for c, d in data.items():                                  # self-check 1: rule 'tuned' == v2.choose(TUNED)
        chos[c] = choose_seed(d, "tuned")
        assert np.array_equal(chos[c], v2.choose(d, v2.TUNED)[1].astype(int)), c
    for part, want in (("dev", (277, 5, 0, 4)), ("heldout", (270, 2, 17, 3))):   # self-check 2: recorded scores
        cnt = Counter(R.outcome(chos[c][f], tr, data[c]["DET_BOX"][f]) for c in data if clips[c] == part
                      for f, tr in panels[c])
        assert (cnt["HIT"], cnt["WRONG"], cnt["MISS"], cnt["OKMASK"]) == want, (part, cnt)
    P("self-checks passed: seed rule 'tuned' == v2.choose(TUNED) on all 60 clips; DEV 277/5/0/4 and held-out "
      "270/2/17/3 reproduce dev_scores.txt / heldout_scores.txt")

    P("\n## A. clips with a masked run > 5 frames: the seed tracklet vs the longest one (held-out = description only)")
    acct = Counter()
    for c, d in data.items():
        info = R.trace(d); T = len(d["NDET"]); lab = dict(panels[c])
        runs = R.gaps(chos[c])
        for s, L, inner in runs:
            ls = [lab[f] for f in range(s, s + L) if f in lab]
            kind = ("unlabelled" if not ls else "signer-absent" if all(x is None for x in ls) else "signer-present")
            acct[(clips[c], "<=5" if L <= 5 else ">5", kind)] += L
        long_runs = [(s, L) for s, L, _ in runs if L > 5]
        if not long_runs:
            continue
        tr, b = info["tr"], info["best"]
        runtxt = "; ".join(f"f{s}-{s + L - 1} ({L} fr, labels "
                           f"{dict(Counter('none' if lab[f] is None else 'signer' for f in range(s, s + L) if f in lab))})"
                           for s, L in long_runs)
        P(f"[{clips[c]}] {c} T={T}: masked {runtxt}")
        # the seed, then every other tracklet covering >= 25% of the clip (two people can span the same frames)
        for k in [b] + sorted((k for k in range(len(tr)) if k != b and len(tr[k]) >= 0.25 * T), key=lambda k: -len(tr[k])):
            lb = Counter(R.outcome(tr[k][f], t_, d["DET_BOX"][f]) for f, t_ in panels[c] if f in tr[k])
            P(f"    {'SEED' if k == b else 'long'}: {len(tr[k])} fr (f{min(tr[k])}-{max(tr[k])}), mean area "
              f"{feats(d, tr[k])['area']:.3f}, cov*area {len(tr[k]) / T * feats(d, tr[k])['area']:.3f}, "
              f"labelled {dict(lb)}")

    P("\n## A2. DEV WRONG panels: the seed tracklet, camera cuts inside it (cuts.py measure), chosen vs signer box")
    import cuts                                                   # local signer-crop mp4s only (no download)
    for c, d in data.items():
        if clips[c] != "dev":
            continue
        info = R.trace(d); DB = d["DET_BOX"]; ch = chos[c]
        wr = [(f, t_) for f, t_ in panels[c] if R.outcome(ch[f], t_, DB[f]) == "WRONG"]
        if not wr:
            continue
        dif = cuts.frame_diffs(c, len(d["NDET"])); seed = info["tr"][info["best"]]
        inside = [t for t in range(min(seed), max(seed)) if dif[t] >= cuts.CUT]
        P(f"  {c}: seed {len(seed)} fr f{min(seed)}-{max(seed)}, cuts inside the seed after frames {inside}")
        for f, t_ in wr:
            a = DB[f, ch[f]]; ar = (a[2] - a[0]) * (a[3] - a[1])
            if t_ is None:
                P(f"    f{f}: signer labelled not detected; chosen box {np.round(a).astype(int).tolist()}")
            else:
                s = DB[f, t_]; sr = (s[2] - s[0]) * (s[3] - s[1])
                inter = max(0, min(a[2], s[2]) - max(a[0], s[0])) * max(0, min(a[3], s[3]) - max(a[1], s[1]))
                P(f"    f{f}: IoU(chosen, signer) {v2.iou(a, s):.2f}, chosen/signer area {ar / sr:.2f}, share of "
                  f"signer box inside chosen {inter / sr:.2f}; chosen {np.round(a).astype(int).tolist()} signer "
                  f"{np.round(s).astype(int).tolist()}")

    P("\n## B. every masked frame of the 60 clips, by run length and by the labels inside its run")
    tot = sum(acct.values())
    for part in ("dev", "heldout"):
        for ln in ("<=5", ">5"):
            row = {k: acct[(part, ln, k)] for k in ("signer-absent", "signer-present", "unlabelled")}
            P(f"  {part:7s} run {ln}: {row}")
    for k in ("signer-absent", "signer-present", "unlabelled"):
        n = sum(v for (p, ln, kk), v in acct.items() if kk == k)
        P(f"  ALL {k:15s}: {n} of {tot} masked frames ({n / tot:.1%})")

    P("\n## C. DEV ONLY seed rules (not scored on held-out). labelled HIT/WRONG/MISS/OKMASK | masked frames")
    base = None
    for rule in ("tuned", "cov*area", "cov>=0.25"):
        cnt, masked, changed = Counter(), 0, []
        for c, d in data.items():
            if clips[c] != "dev":
                continue
            ch = chos[c] if rule == "tuned" else choose_seed(d, rule)
            masked += int((ch < 0).sum())
            if not np.array_equal(ch, chos[c]):
                changed.append(c)
            for f, tr in panels[c]:
                cnt[R.outcome(ch[f], tr, d["DET_BOX"][f])] += 1
        row = (cnt["HIT"], cnt["WRONG"], cnt["MISS"], cnt["OKMASK"])
        base = base or row
        ok = row[1] <= base[1] and row[0] >= base[0] and row[3] == base[3]
        P(f"  seed={rule:10s} {row[0]}/{row[1]}/{row[2]}/{row[3]} | {masked:3d} | clips changed {changed} | "
          f"passes DEV rule: {ok}")
    P("\n## D. DEV: what the frozen v2.1 candidate (residuals.py 'reattach IoU 0.5 (<=5 fr)') actually assigns")
    for c, d in data.items():
        if clips[c] != "dev":
            continue
        new = R.reattach(d, chos[c], 0.5, 5)
        lab = dict(panels[c])
        for t in np.flatnonzero((chos[c] < 0) & (new >= 0)):
            pf = [s for s in np.flatnonzero(chos[c] >= 0)]; s = min(pf, key=lambda s: abs(s - t))
            ref = ("unlabelled" if s not in lab else R.outcome(chos[c][s], lab[s], d["DET_BOX"][s]))
            sa = d["SC_ALL"][t, new[t]]
            P(f"  {c} f{t}: assigned det {new[t]} (body conf mean {sa[:23].mean():.2f}, body kps > {v2.CONF}: "
              f"{int((sa[:23] > v2.CONF).sum())}/23, hands {sa[R.LH].mean():.2f}/{sa[R.RH].mean():.2f}); the path frame "
              f"it extends is f{s}, labelled {ref}")
    (Path(__file__).parent / "seed_selection_output.txt").write_text("\n".join(out) + "\n")


def selftest():
    """Seed rules on a toy clip: a 1-frame large box vs a 10-frame smaller one, far apart (no join possible)."""
    T = 11; ND = np.ones(T, int); DB = np.zeros((T, 1, 4), np.float32)
    DB[:10, 0] = [100, 100, 300, 500]; DB[10, 0] = [600, 0, 1000, 800]
    d = dict(NDET=ND, WH=np.array([1000, 1000]), DET_BOX=DB, KP_ALL=np.zeros((T, 1, 133, 2), np.float32),
             SC_ALL=np.full((T, 1, 133), 0.9, np.float32))
    assert list(choose_seed(d, "tuned")) == [-1] * 10 + [0]           # area-only seeds on the 1-frame giant
    assert list(choose_seed(d, "cov*area")) == [0] * 10 + [-1]
    assert list(choose_seed(d, "cov>=0.25")) == [0] * 10 + [-1]
    print("seed self-test OK")


if __name__ == "__main__":
    selftest()
    main()
