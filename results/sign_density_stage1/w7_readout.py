"""Wave 7 (job 31220) pre-registered readout: paired D - A per seed (42/101/202), sign counts, two-sided 95% seed-level
t-intervals (df = 2, t = 4.303), and the Stage-1 futility rule of results/PREREG_WAVE7_DENSITY_2026-09-23.md §4.

Inputs (all produced and gated in this directory): summary_A_D_s{seed}.json (w7_summaries.py), readouts_{test,val}.txt
and readouts_soup_{test,val}.txt (tools/arm_readouts.py with the 12,439-row real train manifest).
Gates: unseen n = 570 on test in every run; each readout's BLEU equals the summary's beam-5 BLEU to 1e-6 (same dump).
The criterion is beam-5 VALIDATION of best.pt only; everything else, including the soup, is descriptive.
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, T975 = (42, 101, 202), 4.303


def ci(d):
    m = st.mean(d); hw = T975 * st.stdev(d) / len(d) ** 0.5
    return m, f"{' '.join(f'{x:+.3f}' for x in d)} | {sum(x > 0 for x in d)}/3 + | mean {m:+.3f} [{m - hw:+.3f}, {m + hw:+.3f}]"


def readouts(name):
    out = {}
    for l in open(os.path.join(HERE, name)):
        k, js = l.split(" ", 1); out[k] = json.loads(js)
    assert set(out) == {f"{a}_s{s}" for a in "AD" for s in SEEDS}, name
    return out


summ = {s: json.load(open(os.path.join(HERE, f"summary_A_D_s{s}.json"))) for s in SEEDS}
rd = {(ck, sp): readouts(f"readouts_{'soup_' if ck == 'soup' else ''}{sp}.txt") for ck in ("best", "soup") for sp in ("val", "test")}
for (ck, sp), r in rd.items():
    for k, v in r.items():
        a, s = k.split("_s")
        assert abs(v["bleu4"] - summ[int(s)][a][f"{'soup_' if ck == 'soup' else ''}beam5_{sp}"]) < 1e-6, (ck, sp, k)
        assert v["n"] == 800 and (sp != "test" or v["unseen_n"] == 570), (ck, sp, k, v["unseen_n"])
print("GATES PASSED: readout BLEU = summary BLEU for all 24 dumps; unseen n = 570 on test in all 12 test dumps\n")

print("== absolute values, A / D per seed 42 101 202")
for key in ("beam5_val", "beam5_test", "soup_beam5_val", "soup_beam5_test", "val_best", "greedy_test", "best_epoch"):
    print(f"  {key:16s} A " + " ".join(f"{summ[s]['A'][key]:7.3f}" for s in SEEDS)
          + " | D " + " ".join(f"{summ[s]['D'][key]:7.3f}" for s in SEEDS))

print("\n== D - A, paired per seed (42 101 202) | positive seeds | mean [two-sided 95% seed-level t, df 2]")
crit, line = ci([summ[s]["D"]["beam5_val"] - summ[s]["A"]["beam5_val"] for s in SEEDS])
print(f"  beam-5 VAL  best.pt (STAGE-1 CRITERION) {line}")
print(f"  beam-5 TEST best.pt                      {ci([summ[s]['D']['beam5_test'] - summ[s]['A']['beam5_test'] for s in SEEDS])[1]}")
for k in ("val_best", "greedy_test"):
    print(f"  {k:12s} (greedy, descriptive)      {ci([summ[s]['D'][k] - summ[s]['A'][k] for s in SEEDS])[1]}")
for ck in ("best", "soup"):
    tag = "best.pt" if ck == "best" else "soup_last5 (NEVER a criterion)"
    for sp in ("test", "val"):
        r = rd[ck, sp]
        print(f"  -- {tag}, {sp}")
        for k in ("bleu4", "bleu4_unseen", "content_precision", "content_recall", "wer", "len_ratio", "distinct"):
            lab = f"{k} (n={r['A_s42']['unseen_n']})" if k == "bleu4_unseen" else k
            print(f"     {lab:22s} {ci([r[f'D_s{s}'][k] - r[f'A_s{s}'][k] for s in SEEDS])[1]}")

print("\n== Stage-1 futility (prereg §4): stop if the 3-seed mean paired beam-5 validation delta D - A is <= 0")
verdict = "STOP (D enters any family with p = 1)" if crit <= 0 else \
    "CONTINUE to Stage 2: seeds 303/404, fresh in-batch A, plus arm DL (D length-matched to A per clip)"
print(f"  mean = {crit:+.4f} -> {verdict}")
