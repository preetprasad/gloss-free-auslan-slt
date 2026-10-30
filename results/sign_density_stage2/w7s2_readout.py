"""Wave 7 Stage 2 (job 66644) readout, as fixed in results/PREREG_WAVE7_DENSITY_2026-09-23.md sections 4 and 10.

D adoption (family {D}, m = 1): the five paired D - A deltas, seeds 42/101/202 (Stage 1, job 31220) + 303/404 (Stage 2),
go to tools/holm_decision.py --family; adoption needs (a) one-sided 5-seed paired t on beam-5 TEST rejecting at alpha 0.05,
(b) 5-seed mean beam-5 VALIDATION delta > 0, (c) D - A positive on beam-5 TEST at both fresh seeds 303 and 404.
DL (D's signs, sign lengths rescaled per clip to A's frame count) is descriptive on 303/404 only; the section-10 reading:
if D is adopted -> "composition" / "length" / "not separable"; if D is not adopted -> no attribution, estimate only.

Inputs: summary_s{303,404}.json (w7s2_summaries.py, gated), ../sign_density_stage1/summary_A_D_s{42,101,202}.json (Stage 1,
re-gated by its own w7_summaries.py), readouts_*.txt here and in ../sign_density_stage1 (tools/arm_readouts.py).
Gates: every readout BLEU equals its summary BLEU to 1e-6; unseen n = 570 on every test dump.
"""
import json, os, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.join(HERE, "..", "sign_density_stage1")
TOOLS = os.path.join(HERE, "..", "..", "src", "tools")
SEEDS1, SEEDS2, ALL = (42, 101, 202), (303, 404), (42, 101, 202, 303, 404)
T975 = {2: 4.303, 4: 2.776}   # two-sided 95% t, df = n - 1 (descriptive intervals only)

summ = {s: json.load(open(os.path.join(S1, f"summary_A_D_s{s}.json"))) for s in SEEDS1}
summ.update({s: json.load(open(os.path.join(HERE, f"summary_s{s}.json"))) for s in SEEDS2})


def readouts(d, name, arms, seeds):
    out = {}
    for l in open(os.path.join(d, name)):
        k, js = l.split(" ", 1); out[k] = json.loads(js)
    assert set(out) == {f"{a}_s{s}" for a in arms for s in seeds}, (d, name, sorted(out))
    return out


rd = {}
for ck in ("best", "soup"):
    for sp in ("val", "test"):
        name = f"readouts_{'soup_' if ck == 'soup' else ''}{sp}.txt"
        rd[ck, sp] = {**readouts(S1, name, "AD", SEEDS1), **readouts(HERE, name, ("A", "D", "DL"), SEEDS2)}
        for k, v in rd[ck, sp].items():
            a, s = k.rsplit("_s", 1)
            ref = summ[int(s)][a][f"{'soup_' if ck == 'soup' else ''}beam5_{sp}"]
            assert abs(v["bleu4"] - ref) < 1e-6, (ck, sp, k, v["bleu4"], ref)
            assert v["n"] == 800 and (sp != "test" or v["unseen_n"] == 570), (ck, sp, k)
print("GATES PASSED: readout BLEU = summary BLEU for all 48 dumps (Stage 1 24 + Stage 2 24); unseen n = 570 on every test dump\n")


def ci(d):
    m = st.mean(d); hw = T975[len(d) - 1] * st.stdev(d) / len(d) ** 0.5
    return m, (f"{' '.join(f'{x:+.3f}' for x in d)} | {sum(x > 0 for x in d)}/{len(d)} + | mean {m:+.3f} "
               f"[{m - hw:+.3f}, {m + hw:+.3f}]")


def delta(a, b, key, seeds):
    return [summ[s][a][key] - summ[s][b][key] for s in seeds]


print("== Stage 2 absolute values, seeds 303 / 404")
for key in ("beam5_val", "beam5_test", "soup_beam5_val", "soup_beam5_test", "val_best", "greedy_test", "best_epoch"):
    print(f"  {key:16s} " + " | ".join(f"{a} " + " ".join(f"{summ[s][a][key]:7.3f}" for s in SEEDS2) for a in ("A", "D", "DL")))

print("\n== Stage 2 paired deltas, seeds 303 404 | positive | mean [95% t, df 1 not shown: n = 2]")
for a, b in (("D", "A"), ("DL", "A"), ("D", "DL")):
    for key in ("beam5_val", "beam5_test", "soup_beam5_val", "soup_beam5_test"):
        d = delta(a, b, key, SEEDS2)
        print(f"  {a:2s} - {b:2s} {key:16s} {' '.join(f'{x:+.3f}' for x in d)} | {sum(x > 0 for x in d)}/2 + | mean {st.mean(d):+.3f}")

val5, test5 = delta("D", "A", "beam5_val", ALL), delta("D", "A", "beam5_test", ALL)
print("\n== D - A, all five seeds 42 101 202 303 404 (beam-5 best.pt) | positive | mean [two-sided 95% t, df 4]")
print(f"  VALIDATION {ci(val5)[1]}")
print(f"  TEST       {ci(test5)[1]}")

fam = {"alpha": 0.05, "arms": {"D": {"val": val5, "test": test5, "n_stage1": 3}}}
fam_path = os.path.join(HERE, "family_D.json")
json.dump(fam, open(fam_path, "w"), indent=1)
print("\n== tools/holm_decision.py --family family_D.json (family {D}, m = 1)")
out = subprocess.run([sys.executable, os.path.join(TOOLS, "holm_decision.py"), "--family", fam_path],
                     capture_output=True, text=True, check=True).stdout
print("  " + out.strip().replace("\n", "\n  "))

sys.path.insert(0, TOOLS)
from holm_decision import adopt, holm, one_sided_p  # noqa: E402
t, p = one_sided_p(test5)
rej = holm([("D", p)])[0][3]
dec = adopt(val5, test5, rej, 3)
print(f"\n  cross-check (same functions, imported): t = {t:+.3f}, one-sided p = {p:.4f}, Holm reject = {rej}; {dec}")
assert ("ADOPT" in out.upper()) or not dec["adopt"]

print("\n== Pooled secondaries D - A over five seeds (best.pt) | positive | mean [two-sided 95% t, df 4]")
for sp in ("test", "val"):
    r = rd["best", sp]
    print(f"  -- {sp}")
    for k in ("bleu4", "bleu4_unseen", "content_precision", "content_recall", "wer", "len_ratio", "distinct"):
        lab = f"{k} (n={r['A_s42']['unseen_n']})" if k == "bleu4_unseen" else k
        print(f"     {lab:22s} {ci([r[f'D_s{s}'][k] - r[f'A_s{s}'][k] for s in ALL])[1]}")
print("  -- soup_last5 (NEVER a criterion), test")
r = rd["soup", "test"]
for k in ("bleu4", "bleu4_unseen"):
    print(f"     {k:22s} {ci([r[f'D_s{s}'][k] - r[f'A_s{s}'][k] for s in ALL])[1]}")

print("\n== DL secondaries, seeds 303 404 (best.pt): DL - A and D - DL")
for sp in ("test", "val"):
    r = rd["best", sp]
    for k in ("bleu4", "bleu4_unseen", "content_precision", "content_recall", "wer", "len_ratio", "distinct"):
        dla = [r[f"DL_s{s}"][k] - r[f"A_s{s}"][k] for s in SEEDS2]
        ddl = [r[f"D_s{s}"][k] - r[f"DL_s{s}"][k] for s in SEEDS2]
        print(f"  {sp:4s} {k:18s} DL-A {' '.join(f'{x:+.3f}' for x in dla)} (mean {st.mean(dla):+.3f}) | "
              f"D-DL {' '.join(f'{x:+.3f}' for x in ddl)} (mean {st.mean(ddl):+.3f})")

print("\n== Pre-registered reading (section 10)")
print(f"  D adoption: (a) Holm reject on test = {dec['holm_reject']}; (b) 5-seed val mean > 0 = {dec['val_mean_positive']}; "
      f"(c) both fresh seeds positive on test = {dec['fresh_seeds_positive']} -> {'ADOPTED' if dec['adopt'] else 'NOT ADOPTED'}")
if dec["adopt"]:
    dl_a, d_a = st.mean(delta("DL", "A", "beam5_test", SEEDS2)), st.mean(delta("D", "A", "beam5_test", SEEDS2))
    reading = ("composition (fewer signs)" if d_a != 0 and dl_a / d_a >= 0.5 and (dl_a > 0) == (d_a > 0)
               else "length (shorter clips)" if dl_a <= 0 < d_a else "not separable")
else:
    reading = "D not adopted -> no attribution; DL reported as an estimate only"
print(f"  DL: {reading}")
