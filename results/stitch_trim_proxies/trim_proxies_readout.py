"""Readout of Kaya job 25865 (w4-trim proxies, corpora A = --trim_frac 0.15, T = --trim_frac 0.0).

The job printed only `tail -25` of tools/audit_posestitch_motion.py's stdout per arm, so the local log
carries join_gap_synthetic, interior_gap_real, the shuffled-label control and the verdict. Velocity,
pause density and the exact separability live only in /group/.../wave4/trim_proxies/proxies_{A,T}.json.
Ratio definition = E0's "join / real interior" = median synthetic join gap / median real interior gap.
Run: python results/stitch_trim_proxies/trim_proxies_readout.py  (from the repo root)
"""
import hashlib, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "w4-trim-prox-25865.out"
TOOL = HERE.parents[1] / "src/tools/audit_posestitch_motion.py"

# Self-check 1: E0 (results/E0_MOTION_FIDELITY_2026-09-09.md) recorded the ratio as median/median.
assert f"{0.2613 / 0.0539:.2f}" == "4.85" and f"{0.2607 / 0.0566:.2f}" == "4.61", "E0 ratio definition"

text = LOG.read_text()
# Self-check 2: the job ran the local tool byte-for-byte.
ran_md5 = text.split()[0]
assert ran_md5 == hashlib.md5(TOOL.read_bytes()).hexdigest(), "tool md5 differs from the one Kaya ran"
assert text.rstrip().endswith("TRIM_PROXIES_DONE")


def block(sec, key):
    m = re.search(rf'"{key}": (\{{.*?\}}|"[^"]*")', sec, re.S)
    return json.loads(m.group(1))


arms = {}
for name, sec in re.findall(r"== (A|T)\n(.*?)(?=\n== |\nTRIM_PROXIES_DONE)", text, re.S):
    arms[name] = {k: block(sec, k) for k in
                  ("join_gap_synthetic", "interior_gap_real", "shuffled_label_control", "verdict")}
assert set(arms) == {"A", "T"}
# Design check: same seed + same real manifest + per-segment-paired corpora -> identical real side.
assert arms["A"]["interior_gap_real"] == arms["T"]["interior_gap_real"]

out = {}
for a, d in arms.items():
    j, r, s = d["join_gap_synthetic"], d["interior_gap_real"], d["shuffled_label_control"]
    out[a] = {"join_median": j["median"], "join_mean": j["mean"], "join_n": j["n"],
              "real_interior_median": r["median"], "real_interior_mean": r["mean"],
              "join_over_real_median": j["median"] / r["median"], "join_over_real_mean": j["mean"] / r["mean"],
              "shuffled_mean": s["balanced_accuracy_mean"], "shuffled_p95": s["balanced_accuracy_p95"],
              "verdict": d["verdict"]}
    print(f"{a}: join median {j['median']:.3f} (n {j['n']}), real interior median {r['median']:.3f} -> "
          f"ratio {out[a]['join_over_real_median']:.3f}x (means: {out[a]['join_over_real_mean']:.3f}x); "
          f"shuffled-label bal-acc mean {s['balanced_accuracy_mean']:.3f} p95 {s['balanced_accuracy_p95']:.3f}; "
          f"verdict: {d['verdict']}")
red = 1 - out["T"]["join_median"] / out["A"]["join_median"]
print(f"T vs A join-gap median reduction {red:.3f}; E0b local reduction 1 - 3.43/4.85 = {1 - 3.43 / 4.85:.3f}")
print("velocity median / pause density / exact separability + length-only baseline: NOT in the local log")
