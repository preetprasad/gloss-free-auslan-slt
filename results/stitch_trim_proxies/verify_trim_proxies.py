"""Adversarial re-check of the w4-trim proxies readout (Kaya job 25865), independent of trim_proxies_readout.py.

Reads the raw log, the local tool source, the E0 record and the wave-4 training summaries directly.
Run from the repo root:  python results/stitch_trim_proxies/verify_trim_proxies.py
"""
import hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/stitch_trim_proxies/w4-trim-prox-25865.out"
TOOL = ROOT / "src/tools/audit_posestitch_motion.py"
SUMM = ROOT / "results/wave4_tq"

# --- self-check 1: reproduce the RECORDED w4-trim training outcome (PREREG_WAVE4 results, 2026-09-22) ---
T975_DF2 = 4.303
def seed_ci(d):
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    h = T975_DF2 * sd / math.sqrt(n)            # df = n_seeds - 1 = 2
    return m, m - h, m + h
dv, dt = [], []
for s in (42, 101, 202):
    j = json.loads((SUMM / f"summary_w4A_w4T_w4Q_s{s}.json").read_text())
    dv.append(j["w4T"]["beam5_val"] - j["w4A"]["beam5_val"])
    dt.append(j["w4T"]["beam5_test"] - j["w4A"]["beam5_test"])
mv, lv, hv = seed_ci(dv); mt, lt, ht = seed_ci(dt)
assert [round(x, 3) for x in dv] == [-0.116, 0.778, 0.106], dv
assert [round(x, 3) for x in dt] == [-0.538, -0.556, -0.232], dt
assert round(mv, 3) == 0.256 and round(mt, 3) == -0.442
assert (round(lv, 2), round(hv, 2)) == (-0.90, 1.41)
assert abs(lt - -0.895) < 0.002 and abs(ht - 0.011) < 0.002   # recorded [-0.895, +0.011]
print(f"[self-check] recorded T-A reproduced: val {[f'{x:+.3f}' for x in dv]} mean {mv:+.3f} "
      f"[{lv:+.3f}, {hv:+.3f}]; test {[f'{x:+.3f}' for x in dt]} mean {mt:+.3f} [{lt:+.3f}, {ht:+.3f}], "
      f"{sum(x > 0 for x in dt)}/3 positive on test, {sum(x > 0 for x in dv)}/3 on val")

# --- self-check 2: E0 recorded ratio = median join / median real interior ---
assert round(0.2613 / 0.0539, 2) == 4.85 and round(0.2607 / 0.0566, 2) == 4.61

# --- raw log: md5 line, then two blocks, each the tail of a pretty-printed JSON object ---
lines = LOG.read_text().splitlines()
md5_ran, fname = lines[0].split()
assert fname == "tools/audit_posestitch_motion.py"
assert md5_ran == hashlib.md5(TOOL.read_bytes()).hexdigest(), "tool changed since the run"
assert lines[-1] == "TRIM_PROXIES_DONE" and not any(l.startswith("FAIL") for l in lines)
heads = [i for i, l in enumerate(lines) if l.startswith("== ")]
assert [lines[i] for i in heads] == ["== A", "== T"]

def grab(body, key):
    """Brace-match the object that follows '"key": {' in a list of lines (no regex)."""
    txt = "\n".join(body); i = txt.index(f'"{key}": {{') + len(f'"{key}": ')
    depth = 0
    for k in range(i, len(txt)):
        depth += txt[k] == "{"; depth -= txt[k] == "}"
        if depth == 0:
            return json.loads(txt[i:k + 1])

arms = {}
for h, nxt in zip(heads, heads[1:] + [len(lines) - 1]):
    body = lines[h + 1:nxt]
    assert len(body) == 25, len(body)                     # exactly the sbatch's `tail -25`
    txt = "\n".join(body)
    arms[lines[h][3:]] = {
        "join": grab(body, "join_gap_synthetic"), "real": grab(body, "interior_gap_real"),
        "shuf": grab(body, "shuffled_label_control"),
        "verdict": txt.split('"verdict": "')[1].split('"')[0],
        # what the tail CUT OFF -- these keys never appear, so the values are unknowable locally:
        "missing": [k for k in ("matched_pairs", '"heldout_balanced_accuracy"', "velocity_median",
                                "pause_density", "length_only_baseline", "pause_threshold") if k not in txt],
    }

A, T = arms["A"], arms["T"]
assert A["real"] == T["real"]            # identical real side to full float precision
for name, d in arms.items():
    j, r, s = d["join"], d["real"], d["shuf"]
    print(f"{name}: join med {j['median']:.3f} mean {j['mean']:.3f} p10 {j['p10']:.3f} p90 {j['p90']:.3f} n {j['n']} | "
          f"real med {r['median']:.3f} mean {r['mean']:.3f} n {r['n']} | ratio med {j['median']/r['median']:.3f}x "
          f"mean {j['mean']/r['mean']:.3f}x | shuffled mean {s['balanced_accuracy_mean']:.3f} "
          f"p95 {s['balanced_accuracy_p95']:.3f} (n_perm {s['n_permutations']}) | verdict: {d['verdict']}")
    print(f"   not in local log: {d['missing']}")
red_med = 1 - T["join"]["median"] / A["join"]["median"]
red_e0b = 1 - 3.43 / 4.85
print(f"T vs A join-median reduction {red_med:.3f} (= ratio reduction, real side identical); E0b 1-3.43/4.85 = {red_e0b:.3f}")
print(f"production / local ratio: A {A['join']['median']/A['real']['median']/4.85:.3f}, "
      f"T {T['join']['median']/T['real']['median']/3.43:.3f}")
# verdict logic (tool lines 257-261): string means main bal-acc > shuffled p95 AND > length-only bal-acc
for name, d in arms.items():
    assert d["verdict"] == "dynamics separate real from synthetic beyond both controls"
    print(f"{name}: separability > {d['shuf']['balanced_accuracy_p95']:.3f} (strict lower bound); exact value, "
          f"margin and length-only baseline unknown locally")
