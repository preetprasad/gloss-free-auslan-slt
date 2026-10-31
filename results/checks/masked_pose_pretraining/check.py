"""Wave 28 (masked keypoint pre-training): what the dissertation rows state, checked against the saved job output.

Reads the summary (job 239450) and the training log of the failed control task (job 239449 task 2). The wave's
reproduction gate failed, so no score is quoted here: only the pre-training check, the seed counts and the gate.
"""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S = (HERE / "summary_239450.out").read_text()
T = (HERE / "train_task2_239449.out").read_text()

# GATE: the summary's own pre-training check lines (expected: nine runs, MD/MDA/MDR x 3 seeds)
pre = re.findall(r"^\s+(MD1?|MDA1?|MDR1?)_s(\d+): (TRAINED|NOT TRAINED)[^\n]*?is ([\d.]+) of the first evaluation", S, re.M)
print("read from the saved job output summary_239450.out (job 239450)")
if len(pre) != 9:
    sys.exit(f"GATE FAILED: expected 9 pre-training lines, found {len(pre)}")
n_tr = sum(1 for p in pre if p[2] == "TRAINED")
print(f"pre-training runs reported: {len(pre)}; TRAINED under the pre-registered check: {n_tr}  [c17-110]")
print(f"nine pre-training runs, all nine met the check: {n_tr == 9}  [c17-110]")
ratios = [float(p[3]) for p in pre]
print(f"final held-out loss / first evaluation: {min(ratios):.3f} to {max(ratios):.3f} (line 0.5)  [c17-110]")

arms = {}
for a, s, *_ in pre:
    arms.setdefault(a.rstrip("1"), set()).add(int(s))
for a in ("MD", "MDA", "MDR"):
    print(f"arm {a}: pre-training seeds {sorted(arms[a])} = {len(arms[a])} seeds  [c17-109]")
fin = [int(u) for u in re.findall(r"at update (\d+)\)", S)]
print(f"latest update at which a held-out loss was evaluated: {max(fin)} (no evaluation beyond it)")
print("(not checked: the 44,010 dictionary-clip count and the pre-training pool sizes are not printed by any saved job output; see claims.json)")

# controls: the recipe arms that finished vs the gate
done = re.findall(r"^\s+(S|MD|MDA|MDR) beam-5 val/test (.*?) \|", S, re.M)
for a, body in done:
    seeds = re.findall(r"s(\d+) ", body + " ")
    print(f"arm {a}: finished recipe runs for seeds {seeds}  [c03-021]")
gate = re.search(r"^GATE: control S has seeds \[([\d, ]+)\]; the gate needs ([\d, and]+): HARNESS NOT REPRODUCED", S, re.M)
print(f"control S finished for seeds [{gate.group(1)}] of the three needed: {len(gate.group(1).split(','))} of 3  [c03-021]")
fail = re.search(r"FAIL: G-ep1: arm S seed (\d+) epoch-1 line is '([^']*)', wave 5's control C has '([^']*)'", T)
c1 = re.search(r"ctc=([\d.]+)", fail.group(2)).group(1); c2 = re.search(r"ctc=([\d.]+)", fail.group(3)).group(1)
print(f"reproduction check, control seed {fail.group(1)}: epoch-1 ctc loss {c1} here vs {c2} recorded for wave 5 (third decimal differs)  [c03-021 c17-110]")
if c1 == c2:
    sys.exit("GATE FAILED: the logged reproduction check did not fail")
print(f"number of control seeds failing the reproduction check: {len(set(re.findall(r'FAIL: arm S seed (\d+)', T)))}  [c03-021]")
no = len(re.findall(r"NO VERDICT \(S seed 202 missing", S))
print(f"every contrast with the control is labelled NO VERDICT in the summary: {no} lines; contrasts without it carry 'HARNESS NOT REPRODUCED'  [c03-021]")

rule = re.findall(r"RULE \(pre-registration section 5\): HARNESS NOT REPRODUCED (\w+[ \w]*?) \(3-seed mean beam-5 validation difference", S)
print(f"decision rule of the summary reads the 3-seed mean beam-5 VALIDATION difference ({len(rule)} contrasts: {', '.join(rule)}); the criterion is validation, not test  [c17-109]")
