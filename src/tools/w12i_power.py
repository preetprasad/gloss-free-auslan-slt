"""Wave 12i planning numbers: how often does rule (a) of PREREG_WAVE12I fire (one-sided paired t over five test DiDs,
alpha 0.05), if the five per-seed heavy-bin DiDs were Normal(mu, sd)?  Planning only: it reads no result.

  python tools/w12i_power.py          # prints the table and runs its own check (mu = 0 must fire about 5% of the time)

sd 1.196 is the standard deviation of the three wave-9 test values (itself uncertain); 2.0 is a stress value. The
validation condition (b) is not simulated. Uses tstats() of tools/w12i_heavy_did.py, so the p-value is the one the
read-out will print.
"""
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w12i_heavy_did import tstats  # noqa: E402

N, DRAWS = 5, 20000


def power(mu, sd, seed=0):
    rng = random.Random(seed)
    return sum(tstats([rng.gauss(mu, sd) for _ in range(N)])["p_one_sided"] < 0.05 for _ in range(DRAWS)) / DRAWS


if __name__ == "__main__":
    print(f"rule (a) fires, {N} seeds, {DRAWS} simulated read-outs per cell (seed 0)")
    print("  true mean DiD   sd 1.196   sd 2.0")
    for mu, why in ((0.0, "no effect"), (0.59, "wave-3 value, tracker v1"), (1.14, "the planning threshold"),
                    (2.28, "wave-9 test value")):
        print(f"  {mu:5.2f}           {power(mu, 1.196):.2f}       {power(mu, 2.0):.2f}    {why}")
    assert 0.04 < power(0.0, 1.196) < 0.06, "type I error is not about 5%"
    print("check passed")
