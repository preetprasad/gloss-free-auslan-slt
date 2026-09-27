"""Seed-level 95% t intervals (df 4) for the Q2 single differences (V050 − R050 on held and kept test cells), which
tools/learning_curve.py prints per seed but does not summarise. Inputs are its printed 3-decimal values (analysis.txt),
so means carry <= 0.0005 rounding. Self-check: held − kept reproduces the tool's DiD official mean −0.783 first."""
import statistics as st

held = [1.077, 0.592, -0.019, -1.106, 0.368]
kept = [0.712, 0.382, 1.327, 0.873, 1.534]
did = [h - k for h, k in zip(held, kept)]
assert abs(st.mean(did) - (-0.783)) < 0.001, st.mean(did)
print(f"self-check OK: held − kept mean {st.mean(did):+.3f} = tool DiD official −0.783")
for name, d in (("single diff held (V050 − R050)", held), ("single diff kept (V050 − R050)", kept)):
    m = st.mean(d); hw = 2.776 * st.stdev(d) / 5 ** 0.5
    print(f"{name}: mean {m:+.3f} [{m - hw:+.3f}, {m + hw:+.3f}] | +{sum(x > 0 for x in d)}/5")
