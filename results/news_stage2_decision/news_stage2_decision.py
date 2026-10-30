"""w4-news Stage-2 decision (R2 protocol, News its own family, m = 1). Written 24 Sep before the nSA_s303/s404 results.
Adopt nSA only if ALL: one-sided 5-seed paired t on beam-5 test deltas p < 0.05; 5-seed mean beam-5 validation delta
> 0; both fresh seeds (303, 404) positive on test. Reads beam5_{val,test}_best.json under /group/.../runs_news."""
import json, statistics as st, sys
from scipy import stats
R = sys.argv[1] if len(sys.argv) > 1 else "/group/pmc099/pprasad/wave4/runs_news"
K = "tokenize=13a;detok=True"; S = [42, 101, 202, 303, 404]
b = lambda arm, s, sp: json.load(open(f"{R}/{arm}_s{s}/beam5_{sp}_best.json"))["beam_result"]["metrics"][K]
rec = {("nA", 42): (2.009, 1.873), ("nA", 101): (2.218, 1.699), ("nA", 202): (2.004, 1.874),
       ("nSA", 42): (1.821, 2.103), ("nSA", 101): (2.257, 2.302), ("nSA", 202): (2.357, 2.273),
       ("nA", 303): (1.558, 1.612)}
for (arm, s), (v, t) in rec.items():   # harness gate: recorded Stage-1 (and nA_s303) values reproduce
    assert abs(b(arm, s, "val") - v) < 0.002 and abs(b(arm, s, "test") - t) < 0.002, (arm, s)
print("GATE OK: recorded News values reproduce from the saved decodes")
for arm in ("nA", "nSA"):
    print(arm, " ".join(f"s{s} {b(arm, s, 'val'):.3f}/{b(arm, s, 'test'):.3f}" for s in S))
dv = [b("nSA", s, "val") - b("nA", s, "val") for s in S]
dt = [b("nSA", s, "test") - b("nA", s, "test") for s in S]
t, p2 = stats.ttest_1samp(dt, 0.0); p1 = p2 / 2 if t > 0 else 1 - p2 / 2
hw = 2.776 * st.stdev(dt) / 5 ** 0.5; hv = 2.776 * st.stdev(dv) / 5 ** 0.5
print(f"nSA - nA val : {' '.join(f'{x:+.3f}' for x in dv)} | mean {st.mean(dv):+.3f} [{st.mean(dv)-hv:+.3f}, {st.mean(dv)+hv:+.3f}] | +{sum(x>0 for x in dv)}/5")
print(f"nSA - nA test: {' '.join(f'{x:+.3f}' for x in dt)} | mean {st.mean(dt):+.3f} [{st.mean(dt)-hw:+.3f}, {st.mean(dt)+hw:+.3f}] | +{sum(x>0 for x in dt)}/5 | one-sided p {p1:.4f}")
c = dict(t_test=p1 < 0.05, val_mean=st.mean(dv) > 0, fresh_seeds=dt[3] > 0 and dt[4] > 0)
print("conditions", c, "->", "ADOPTED (News)" if all(c.values()) else "NOT ADOPTED")
print(f"absolute nSA test mean {st.mean(b('nSA', s, 'test') for s in S):.3f} (published: SL-Transf 2.11, GASLT 2.72, SL-Luong 2.81)")
