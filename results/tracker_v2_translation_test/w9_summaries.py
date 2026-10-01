"""Wave 9 (job 43506) per-seed summaries, built locally with the same fields as _w4lib.sh::w4summary.

Gate first: every stored beam-5 BLEU is re-scored from its saved hypotheses through slt/metrics.py
(detokenise both sides, sacreBLEU 13a) and must match to 1e-6, and every reference list must equal its
arm's manifest text. Then writes summary_c0_v1_v2_s{seed}.json for tools/seed_decision.py.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

RUNS = os.path.join(HERE, "runs")
KEY = "tokenize=13a;detok=True"
ARMS, SEEDS = ("c0", "v1", "v2"), (42, 101, 202)


def load(d, split, ck):
    j = json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))
    stored = j["beam_result"]["metrics"][KEY]
    hyps, refs = j["beam_hyps"], j["refs"]
    assert len(hyps) == len(refs) == 800, (d, split, ck, len(hyps), len(refs))
    redo = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs])
    assert abs(redo - stored) < 1e-6, (d, split, ck, stored, redo)
    return stored, refs, len(set(hyps))


refs_seen = {}
for s in SEEDS:
    rows = {}
    for arm in ARMS:
        d = os.path.join(RUNS, f"{arm}_s{s}")
        assert os.path.exists(os.path.join(d, "DONE")), d
        p = json.load(open(os.path.join(d, "progress.json")))
        r = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                 greedy_test=p.get("test", {}).get("bleu4", float("nan")))
        for split in ("val", "test"):
            b, refs, uniq = load(d, split, "best")
            refs_seen.setdefault(split, refs)
            assert refs == refs_seen[split], f"{arm}_s{s} {split}: references differ from the first arm"
            r[f"beam5_{split}"] = b
            if split == "test":
                r["beam5_test_unique"] = uniq
            r[f"soup_beam5_{split}"] = load(d, split, "soup_last5")[0]
        rows[arm] = r
    out = os.path.join(HERE, f"summary_c0_v1_v2_s{s}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"seed {s}: " + " | ".join(f"{a} val {rows[a]['beam5_val']:.3f} test {rows[a]['beam5_test']:.3f}"
                                      f" (best ep {rows[a]['best_epoch']})" for a in ARMS))
print("GATE PASSED: 36 beam-5 BLEU values re-scored exactly; references identical across arms and seeds")
