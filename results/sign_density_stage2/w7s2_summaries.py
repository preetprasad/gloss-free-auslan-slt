"""Wave 7 Stage 2 (job 66644: A, D, DL x seeds 303/404) per-seed summaries, built locally with the same fields as _w5lib.sh::w4summary.

Gates (all asserts, run before anything is written):
  * every run has DONE, and progress.json says status done, epoch == epochs == 40, 40 history rows;
  * every stored beam-5 BLEU (val/test x best/soup_last5) re-scores from its saved hypotheses through slt/metrics.py
    (detokenise both sides, sacreBLEU 13a) to 1e-6, and its stored unique-hypothesis count matches;
  * each dump was decoded on the official manifest of its split, has 800 hypotheses and 800 references, and its
    references equal the English column of manifests/{val,test}.tsv (copied from Kaya; md5 = real_data_learning_curve copies),
    so they are identical across arms and seeds.
Then writes summary_s{seed}.json (arms A, D, DL); manifests/ are the Stage-1 copies, md5-identical to Kaya.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

RUNS = os.path.join(HERE, "runs")
KEY = "tokenize=13a;detok=True"
ARMS, SEEDS = ("A", "D", "DL"), (303, 404)
REFS = {s: [l.rstrip("\n").split("\t", 1)[1] for l in open(os.path.join(HERE, "manifests", f"{s}.tsv")) if "\t" in l]
        for s in ("val", "test")}
assert len(REFS["val"]) == len(REFS["test"]) == 800


def load(d, split, ck):
    j = json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))
    assert j["manifest"].endswith(f"/manifests/{split}.tsv"), (d, split, ck, j["manifest"])
    hyps, refs = j["beam_hyps"], j["refs"]
    assert len(hyps) == len(refs) == 800, (d, split, ck, len(hyps), len(refs))
    assert refs == REFS[split], f"{d} {split} {ck}: references differ from the official manifest"
    stored = j["beam_result"]["metrics"][KEY]
    redo = corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs], "13a")
    assert abs(redo - stored) < 1e-6, (d, split, ck, stored, redo)
    assert j["beam_result"]["unique_hypotheses"] == len(set(hyps)), (d, split, ck)
    return stored, len(set(hyps))


n = 0
for s in SEEDS:
    rows = {}
    for arm in ARMS:
        d = os.path.join(RUNS, f"{arm}_s{s}")
        assert os.path.exists(os.path.join(d, "DONE")), d
        p = json.load(open(os.path.join(d, "progress.json")))
        assert p["status"] == "done" and p["epoch"] == p["epochs"] == 40 and len(p["history"]) == 40, (d, p["epoch"])
        r = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                 greedy_test=p.get("test", {}).get("bleu4", float("nan")))
        for split in ("val", "test"):
            r[f"beam5_{split}"], uniq = load(d, split, "best")
            if split == "test":
                r["beam5_test_unique"] = uniq
            r[f"soup_beam5_{split}"] = load(d, split, "soup_last5")[0]
            n += 2
        rows[arm] = r
    json.dump(rows, open(os.path.join(HERE, f"summary_s{s}.json"), "w"), indent=1)
    print(f"seed {s}: " + " | ".join(f"{a} val {rows[a]['beam5_val']:.3f} test {rows[a]['beam5_test']:.3f}"
                                      f" (best ep {rows[a]['best_epoch']})" for a in ARMS))
assert n == 24  # 3 arms x 2 seeds x 4 dumps
print(f"GATE PASSED: {n} beam-5 BLEU values re-scored exactly; references = official val/test manifests (800/800) "
      "in every arm and seed; every run 40/40 epochs")
