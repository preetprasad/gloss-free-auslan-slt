"""Prep (run once in the private tree, not by make): turn saved News decodes into per-sentence BLEU sufficient statistics.
News references are not in the Communication spreadsheet that `make data` restores, so they are not shipped; the
per-sentence n-gram match/total counts and lengths (sacreBLEU 13a on detokenised text) carry the same corpus BLEU.
usage: python make_stats.py RUNS_DIR   (RUNS_DIR/<arm>_s<seed>/beam5_<split>_best.json) -> data/<arm>_s<seed>_<split>.json"""
import json, sys
from pathlib import Path
import sacrebleu
from sacrebleu.metrics import BLEU
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from slt.metrics import detokenize
R = Path(sys.argv[1]); out = Path(__file__).parent / "data"; out.mkdir(exist_ok=True)
for run in sorted(R.glob("n*_s*")):
    for sp in ("val", "test"):
        d = json.load(open(run / f"beam5_{sp}_best.json"))
        h = [detokenize(x) for x in d["beam_hyps"]]; r = [detokenize(x) for x in d["refs"]]
        b = BLEU(); st = b._extract_corpus_statistics(h, [r])
        rec = d["beam_result"]["metrics"]["tokenize=13a;detok=True"]
        assert abs(sacrebleu.corpus_bleu(h, [r]).score - rec) < 1e-6
        json.dump({"arm_seed": run.name, "split": sp, "job": open(run / "slurm_job.txt").read().split()[0],
                   "recorded_bleu": rec, "hyps": d["beam_hyps"], "stats": [[int(v) for v in s] for s in st]},
                  open(out / f"{run.name}_{sp}.json", "w"), separators=(",", ":"))
