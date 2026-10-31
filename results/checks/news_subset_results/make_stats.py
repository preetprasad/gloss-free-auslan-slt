"""Prep (run once in the private tree, not by make): News decodes -> per-sentence BLEU sufficient statistics.
News references are not restored by `make data`, so none are shipped; n-gram match/total counts and lengths carry the corpus BLEU.
Reads results/news_decoding_length_check/ (beam-5 runs of jobs 16398/7950 copied, LP sweeps of job 27382, nSA validation sweeps and
the 5-member ensemble of job 21744). Writes data/stats.json."""
import json, re, sys
from pathlib import Path
import sacrebleu
from sacrebleu.metrics import BLEU
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize
B = ROOT / "results/news_decoding_length_check"
LPS = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "2.0"]
SEEDS = [42, 101, 202, 303, 404]
bl = BLEU()

def tot(h, r):
    st = bl._extract_corpus_statistics([detokenize(x) for x in h], [[detokenize(x) for x in r]])
    return [sum(int(s[i]) for s in st) for i in range(10)]

def sweep(d):
    log = {}
    for l in open(str(d) + ".log"):
        m = re.match(r"\s+LP=([0-9.]+)\s+BLEU-4=\s*([0-9.]+)", l)
        if m: log[m.group(1)] = float(m.group(2))
    out = {}
    for lp in LPS:
        rs = sorted((json.loads(l) for l in open(d / f"hyps_LP={lp}.jsonl")), key=lambda r: r["idx"])
        t = tot([r["hyp_beam"] for r in rs], [r["ref"] for r in rs])
        assert abs(bl._compute_score_from_stats(t).score - sacrebleu.corpus_bleu([detokenize(r["hyp_beam"]) for r in rs], [[detokenize(r["ref"]) for r in rs]]).score) < 1e-9
        out[lp] = {"tot": t, "log": log[lp]}
    return out

D = {"runs": {}, "sweeps": {}}
for arm in ("nA", "nSA"):
    for i, s in enumerate(SEEDS):
        run = B / "runs_news" / f"{arm}_s{s}"
        for sp in ("val", "test"):
            d = json.load(open(run / f"beam5_{sp}_best.json"))
            rec = d["beam_result"]["metrics"]["tokenize=13a;detok=True"]
            t = tot(d["beam_hyps"], d["refs"])
            assert abs(bl._compute_score_from_stats(t).score - rec) < 1e-6
            D["runs"][f"{arm}_s{s}_{sp}"] = {"recorded": rec, "tot": t, "n": len(d["refs"]), "split": sp}
        D["sweeps"][f"{arm}_s{s}_test"] = sweep(B / "news_decode" / f"{arm}_s{s}_test")
        D["sweeps"][f"{arm}_s{s}_val"] = sweep(B / ("news_ens/s%d_val" % i if arm == "nSA" else f"news_decode/{arm}_s{s}_val"))
D["sweeps"]["ens5_val"] = sweep(B / "news_ens/ens5_val")
json.dump(D, open(Path(__file__).parent / "data/stats.json", "w"), separators=(",", ":"))
