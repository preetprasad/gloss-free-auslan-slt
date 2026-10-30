#!/bin/bash -l
# WAVE 5 step 4 (CPU, afterany the training array) -- per-seed summaries and the Stage-1 seed-level decisions.
# Readouts only; the decisions are read against results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md section 6.
# Only arms carrying the DONE marker (criterion decodes present, G-transfer passed) are read.
#SBATCH --job-name=w5-summary
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; RUNS=$G/wave5/runs; TRAIN=$G/scratch_backup/dwpose/manifests/train.tsv
cd "$G/wave5/code"
for s in 42 101 202; do python - "$RUNS" "$s" "$TRAIN" C P25 P100 P200 P300 PJ <<'PYSUM'
import json, os, sys
from collections import Counter
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from slt.metrics import corpus_bleu4, detokenize
from content_failure_analysis import content
from stitch_lexicon import STOP
runs, s, train_tsv, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
KEY = "tokenize=13a;detok=True"
norm = lambda t: " ".join(t.lower().split())
train = {norm(l.split("\t", 1)[1]) for l in open(train_tsv) if "\t" in l}
def b5(d, split, ck):
    try: return json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))["beam_result"]["metrics"][KEY]
    except Exception: return float("nan")
def extra(d):   # prereg section 5: unseen-570 BLEU, pooled content-word precision / recall (stitch_lexicon.STOP)
    j = json.load(open(os.path.join(d, "beam5_test_best.json")))
    H, R = j["beam_hyps"], j["refs"]
    un = [i for i, r in enumerate(R) if norm(r) not in train]
    hit = nh = nr = 0
    for h, r in zip(H, R):
        ch, cr = content(h, STOP), content(r, STOP)
        hit += sum((Counter(ch) & Counter(cr)).values()); nh += len(ch); nr += len(cr)
    return dict(unseen_n=len(un),
                beam5_test_unseen=corpus_bleu4([detokenize(H[i]) for i in un], [detokenize(R[i]) for i in un], "13a"),
                content_precision=100 * hit / max(nh, 1), content_recall=100 * hit / max(nr, 1))
rows = {}
for arm in arms:
    d = os.path.join(runs, f"{arm}_s{s}")
    if not os.path.exists(os.path.join(d, "DONE")):
        print(f"{arm} s{s}: NOT DONE (task failed or unfinished) -- excluded"); continue
    p = json.load(open(os.path.join(d, "progress.json")))
    rows[arm] = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                     greedy_test=p.get("test", {}).get("bleu4", float("nan")),
                     beam5_val=b5(d, "val", "best"), beam5_test=b5(d, "test", "best"),
                     soup_beam5_val=b5(d, "val", "soup_last5"), soup_beam5_test=b5(d, "test", "soup_last5"),
                     **extra(d))
    print(arm, s, json.dumps(rows[arm]))
json.dump(rows, open(os.path.join(runs, f"summary_s{s}.json"), "w"), indent=1)
PYSUM
done
S=("$RUNS"/summary_s42.json "$RUNS"/summary_s101.json "$RUNS"/summary_s202.json)
dec () { echo "--- $2 - $1 ---"; python tools/seed_decision.py --control "$1" --arm "$2" "${S[@]}" || true; }
echo "=== Q1 (Stage-1 futility, family arms): each P vs C ==="
for arm in P25 P100 P200 P300; do dec C $arm; done
echo "=== compute control: PJ vs C (descriptive), and each P vs PJ (attribution to the corpus) ==="
dec C PJ; for arm in P25 P100 P200 P300; do dec PJ $arm; done
echo "=== Q2 at fixed compute: P300 vs P25 and P300 vs P100 ==="
dec P25 P300; dec P100 P300
echo "W5_SUMMARY_DONE"
